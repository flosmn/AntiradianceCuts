#include "CAVPLImportanceSampling.h"

#include "OGLResources\COGLTexture2D.h"
#include "AVPL.h"
#include "Scene.h"
#include "CConfigManager.h"
#include "CCamera.h"
#include "Ray.h"
#include "Utils\Util.h"

#include <iostream>

CAVPLImportanceSampling::CAVPLImportanceSampling(Scene* pScene, CConfigManager* pConfManager)
	: m_pScene(pScene), m_pConfManager(pConfManager)
{
	m_AvgAntiirradiance = glm::vec3(0.f);
	m_AvgIrradiance = glm::vec3(0.f);
}

CAVPLImportanceSampling::~CAVPLImportanceSampling()
{
}

void CAVPLImportanceSampling::SetNumberOfSceneSamples(uint num)
{
	m_NumSceneSamples = num;
}

void CAVPLImportanceSampling::UpdateCurrentIrradiance(COGLTexture2D* pTexture)
{
	uint width = pTexture->GetWidth();
	uint height = pTexture->GetHeight();
	glm::vec4* pData = new glm::vec4[width * height];
	pTexture->GetPixelData(pData);
	
	glm::vec3 irradiance = glm::vec3(0.f);
	for (uint x = 0; x < width; ++x) {
		for (uint y = 0; y < height; ++y) {
			irradiance += glm::vec3(pData[y * width + x]); 
		}
	}

	delete [] pData;

	m_AvgIrradiance = irradiance / float(width * height);
}

void CAVPLImportanceSampling::UpdateCurrentAntiirradiance(COGLTexture2D* pTexture)
{
	uint width = pTexture->GetWidth();
	uint height = pTexture->GetHeight();
	glm::vec4* pData = new glm::vec4[width * height];
	pTexture->GetPixelData(pData);
	
	glm::vec3 antiirradiance = glm::vec3(0.f);
	for (uint x = 0; x < width; ++x) {
		for (uint y = 0; y < height; ++y) {
			antiirradiance += glm::vec3(pData[y * width + x]); 
		}
	}

	delete [] pData;

	m_AvgAntiirradiance = antiirradiance / float(width * height);
}

bool CAVPLImportanceSampling::EvaluateAVPLImportance(AVPL* avpl, float* scale)
{
	const int ConeFactorScale = m_pConfManager->GetConfVars()->ConeFactorScale;

	glm::vec3 irradiance = glm::vec3(0.f);
	glm::vec3 antiirradiance = glm::vec3(0.f);
	for(uint i = 0; i < m_NumSceneSamples; ++i)
	{
		irradiance += avpl->GetIrradiance(m_SceneSamples[i]);
		antiirradiance += avpl->GetAntiirradiance(m_SceneSamples[i], ConeFactorScale);
	}
	irradiance *= 1.f/float(m_NumSceneSamples);
	antiirradiance *= 1.f/float(m_NumSceneSamples);
			
	const float alpha = m_pConfManager->GetConfVars()->IrradAntiirradWeight;
	const float epsilon = m_pConfManager->GetConfVars()->AcceptProbabEpsilon;
	
	const float avgIrrad = Luminance(m_AvgIrradiance);
	const float avgAntiirrad = Luminance(m_AvgAntiirradiance);
	if(avgIrrad == 0 || avgAntiirrad == 0)
	{
		*scale = 1.f;
		return true;
	}

	if(Luminance(irradiance) >= avgIrrad || Luminance(antiirradiance) >= avgAntiirrad)
	{
		*scale = 1.f;
		return true;
	}

	const float p_accept_irr = std::min(1.f, Luminance(irradiance) / avgIrrad + epsilon);
	const float p_accept_antiirr = std::min(1.f, Luminance(antiirradiance) / avgAntiirrad + epsilon);

	bool accept = false;
	float u_irr = Rand01();
	float u_antiirr = Rand01();

	if(u_irr < p_accept_irr || u_antiirr < p_accept_antiirr)
	{
		*scale = 1.f / (p_accept_irr + p_accept_antiirr);
		return true;
	}
		
	return false;
}

bool CAVPLImportanceSampling::EvaluateAVPLAntiintensityImportance(AVPL* avpl, float* scale)
{
	const int ConeFactorScale = m_pConfManager->GetConfVars()->ConeFactorScale;

	glm::vec3 antiirradiance = glm::vec3(0.f);
	for(uint i = 0; i < m_SceneSamples.size(); ++i)
	{
		antiirradiance += avpl->GetAntiirradiance(m_SceneSamples[i], float(ConeFactorScale));
	}
	antiirradiance *= 1.f/float(m_NumSceneSamples);
		
	const float epsilon = m_pConfManager->GetConfVars()->AcceptProbabEpsilon;
	const float alpha = m_pConfManager->GetConfVars()->IrradAntiirradWeight;	

	const float avgAntiirrad = Luminance(m_AvgAntiirradiance);
	const float A = Luminance(antiirradiance);
	if(avgAntiirrad == 0 || A > 0.f)
	{
		*scale = 1.f;
		return true;
	}
		
	const float p_accept = std::min(1.f, alpha + epsilon);

	bool accept = false;
	float u = Rand01();
	
	if(u < p_accept)
	{
		*scale = 1.f / p_accept;
		return true;
	}

	return false;
}

void CAVPLImportanceSampling::CreateSceneSamples()
{
	std::vector<Ray> eye_rays;
	std::vector<glm::vec2> samples;
	
	m_SceneSamples.clear();
	m_SceneSamples.reserve(m_NumSceneSamples);

	while(m_NumSceneSamples - m_SceneSamples.size() > 0)
	{
		m_pScene->GetCamera()->GetEyeRays(eye_rays, samples, m_NumSceneSamples);
		
		for(int i = 0; i < eye_rays.size() && m_SceneSamples.size() < m_NumSceneSamples; ++i)
		{		
			float t = 0.f;
			Intersection intersection;
			bool isect = m_pScene->IntersectRayScene(eye_rays[i], &t, &intersection);
			
			if(isect) {
				SceneSample ss;
				ss.position = intersection.GetPosition();
				ss.normal = intersection.GetNormal();
				ss.material = intersection.GetMaterial();
				m_SceneSamples.push_back(ss);			
			}
		}
	}

	eye_rays.clear();
	samples.clear();
}
