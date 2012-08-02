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
	m_AvgAntiirradiance = 0.f;
	m_AvgIrradiance = 0.f;
}

CAVPLImportanceSampling::~CAVPLImportanceSampling()
{
}

void CAVPLImportanceSampling::SetNumberOfSceneSamples(uint num)
{
	m_NumSceneSamples = num;
	m_OneOverNumSceneSamples = 1.f / float(num);
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
	
	m_AvgIrradiance = Luminance(irradiance) / float(width * height);
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

	m_AvgAntiirradiance = Luminance(antiirradiance) / float(width * height);
}

bool CAVPLImportanceSampling::EvaluateAVPLImportance(const AVPL& avpl, float* scale)
{
	if(m_AvgAntiirradiance == 0 || m_AvgIrradiance == 0)
	{
		*scale = 1.f;
		return true;
	}
	
	glm::vec3 irradiance = glm::vec3(0.f);
	glm::vec3 antiirradiance = glm::vec3(0.f);
	for(uint i = 0; i < m_NumSceneSamples; ++i)
	{
		irradiance += avpl.GetIrradiance(m_SceneSamples[i]);
		antiirradiance += avpl.GetAntiirradiance(m_SceneSamples[i], m_ConeFactor);
	}
	irradiance *= m_OneOverNumSceneSamples;
	antiirradiance *= m_OneOverNumSceneSamples;
	
	const float frac_irr = Luminance(irradiance) / m_AvgIrradiance;
	const float frac_antiirr = Luminance(irradiance) / m_AvgAntiirradiance;

	if(frac_irr >= 1.f || frac_antiirr)
	{
		*scale = 1.f;
		return true;
	}

	const float p_accept_irr = std::min(1.f, frac_irr + m_Epsilon);
	const float p_accept_antiirr = std::min(1.f, frac_antiirr + m_Epsilon);
	
	if(Rand01() < p_accept_irr || Rand01() < p_accept_antiirr)
	{
		*scale = 1.f / (p_accept_irr + p_accept_antiirr);
		return true;
	}
		
	return false;
}

bool CAVPLImportanceSampling::EvaluateAVPLAntiintensityImportance(const AVPL& avpl, float* scale)
{
	glm::vec3 antiirradiance = glm::vec3(0.f);
	for(uint i = 0; i < m_SceneSamples.size(); ++i)
	{
		antiirradiance += avpl.GetAntiirradiance(m_SceneSamples[i], m_ConeFactor);
	}
	antiirradiance *= m_OneOverNumSceneSamples;
			
	const float A = Luminance(antiirradiance);
	if(m_AvgAntiirradiance == 0 || A > 0.f)
	{
		*scale = 1.f;
		return true;
	}
		
	const float p_accept = std::min(1.f, m_Alpha + m_Epsilon);
	if(Rand01() < p_accept)
	{
		*scale = 1.f / p_accept;
		return true;
	}

	return false;
}

void CAVPLImportanceSampling::CreateSceneSamples()
{
	m_Alpha = m_pConfManager->GetConfVars()->IrradAntiirradWeight;
	m_Epsilon = m_pConfManager->GetConfVars()->AcceptProbabEpsilon;
	m_ConeFactor = float(m_pConfManager->GetConfVars()->ConeFactorIS);

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

		// no scene parts visible?
		if(m_SceneSamples.size() == 0)
			break;
	}

	eye_rays.clear();
	samples.clear();
}
