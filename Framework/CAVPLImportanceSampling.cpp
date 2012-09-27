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
		
	m_RadianceContrib = 0.0f;
	m_NumContribSamples = 1;
}

CAVPLImportanceSampling::~CAVPLImportanceSampling()
{
}

void CAVPLImportanceSampling::ImportanceSampling(const std::vector<AVPL>& avpls, std::vector<AVPL>& result)
{
	for(int i = 0; i < avpls.size(); ++i)
	{
		float scale = 0.f;
		AVPL a = avpls[i];
		if(EvaluateAVPLImportance(a, &scale))
		{
			a.ScaleIncidentRadiance(scale);
			a.ScaleAntiradiance(scale);
			result.push_back(a);
		}
	}
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
	
	m_AvgIrradiance = Average(irradiance) / float(width * height);
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

	m_AvgAntiirradiance = Average(antiirradiance) / float(width * height);
}

glm::vec3 CAVPLImportanceSampling::f(const AVPL& avpl, const SceneSample& ss)
{
	const glm::vec3 w_i = glm::normalize(avpl.GetPosition() - ss.position);
	const glm::vec3 w_o = glm::normalize(m_pScene->GetCamera()->GetPosition() - ss.position);
	const glm::vec3 n = ss.normal;
	
	glm::vec4 BRDF = glm::vec4(0.f);
	
	if(glm::dot(w_i, n) >= 0.f && glm::dot(w_o, n) >= 0.f)
		BRDF = Phong(w_i, w_o, n, m_pScene->GetMaterial(ss));

	return glm::vec3(BRDF);
}

glm::vec3 CAVPLImportanceSampling::f_light(const AVPL& avpl, const SceneSample& ss)
{
	glm::vec3 direction = glm::normalize(ss.position - avpl.GetPosition());
	glm::vec4 BRDF_light = glm::vec4(0.f);
	
	if( glm::dot(direction, ss.normal) >= 0.f &&
		glm::dot(-avpl.GetDirection(), ss.normal) >= 0.f &&
		glm::length(avpl.GetDirection()) != 0.f)
	{
		BRDF_light = Phong(-avpl.GetDirection(), direction, ss.normal, m_pScene->GetMaterial(ss));
	}

	// check for light source AVPL
	if(glm::length(avpl.GetDirection()) == 0.f)
		BRDF_light = glm::vec4(1.f);

	return glm::vec3(BRDF_light);
}

bool CAVPLImportanceSampling::EvaluateAVPLImportance0(AVPL& avpl, float* scale)
{
	if(m_AvgAntiirradiance == 0 || m_AvgIrradiance == 0)
	{
		*scale = 1.f;
		return true;
	}
	
	glm::vec3 radianceContrib = glm::vec3(0.f);
	glm::vec3 antiradianceContrib = glm::vec3(0.f);
	for(uint i = 0; i < m_NumSceneSamples; ++i)
	{
		glm::vec3 BRDF = f(avpl, m_SceneSamples[i]);
		radianceContrib += BRDF * avpl.GetIrradiance(m_SceneSamples[i]);
		antiradianceContrib += BRDF * avpl.GetAntiirradiance(m_SceneSamples[i]);
	}
	radianceContrib /= float(m_NumSceneSamples);
	antiradianceContrib /= float(m_NumSceneSamples);
	
	const float frac_irr = Average(radianceContrib) / m_AvgIrradiance;
	const float frac_antiirr = Average(antiradianceContrib) / m_AvgAntiirradiance;

	if(frac_irr >= 1.f || frac_antiirr >= 1.f)
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

bool CAVPLImportanceSampling::EvaluateAVPLImportance1(AVPL& avpl, float* scale)
{
	if(m_AvgAntiirradiance == 0 || m_AvgIrradiance == 0)
	{
		*scale = 1.f;
		return true;
	}

	glm::vec3 radianceContrib = glm::vec3(0.f);
	glm::vec3 antiradianceContrib = glm::vec3(0.f);
	for(uint i = 0; i < m_NumSceneSamples; ++i)
	{
		glm::vec3 BRDF = f(avpl, m_SceneSamples[i]);
		radianceContrib += BRDF * avpl.GetIrradiance(m_SceneSamples[i]);
		antiradianceContrib += BRDF * avpl.GetAntiirradiance(m_SceneSamples[i]);
	}
	radianceContrib /= float(m_NumSceneSamples);
	antiradianceContrib /= float(m_NumSceneSamples);

	if(glm::length(antiradianceContrib) > 0.1f)
	{
		*scale = 1.f;
		return true;
	}
	
	const float frac_irr = Average(radianceContrib) / m_AvgIrradiance;
	
	if(frac_irr >= 1.f)
	{
		*scale = 1.f;
		return true;
	}

	const float p_accept = std::min(1.f, frac_irr + m_Epsilon);
	const float p = Rand01();
	if(p < p_accept)
	{
		*scale = 1.f / (p_accept);
		return true;
	}
		
	return false;
}

bool CAVPLImportanceSampling::EvaluateAVPLImportance(AVPL& avpl, float* scale)
{
	*scale = 1.f;

	if(m_AvgIrradiance == 0) return true;
	
	glm::vec3 radianceContrib = glm::vec3(0.f);
	for(uint i = 0; i < m_NumSceneSamples; ++i)
	{
		const glm::vec3 w_A = avpl.GetDirection();
		const glm::vec3 w = glm::normalize(m_SceneSamples[i].position - avpl.GetPosition());
		const float theta = acos(clamp(glm::dot(w, w_A), 0, 1));
		
		if(theta < PI/10.f && glm::dot(m_SceneSamples[i].normal, -w_A) >= 0.f)
			return true;
		
		glm::vec3 BRDF = f(avpl, m_SceneSamples[i]);
		radianceContrib += BRDF * avpl.GetIrradiance(m_SceneSamples[i]);
	}
	radianceContrib /= float(m_NumSceneSamples);
	
	const float frac_irr = Average(radianceContrib) / m_AvgIrradiance;
	
	if(frac_irr >= 1.f) return true;

	const float p_accept_irr = std::min(1.f, frac_irr + m_Epsilon);
	
	if(Rand01() < p_accept_irr)
	{
		*scale = 1.f / (p_accept_irr);
		return true;
	}
		
	return false;
}

void CAVPLImportanceSampling::CreateSceneSamples()
{
	m_Alpha = m_pConfManager->GetConfVars()->IrradAntiirradWeight;
	m_Epsilon = m_pConfManager->GetConfVars()->AcceptProbabEpsilon;
	m_ConeFactor = float(m_pConfManager->GetConfVars()->ConeFactorIS);

	m_SceneSamples.clear();
	m_SceneSamples.reserve(m_NumSceneSamples);

	int k = 0;
	while(m_NumSceneSamples - m_SceneSamples.size() > 0)
	{
		Ray r = m_pScene->GetCamera()->GetEyeRay();
		
		float t = 0.f;
		Intersection intersection;
		bool isect = m_pScene->IntersectRayScene(r, &t, &intersection, CPrimitive::FRONT_FACE);
			
		if(isect) {
			SceneSample ss(intersection);
			m_SceneSamples.push_back(ss);
		}
		
		// no scene parts visible?
		k++;
		if(k > int(100 * m_NumSceneSamples))
			break;
	}
}

bool CAVPLImportanceSampling::HasAntiradianceContribution(const AVPL& avpl)
{
	for(uint i = 0; i < m_NumSceneSamples; ++i)
	{
		const glm::vec3 w_A = avpl.GetDirection();
		const glm::vec3 w = glm::normalize(m_SceneSamples[i].position - avpl.GetPosition());
		const float theta = acos(clamp(glm::dot(w, w_A), 0, 1));
		
		if(theta < PI/10.f && glm::dot(m_SceneSamples[i].normal, -w_A) >= 0.f)
			return true;
	}

	return false;
}