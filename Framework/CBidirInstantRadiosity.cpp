#include "CBidirInstantRadiosity.h"

#include "AVPL.h"
#include "Scene.h"
#include "CCamera.h"
#include "CConfigManager.h"
#include "Utils\Util.h"
#include "CTimer.h"

#include <iostream>
#include <assert.h>

const float EPSILON = 0.05f;

CBidirInstantRadiosity::CBidirInstantRadiosity(Scene* pScene, CConfigManager* pConfManager)
	: m_pScene(pScene), m_pConfManager(pConfManager)
{
}

CBidirInstantRadiosity::~CBidirInstantRadiosity()
{
}

void CBidirInstantRadiosity::CreatePaths(std::vector<AVPL>& avpls, int& numPaths, bool profile)
{
	return;
	
	m_NumVisibles = m_pConfManager->GetConfVars()->NumSamplesForPESS;
	m_Visibles.clear();
	CreateVisibles(m_Visibles, m_NumVisibles);
	
	CreateSceneSamples(profile, SS);
	CreateSceneSamples(profile, ASS);

	for(int i = 0; i < m_pConfManager->GetConfVars()->NumPathsPerFrame; ++i)
	{
		int before = (int)avpls.size();
		CreatePath(avpls, profile);
		if(avpls.size() != before)
			numPaths++;
		else
		{
			--i;
		}
	}
}

void CBidirInstantRadiosity::CreatePath(std::vector<AVPL>& avpls, bool profile)
{
	CreateSceneSamples(profile, SS);

	AVPL pred, succ;
		
	// create new primary light on light source
	m_pScene->CreatePrimaryAVPL(&pred);
	
	float scaleSS = 1.f/float(m_pConfManager->GetConfVars()->NumEyeRaysSS);
	float scaleASS = 1.f/float(m_pConfManager->GetConfVars()->NumEyeRaysASS);

	ConnectToSceneSamples(pred, avpls, scaleSS);
	ConnectToAntiSceneSamples(pred, avpls, scaleASS);
		
	//avpls.push_back(pred);
}

void CBidirInstantRadiosity::ConnectToSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale)
{
	float scaleASS = 1.f/float(m_pConfManager->GetConfVars()->NumEyeRaysASS);
	for(uint i = 0; i < m_SceneSamples.size(); ++i)
	{
		if(Visible(avpl, m_SceneSamples[i], SS))
		{
			AVPL newAvpl;
			if(CreateAVPLAtSceneSample(m_SceneSamples[i], avpl, &newAvpl, SS))
			{
				ConnectToAntiSceneSamples(newAvpl, avpls, scaleASS * scale);
				newAvpl.ScaleIncidentRadiance(scale);
				avpls.push_back(newAvpl);
			}
		}
	}
}

void CBidirInstantRadiosity::ConnectToAntiSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale)
{
	for(int i = 0; i < m_AntiSceneSamples.size(); ++i)
	{
		glm::vec3 direction = glm::normalize(m_AntiSceneSamples[i].position - avpl.GetPosition());
		if(glm::dot(m_AntiSceneSamples[i].normal, -direction) > 0.f && glm::dot(avpl.GetOrientation(), direction) > 0.f)
		{
			AVPL newAvpl;
			if(CreateAVPLAtSceneSample(m_AntiSceneSamples[i], avpl, &newAvpl, ASS))
			{
				newAvpl.ScaleAntiradiance(scale);
				avpls.push_back(newAvpl);
			}
		}
	}
}

bool CBidirInstantRadiosity::Visible(const AVPL& from_avpl, const SceneSample& ss, SceneSampleType ss_type)
{
	SceneSample from;
	from.position = from_avpl.GetPosition();
	from.normal = from_avpl.GetOrientation();
	
	return Visible(from, ss, ss_type);
}

bool CBidirInstantRadiosity::Visible(const SceneSample& from_param, const SceneSample& ss_param, SceneSampleType ss_type)
{
	SceneSample from = from_param;
	SceneSample ss = ss_param;

	glm::vec3 direction = glm::normalize(ss.position - from.position);
	
	// flip normal if the scene sample ss is an anti scene sample
	if(ss_type == ASS) ss.normal *= -1.f;
	
	if(glm::dot(ss.normal, -direction) <= 0.f || glm::dot(from.normal, direction) <= 0.f)
		return false;

	glm::vec3 ray_origin = from.position + EPSILON * direction;
	Ray r(ray_origin, direction);
	float dist = glm::length(ss.position - ray_origin);

	float t = 0.f;
	Intersection intersection;
	const CPrimitive::IsectMode isect_mode = ss_type == SS ? CPrimitive::FRONT_FACE : CPrimitive::BACK_FACE;
	bool isect = m_pScene->IntersectRaySceneSimple(r, &t, &intersection, isect_mode);
			
	const float big = std::max(dist, t);
	const float small = std::min(dist, t);

	const float temp = small/big;

	if(isect && temp > 0.99f ) {
		return true;
	}

	return false;
}

bool CBidirInstantRadiosity::CreateAVPLAtSceneSample(const SceneSample& ss, const AVPL& pred, AVPL* newAvpl, SceneSampleType ss_type)
{
	glm::vec3 direction = glm::normalize(ss.position - pred.GetPosition());
	
	const float dist = glm::length(ss.position - pred.GetPosition());
	const float cos_theta_ss = glm::dot(ss.normal, -direction);
	const float cos_theta_pred = glm::dot(pred.GetOrientation(), direction);

	if(cos_theta_ss < 0.f || cos_theta_pred < 0.f)
	{
		std::cout << "that should not happen!" << std::endl;

		newAvpl = 0;
		return false;
	}
	
	if(cos_theta_ss == 0.f || cos_theta_pred == 0.f)
	{
		newAvpl = 0;
		return false;
	}

	const float G = (cos_theta_pred * cos_theta_ss) / (dist * dist);

	glm::vec3 pred_pos = pred.GetPosition();

	// gather information for the new VPL
	glm::vec3 pos = ss.position;
	glm::vec3 norm = ss.normal;
	
	glm::vec3 contrib = glm::vec3(0.f);
	if(ss.pdf > 0.f && ss_type == SS)
		contrib = G/ss.pdf * pred.GetRadiance(direction);
	
	//const float cone_min = m_pConfManager->GetConfVars()->ClampCone ? PI / (PI/2.f - acos(glm::dot(norm, -direction))) : 0.f;
	//const float coneFactor = std::max(cone_min, m_pConfManager->GetConfVars()->ConeFactor);
	
	const float coneFactor = m_pConfManager->GetConfVars()->ConeFactor;
	const float area = 2 * PI * ( 1 - cos(PI/coneFactor) );
	
	glm::vec3 antiintensity = glm::vec3(0.f);
	if(ss.pdf > 0.f && ss_type == ASS)
		antiintensity = G/ss.pdf * pred.GetRadiance(direction) * 1.f/area;
	
	AVPL avpl(pos, norm, contrib, antiintensity, direction, coneFactor, pred.GetBounce() + 1, ss.materialIndex, m_pScene->GetMaterialBuffer(), m_pConfManager);
	*newAvpl = avpl;
	return true;
}

void CBidirInstantRadiosity::CreateSceneSamples(bool profile, SceneSampleType ss_type)
{
	CTimer timer(CTimer::CPU);

	int N = 0;
	CPrimitive::IsectMode isect_mode;
	
	std::vector<SceneSample>* sceneSamples;
	if(ss_type == SS)
	{
		N = m_pConfManager->GetConfVars()->NumEyeRaysSS;
		isect_mode = CPrimitive::FRONT_FACE;
		sceneSamples = &m_SceneSamples;
	}
	else
	{
		N = m_pConfManager->GetConfVars()->NumEyeRaysASS;
		isect_mode = CPrimitive::BACK_FACE;
		sceneSamples = &m_AntiSceneSamples;
	}
	
	sceneSamples->clear();
	sceneSamples->reserve(N);

	std::vector<SceneSample> tempVisibles;
	tempVisibles.reserve(N);
	
	if(profile) timer.Start();
	
	CreateVisibles(tempVisibles, N);

	if(tempVisibles.size() != N)
	{
		std::cout << "cannot happen in empty closed CB" << std::endl;
		return;
	}
		
	if(tempVisibles.size() == 0)
	{
		std::cout << "no part of scene visible" << std::endl;
		return;
	}

	if(profile) timer.Stop("create directly visible scene samples");

	if(profile) timer.Start();
	// determine scene samples
	for(int i = 0; i < tempVisibles.size(); ++i)
	{		
		float t = 0.f;
		Intersection intersection;
		bool isect = false;

		// get random direction
		float pdf = 0.f;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(tempVisibles[i].normal, Rand01(), Rand01(), pdf, 1);
		Ray r(tempVisibles[i].position + 0.05f * direction, direction);
		
		// create scene sample		
		isect = m_pScene->IntersectRayScene(r, &t, &intersection, isect_mode);
		
		if(isect) {
			SceneSample ss(intersection);
			ss.pdf = Probability(tempVisibles[i], ss, ss_type);

			if(ss.pdf == 0.0) std::cout << "probability estimate is 0" << std::endl;

			if(m_Visibles.size() != m_NumVisibles)
			{
				std::cout << "cannot happen in empty closed CB" << std::endl;
			}
			
			sceneSamples->push_back(ss);	
		}
	}

	tempVisibles.clear();

	if(profile) timer.Stop("create second bounce scene samples");
	if(profile) std::cout << "number of scene samples: " << m_SceneSamples.size() << std::endl;
}

void CBidirInstantRadiosity::CreateVisibles(std::vector<SceneSample>& sceneSamples, int numVisibles)
{
	// non stratified samples for rest
	for(int i = 0; i < numVisibles; ++i)
	{
		Ray e = m_pScene->GetCamera()->GetEyeRay();
		
		float t = 0.f;
		Intersection intersection;
		bool isect = m_pScene->IntersectRayScene(e, &t, &intersection, CPrimitive::FRONT_FACE);
	
		if(isect) {
			SceneSample ss(intersection);
			glm::vec3 v = m_pScene->GetCamera()->GetViewDirection();
			const float cos_theta_x0 = glm::dot(v, e.d);
			const float cos_theta_x1 = glm::dot(ss.normal, -e.d);
			ss.pdf = cos_theta_x1 / cos_theta_x0;
			sceneSamples.push_back(ss);
		}
	}
}

float CBidirInstantRadiosity::Probability(const SceneSample& from, const SceneSample& to, SceneSampleType ss_type)
{
	CPrimitive::IsectMode isect_mode = ss_type == SS ? CPrimitive::FRONT_FACE :
		CPrimitive::BACK_FACE;

	float p = 0;
	
	if(Visible(from, to, ss_type))
	{
		const glm::vec3 from_to = glm::normalize(to.position - from.position);
		const float dist = glm::length(to.position - from.position);
		
		const float sign = ss_type == SS ? -1.f : 1.f;
		const float cos_theta = glm::dot(to.normal, sign * from_to);
		
		if(cos_theta <= 0.f)
		{
			std::cout << "cannot happen in empty closed CB" << std::endl;
			return p;			
		}
	
		float pdf = 0.f;
		GetRandomSampleDirectionProbability(from.normal, from_to, pdf, 1);
		if(pdf <= 0.f) 
		{
			std::cout << "cannot happen in empty closed CB" << std::endl;
			return p;		
		}

		p = pdf * cos_theta / (dist * dist);
	}
	return p;
}