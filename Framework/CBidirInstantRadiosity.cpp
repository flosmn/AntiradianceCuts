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

void CBidirInstantRadiosity::CreatePaths(std::vector<AVPL>& avpls, int& numCreatedPaths)
{
}

void CBidirInstantRadiosity::CreatePath(std::vector<AVPL>& avpls, int& numCreatedPaths)
{
	CreateSceneSamples(false);

	numCreatedPaths = 0;
	int currentBounce = 0;
	
	AVPL pred, succ;
		
	// create new primary light on light source
	m_pScene->CreateAVPL(0, &pred);
	
	float scale = 1.f/float(m_Visibles.size());

	ConnectToSceneSamples(pred, avpls, scale);
	ConnectToAntiSceneSamples(pred, avpls, scale);

	numCreatedPaths++;
	avpls.push_back(pred);
}

void CBidirInstantRadiosity::ConnectToSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale)
{
	for(int i = 0; i < m_SceneSamples.size(); ++i)
	{
		if(Visible(m_SceneSamples[i], avpl, CPrimitive::FRONT_FACE))
		{
			AVPL newAvpl;
			if(CreateAVPLAtSceneSample(m_SceneSamples[i], avpl, &newAvpl))
			{
				ConnectToAntiSceneSamples(newAvpl, avpls, scale * scale);
				newAvpl.SetIntensity(scale * newAvpl.GetMaxIntensity());
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
			if(CreateAVPLAtAntiSceneSample(m_AntiSceneSamples[i], avpl, &newAvpl))
			{
				newAvpl.SetAntiintensity(scale * newAvpl.GetMaxAntiintensity());
				avpls.push_back(newAvpl);
			}
		}
	}
}

bool CBidirInstantRadiosity::Visible(const SceneSample& ss, const AVPL& avpl, CPrimitive::IsectMode isect_mode)
{
	glm::vec3 direction = glm::normalize(ss.position - avpl.GetPosition());
		
	if(isect_mode == CPrimitive::FRONT_FACE && (glm::dot(ss.normal, -direction) < 0.f || glm::dot(avpl.GetOrientation(), direction) < 0.f))
		return false;

	if(isect_mode == CPrimitive::BACK_FACE && (glm::dot(ss.normal, direction) < 0.f || glm::dot(avpl.GetOrientation(), direction) < 0.f))
		return false;
	
	glm::vec3 ray_origin = avpl.GetPosition() + EPSILON * direction;
	Ray r(ray_origin, direction);
	float dist = glm::length(ss.position - ray_origin);

	float t = 0.f;
	Intersection intersection;
	bool isect = m_pScene->IntersectRayScene(r, &t, &intersection, isect_mode);
	
	const float big = std::max(dist, t);
	const float small = std::min(dist, t);

	if(isect && small / big > 0.999f ) {
		return true;
	}

	return false;
}

bool CBidirInstantRadiosity::Visible(const SceneSample& ss1, const SceneSample& ss2, CPrimitive::IsectMode isect_mode)
{
	glm::vec3 direction = glm::normalize(ss1.position - ss2.position);
	
	if(isect_mode == CPrimitive::FRONT_FACE && (glm::dot(ss1.normal, -direction) < 0.f || glm::dot(ss2.normal, direction) < 0.f))
		return false;

	if(isect_mode == CPrimitive::BACK_FACE && (glm::dot(ss1.normal, direction) < 0.f || glm::dot(ss2.normal, direction) < 0.f))
		return false;
	
	glm::vec3 ray_origin = ss2.position + EPSILON * direction;
	Ray r(ray_origin, direction);
	float dist = glm::length(ss1.position - ray_origin);

	float t = 0.f;
	Intersection intersection;
	bool isect = m_pScene->IntersectRayScene(r, &t, &intersection, isect_mode);
			
	const float big = std::max(dist, t);
	const float small = std::min(dist, t);

	if(isect && small / big > 0.999f ) {
		return true;
	}

	return false;
}

bool CBidirInstantRadiosity::CreateAVPLAtSceneSample(const SceneSample& ss, const AVPL& pred, AVPL* newAvpl)
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
	glm::vec3 rho = glm::vec3(ss.material.diffuseColor);
		
	glm::vec3 intensity = rho/PI * G * 1.f/ss.pdf * pred.GetMaxIntensity();
	
	//const float cone_min = m_pConfManager->GetConfVars()->ClampCone ? PI / (PI/2.f - acos(glm::dot(norm, -direction))) : 0.f;
	//const float coneFactor = std::max(cone_min, m_pConfManager->GetConfVars()->ConeFactor);
	
	const float coneFactor = m_pConfManager->GetConfVars()->ConeFactor;
	const float area = 2 * PI * ( 1 - cos(PI/coneFactor) );
	
	glm::vec3 antiintensity = glm::vec3(0.f); //pred.GetMaxIntensity() * G * 1.f/ss.pdf * 1.f/area;
	
	AVPL avpl(pos, norm, intensity, antiintensity, direction, coneFactor, pred.GetBounce() + 1, m_pConfManager);
	*newAvpl = avpl;
	return true;
}

bool CBidirInstantRadiosity::CreateAVPLAtAntiSceneSample(const SceneSample& ss, const AVPL& pred, AVPL* newAvpl)
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
	glm::vec3 rho = glm::vec3(ss.material.diffuseColor);
		
	glm::vec3 intensity = glm::vec3(0.f);
	
	//const float cone_min = m_pConfManager->GetConfVars()->ClampCone ? PI / (PI/2.f - acos(glm::dot(norm, -direction))) : 0.f;
	//const float coneFactor = std::max(cone_min, m_pConfManager->GetConfVars()->ConeFactor);
	
	const float coneFactor = m_pConfManager->GetConfVars()->ConeFactor;
	const float area = 2 * PI * ( 1 - cos(PI/coneFactor) );
	
	glm::vec3 antiintensity = pred.GetMaxIntensity() * G * 1.f/ss.pdf * 1.f/area;
	
	AVPL avpl(pos, norm, intensity, antiintensity, direction, coneFactor, pred.GetBounce() + 1, m_pConfManager);
	*newAvpl = avpl;
	return true;
}

void CBidirInstantRadiosity::CreateSceneSamples(bool profile)
{
	CTimer timer(CTimer::CPU);

	std::vector<Ray> eye_rays;
	std::vector<glm::vec2> samples;
	
	int N = m_pConfManager->GetConfVars()->NumEyeRays;
	int M = m_pConfManager->GetConfVars()->NumSamplesForPE;

	m_SceneSamples.clear();
	m_AntiSceneSamples.clear();
	m_Visibles.clear();
	m_SceneSamples.reserve(N);
	m_AntiSceneSamples.reserve(N);
	m_Visibles.reserve(N);

	if(profile) timer.Start();

	// determine directly visible scene samples

	// create some samples which are used for probability estimation	
	while(m_Visibles.size() < M)
	{
		m_pScene->GetCamera()->GetEyeRays(eye_rays, samples, M);
		
		for(int i = 0; i < eye_rays.size() && m_Visibles.size() < M; ++i)
		{		
			float t = 0.f;
			Intersection intersection;
			bool isect = m_pScene->IntersectRayScene(eye_rays[i], &t, &intersection, CPrimitive::FRONT_FACE);
	
			if(isect) {
				SceneSample ss(intersection);
				ss.pdf = 0.f;
				m_Visibles.push_back(ss);
			}
		}

		// no scene parts visible?
		if(m_Visibles.size() == 0)
			break;
	}

	// create rest of the visible samples
	while(m_Visibles.size() < N)
	{
		m_pScene->GetCamera()->GetEyeRays(eye_rays, samples, N - M);
		
		for(int i = 0; i < eye_rays.size() && m_Visibles.size() < N; ++i)
		{		
			float t = 0.f;
			Intersection intersection;
			bool isect = m_pScene->IntersectRayScene(eye_rays[i], &t, &intersection, CPrimitive::FRONT_FACE);
	
			if(isect) {
				SceneSample ss(intersection);
				ss.pdf = 0.f;
				m_Visibles.push_back(ss);
			}
		}

		// no scene parts visible?
		if(m_Visibles.size() == 0)
			break;
	}

	eye_rays.clear();
	samples.clear();

	if(profile) timer.Stop("create directly visible scene samples");

	if(m_Visibles.size() == 0) return;

	if(profile) timer.Start();
	// determine scene samples
	for(int i = 0; i < m_Visibles.size(); ++i)
	{		
		float t = 0.f;
		Intersection intersection;
		bool isect = false;

		// get random direction
		float pdf = 0.f;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(m_Visibles[i].normal, Rand01(), Rand01(), pdf, 1);
		Ray r(m_Visibles[i].position + 0.05f * direction, direction);
		
		// create scene sample
		isect = m_pScene->IntersectRayScene(r, &t, &intersection, CPrimitive::FRONT_FACE);
		
		if(isect) {
			SceneSample ss(intersection);
			
			// estimate probability
			float p = 0.f;
			const glm::vec3 dir = glm::normalize(ss.position - m_Visibles[i].position);
			const float dist = glm::length(ss.position - m_Visibles[i].position);
			const float cos_theta = glm::dot(ss.normal, -dir);
			p += pdf * cos_theta / (dist * dist);
					
			for(int j = 0; j < M; ++j)
			{
				if(Visible(ss, m_Visibles[j], CPrimitive::FRONT_FACE))
				{
					const glm::vec3 dir = glm::normalize(ss.position - m_Visibles[j].position);
					const float dist = glm::length(ss.position - m_Visibles[j].position);
					
					const float cos_theta = glm::dot(ss.normal, -dir);
					if(cos_theta <= 0.f) continue;

					float p_dir = 0.f;
					GetRandomSampleDirectionProbability(m_Visibles[j].normal, dir, p_dir, 1);
					if(p_dir <= 0.f) continue;				

					p += p_dir * cos_theta / (dist * dist);
				}
 			}
			ss.pdf = p / float(M+1);
						
			m_SceneSamples.push_back(ss);			
		}
		
		// create anti scene sample
		isect = m_pScene->IntersectRayScene(r, &t, &intersection, CPrimitive::BACK_FACE);
		
		if(isect) {
			SceneSample ss(intersection);
			
			// estimate probability
			float p = 0.f;
			const glm::vec3 dir = glm::normalize(ss.position - m_Visibles[i].position);
			const float dist = glm::length(ss.position - m_Visibles[i].position);
			const float cos_theta = glm::dot(ss.normal, dir);
			p += pdf * cos_theta / (dist * dist);

			for(int j = 0; j < M; ++j)
			{
				if(Visible(ss, m_Visibles[j], CPrimitive::BACK_FACE))
				{
					const glm::vec3 dir = glm::normalize(ss.position - m_Visibles[j].position);
					const float dist = glm::length(ss.position - m_Visibles[j].position);
					
					const float cos_theta = glm::dot(ss.normal, dir);
					if(cos_theta <= 0.f) continue;

					float p_dir = 0.f;
					GetRandomSampleDirectionProbability(m_Visibles[j].normal, dir, p_dir, 1);
					if(p_dir <= 0.f) continue;				

					p += p_dir * cos_theta / (dist * dist);
				}
 			}
			ss.pdf = p / float(M+1);
						
			m_AntiSceneSamples.push_back(ss);			
		}
	}

	if(profile) timer.Stop("create second bounce scene samples");
	if(profile) std::cout << "number of scene samples: " << m_SceneSamples.size() 
		<< ", number of anti scene samples: " << m_AntiSceneSamples.size() << std::endl;

}