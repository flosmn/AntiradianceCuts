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
	m_NumVisibles = 49;
	CreateVisibles(m_Visibles, m_NumVisibles);
	
	CreateSceneSamples(profile);
	CreateAntiSceneSamples(profile);

	for(int i = 0; i < m_pConfManager->GetConfVars()->NumPathsPerFrame; ++i)
	{
		CreatePath(avpls, profile);
		numPaths++;
	}
}

void CBidirInstantRadiosity::CreatePath(std::vector<AVPL>& avpls, bool profile)
{
	AVPL pred, succ;
		
	// create new primary light on light source
	m_pScene->CreateAVPL(0, &pred);
	
	float scaleSS = 1.f/float(m_VisiblesSS.size());
	float scaleASS = 1.f/float(m_VisiblesASS.size());

	ConnectToSceneSamples(pred, avpls, scaleSS);
	ConnectToAntiSceneSamples(pred, avpls, scaleASS);
		
	avpls.push_back(pred);
	
	/*
	int currentBounce = 2;
	bool terminate = false;
	int bl = m_pConfManager->GetConfVars()->LimitBounces;
	const float rrProb = (bl == -1 ? 0.8f : 1.0f);
	while(!terminate)
	{
		// decide whether to terminate path
		float rand_01 = Rand01();
		if(bl == -1 ? rand_01 > rrProb : currentBounce > bl)
		{
			terminate = true;
		}
		else
		{
			// follow the path with cos-sampled direction (importance sample diffuse surface)
			// if the ray hits geometry
			float pdf = 0.f;
			glm::vec3 direction = GetRandomSampleDirectionCosCone(pred.GetOrientation(), Rand01(), Rand01(), pdf, 1);
			Ray r(pred.GetPosition() + EPSILON * direction, direction);
			
			// create scene sample
			float t = 0.f;
			Intersection intersection;
			bool isect = m_pScene->IntersectRayScene(r, &t, &intersection, CPrimitive::FRONT_FACE);
			
			if(isect) {
				SceneSample ss(intersection);
				ss.pdf = pdf;
				CreateAVPLAtSceneSample(ss, pred, &succ);

				succ.SetIntensity(succ.GetIntensity(succ.GetOrientation()) / rrProb);
				avpls.push_back(succ);
				pred = succ;

				scaleSS *= scaleSS;
				scaleASS *= scaleASS;
				ConnectToSceneSamples(pred, avpls, scaleSS);
				ConnectToAntiSceneSamples(pred, avpls, scaleASS);
			}
			else
			{
				// if the ray hits no geometry the transpored energy
				// goes to nirvana and is lost
				terminate = true;
			}
		}

		currentBounce++;
	}	
	*/
}

void CBidirInstantRadiosity::ConnectToSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale)
{
	float scaleASS = 1.f/float(m_VisiblesASS.size());
	for(int i = 0; i < m_SceneSamples.size(); ++i)
	{
		if(Visible(m_SceneSamples[i], avpl, CPrimitive::FRONT_FACE))
		{
			AVPL newAvpl;
			if(CreateAVPLAtSceneSample(m_SceneSamples[i], avpl, &newAvpl))
			{
				ConnectToAntiSceneSamples(newAvpl, avpls, scaleASS * scale);
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
		
	glm::vec3 intensity = glm::vec3(0.f);
	if(ss.pdf > 0.f)
		intensity = rho/PI * G * 1.f/ss.pdf * pred.GetMaxIntensity();
	
	//const float cone_min = m_pConfManager->GetConfVars()->ClampCone ? PI / (PI/2.f - acos(glm::dot(norm, -direction))) : 0.f;
	//const float coneFactor = std::max(cone_min, m_pConfManager->GetConfVars()->ConeFactor);
	
	const float coneFactor = m_pConfManager->GetConfVars()->ConeFactor;
	const float area = 2 * PI * ( 1 - cos(PI/coneFactor) );
	
	glm::vec3 antiintensity = glm::vec3(0.f);
	
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
	
	glm::vec3 antiintensity = glm::vec3(0.f);
	if(ss.pdf > 0.f)
		antiintensity = pred.GetMaxIntensity() * G * 1.f/ss.pdf * 1.f/area;
	
	AVPL avpl(pos, norm, intensity, antiintensity, direction, coneFactor, pred.GetBounce() + 1, m_pConfManager);
	*newAvpl = avpl;
	return true;
}

void CBidirInstantRadiosity::CreateSceneSamples(bool profile)
{
	CTimer timer(CTimer::CPU);

	int N = m_pConfManager->GetConfVars()->NumEyeRaysSS;
	int M = m_pConfManager->GetConfVars()->NumSamplesForPESS;

	m_SceneSamples.clear();
	m_VisiblesSS.clear();
	m_SceneSamples.reserve(N);
	m_VisiblesSS.reserve(N);

	if(profile) timer.Start();

	// determine directly visible scene samples
	CreateVisibleSceneSamples(m_VisiblesSS, M);

	CreateVisibleSceneSamples(m_VisiblesSS, N-M);
	
	if(m_VisiblesSS.size() < N)
	{
		std::cout << "not enough visible scene samples" << std::endl;
		return;
	}

	if(profile) timer.Stop("create directly visible scene samples");

	if(profile) timer.Start();
	// determine scene samples
	for(int i = 0; i < m_VisiblesSS.size(); ++i)
	{		
		float t = 0.f;
		Intersection intersection;
		bool isect = false;

		// get random direction
		float pdf = 0.f;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(m_VisiblesSS[i].normal, Rand01(), Rand01(), pdf, 1);
		Ray r(m_VisiblesSS[i].position + 0.05f * direction, direction);
		
		// create scene sample
		isect = m_pScene->IntersectRayScene(r, &t, &intersection, CPrimitive::FRONT_FACE);
		
		if(isect) {
			SceneSample ss(intersection);
			
			// estimate probability
			float p = 0.f;					
			for(int j = 0; j < M; ++j)
			{
				if(Visible(ss, m_VisiblesSS[j], CPrimitive::FRONT_FACE))
				{
					const glm::vec3 dir = glm::normalize(ss.position - m_VisiblesSS[j].position);
					const float dist = glm::length(ss.position - m_VisiblesSS[j].position);
					
					const float cos_theta = glm::dot(ss.normal, -dir);
					if(cos_theta <= 0.f) continue;

					float p_dir = 0.f;
					GetRandomSampleDirectionProbability(m_VisiblesSS[j].normal, dir, p_dir, 1);
					if(p_dir <= 0.f) continue;				

					p += p_dir * cos_theta / (dist * dist);
				}
 			}
			ss.pdf = p / float(M);
						
			m_SceneSamples.push_back(ss);			
		}
	}

	if(profile) timer.Stop("create second bounce scene samples");
	if(profile) std::cout << "number of scene samples: " << m_SceneSamples.size() << std::endl;

}

void CBidirInstantRadiosity::CreateAntiSceneSamples(bool profile)
{
	CTimer timer(CTimer::CPU);

	int N = m_pConfManager->GetConfVars()->NumEyeRaysASS;
	int M = m_pConfManager->GetConfVars()->NumSamplesForPEASS;

	m_AntiSceneSamples.clear();
	m_VisiblesASS.clear();
	m_AntiSceneSamples.reserve(N);
	m_VisiblesASS.reserve(N);

	if(profile) timer.Start();

	// determine directly visible scene samples
	CreateVisibleSceneSamples(m_VisiblesASS, M);

	CreateVisibleSceneSamples(m_VisiblesASS, N-M);
	
	if(m_VisiblesASS.size() < N)
	{
		std::cout << "not enough visible anti scene samples" << std::endl;
		return;
	}

	if(profile) timer.Stop("create directly visible scene samples");

	if(profile) timer.Start();
	// determine scene samples
	for(int i = 0; i < m_VisiblesASS.size(); ++i)
	{		
		float t = 0.f;
		Intersection intersection;
		bool isect = false;

		// get random direction
		float pdf = 0.f;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(m_VisiblesASS[i].normal, Rand01(), Rand01(), pdf, 1);
		Ray r(m_VisiblesASS[i].position + 0.05f * direction, direction);
				
		// create anti scene sample
		isect = m_pScene->IntersectRayScene(r, &t, &intersection, CPrimitive::BACK_FACE);
		
		if(isect) {
			SceneSample ss(intersection);
			
			// estimate probability
			float p = 0.f;
			for(int j = 0; j < M; ++j)
			{
				if(Visible(ss, m_VisiblesASS[j], CPrimitive::BACK_FACE))
				{
					const glm::vec3 dir = glm::normalize(ss.position - m_VisiblesASS[j].position);
					const float dist = glm::length(ss.position - m_VisiblesASS[j].position);
					
					const float cos_theta = glm::dot(ss.normal, dir);
					if(cos_theta <= 0.f) continue;

					float p_dir = 0.f;
					GetRandomSampleDirectionProbability(m_VisiblesASS[j].normal, dir, p_dir, 1);
					if(p_dir <= 0.f) continue;				

					p += p_dir * cos_theta / (dist * dist);
				}
 			}
			ss.pdf = p / float(M);
						
			m_AntiSceneSamples.push_back(ss);			
		}
	}

	if(profile) timer.Stop("create second bounce scene samples");
	if(profile) std::cout << "number of anti scene samples: " << m_AntiSceneSamples.size() << std::endl;
}

void CBidirInstantRadiosity::CreateVisibleSceneSamples(std::vector<SceneSample>& sceneSamples, int numSS)
{
	int maxGeneratedSamples = 10 * numSS;
	int numGeneratedSamples = 0;
	int numValidSamples = 0; 
	
	if(m_pConfManager->GetConfVars()->UseStratification)
	{
		// stratified samples
		std::vector<Ray> rays;
		std::vector<glm::vec2> samples;
		m_pScene->GetCamera()->GetEyeRays(rays, samples, numSS);
		for(int i = 0; i < numSS; ++i)
		{
			float t = 0.f;
			Intersection intersection;
			bool isect = m_pScene->IntersectRayScene(rays[i], &t, &intersection, CPrimitive::FRONT_FACE);
			
			if(isect) {
				numValidSamples++;
				SceneSample ss(intersection);
				ss.pdf = 0.f;
				sceneSamples.push_back(ss);
			}
		}
		rays.clear();
		samples.clear();
	}

	// non stratified samples for rest
	while(numValidSamples < numSS && numGeneratedSamples < maxGeneratedSamples)
	{
		numGeneratedSamples++;
		Ray e = m_pScene->GetCamera()->GetEyeRay();
		
		float t = 0.f;
		Intersection intersection;
		bool isect = m_pScene->IntersectRayScene(e, &t, &intersection, CPrimitive::FRONT_FACE);
	
		if(isect) {
			numValidSamples++;
			SceneSample ss(intersection);
			ss.pdf = 0.f;
			sceneSamples.push_back(ss);
		}
	}
}

void CBidirInstantRadiosity::CreateVisibles(std::vector<SceneSample>& sceneSamples, int numVisibles)
{
	if(m_pConfManager->GetConfVars()->UseStratification)
	{
		// stratified samples
		std::vector<Ray> rays;
		std::vector<glm::vec2> samples;
		m_pScene->GetCamera()->GetEyeRays(rays, samples, numVisibles);
		for(int i = 0; i < numVisibles; ++i)
		{
			float t = 0.f;
			Intersection intersection;
			bool isect = m_pScene->IntersectRayScene(rays[i], &t, &intersection, CPrimitive::FRONT_FACE);
			
			if(isect) {
				SceneSample ss(intersection);
				ss.pdf = 0.f;
				sceneSamples.push_back(ss);
			}
		}
		rays.clear();
		samples.clear();
	}
	else
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
				ss.pdf = 0.f;
				sceneSamples.push_back(ss);
			}
		}
	}
}