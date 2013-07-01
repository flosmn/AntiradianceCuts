#include "Scene.h"

typedef unsigned int uint;

#include "Defines.h"
#include "Structs.h"

#include "AVPL.h"
#include "AreaLight.h"
#include "CCamera.h"
#include "KdTreeAccelerator.h"
#include "Triangle.h"
#include "CConfigManager.h"
#include "CTimer.h"
#include "CMaterialBuffer.h"
#include "CReferenceImage.h"

#include "OGLResources\COGLUniformBuffer.h"

#include "OCLResources\COCLContext.h"

#include "MeshResources\CMesh.h"
#include "MeshResources\CModel.h"
#include "MeshResources\CSubModel.h"

#include <set>
#include <iostream>
#include <algorithm>
#include <iterator>

#include <omp.h>

Scene::Scene(CCamera* camera, CConfigManager* confManager, COCLContext* clContext)
	: m_camera(camera), m_confManager(confManager)
{
	m_materialBuffer.reset(new CMaterialBuffer(clContext));

	m_CurrentBounce = 0;
	m_NumCreatedAVPLs = 0;
	m_NumLightPaths = 0;
	m_HasLightSource = true;
}

Scene::~Scene()
{
}

void Scene::ClearScene() 
{
	m_models.clear();
	
	ClearLighting();
}

 void Scene::ClearLighting()
{			
	m_CurrentBounce = 0;
	m_NumLightPaths = 0;

	m_NumCreatedAVPLs = 0;
}

void Scene::DrawScene(COGLUniformBuffer* ubTransform, COGLUniformBuffer* ubMaterial)
{
	std::vector<std::unique_ptr<CModel>>::iterator it;

	for (it = m_models.begin(); it < m_models.end(); it++) {
		(*it)->Draw(m_camera->GetViewMatrix(), m_camera->GetProjectionMatrix(), ubTransform, ubMaterial);
	}
}

void Scene::DrawScene(COGLUniformBuffer* ubTransform)
{
	DrawScene(m_camera->GetViewMatrix(), m_camera->GetProjectionMatrix(), ubTransform);
}

void Scene::DrawScene(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* ubTransform)
{
	std::vector<std::unique_ptr<CModel>>::iterator it;

	for (it = m_models.begin(); it < m_models.end(); it++) {
		(*it)->Draw(mView, mProj, ubTransform);
	}
}

void Scene::DrawAreaLight(COGLUniformBuffer* ubTransform, COGLUniformBuffer* ubAreaLight)
{
	m_areaLight->Draw(m_camera, ubTransform, ubAreaLight);
}

void Scene::DrawAreaLight(COGLUniformBuffer* ubTransform, COGLUniformBuffer* ubAreaLight, glm::vec3 color)
{
	m_areaLight->Draw(m_camera, ubTransform, ubAreaLight, color);
}

bool Scene::CreateAVPL(AVPL* pred, AVPL* newAVPL)
{
	bool mis = m_confManager->GetConfVars()->UseMIS == 1 ? true : false;

	float pdf = 0.f;
	glm::vec3 direction = SamplePhong(pred->GetDirection(), pred->GetOrientation(), 
		m_materialBuffer->GetMaterial(pred->GetMaterialIndex()), pdf, mis);
	
	if(pdf < 0.f)
	{
		std::cout << "pdf < 0.f" << std::endl;
		newAVPL = 0;
		return false;
	}

	if(pdf == 0.f)
	{
		std::cout << "pdf == 0.f" << std::endl;
		newAVPL = 0;
		return false;
	}

	AVPL avpl;
	if(ContinueAVPLPath(pred, &avpl, direction, pdf))
	{
		*newAVPL = avpl;
		return true;
	}

	newAVPL = 0;
	return false;
}

bool Scene::CreatePrimaryAVPL(AVPL* newAVPL)
{
	if(!m_areaLight)
	{
		newAVPL = 0;
		return false;
	}
	
	// create VPL on light source
	float pdf;
	glm::vec3 pos = m_areaLight->SamplePos(pdf);

	glm::vec3 ori = m_areaLight->GetFrontDirection();
	glm::vec3 I = m_areaLight->GetRadiance();

	if(!(pdf > 0.f))
		std::cout << "Warning: pdf is 0" << std::endl;

	AVPL avpl(pos, ori, I / pdf, glm::vec3(0), glm::vec3(0), m_confManager->GetConfVars()->ConeFactor, 
		0, m_areaLight->GetMaterialIndex(), m_materialBuffer.get(), m_confManager);
	*newAVPL = avpl;

	return true;
}

bool Scene::ContinueAVPLPath(AVPL* pred, AVPL* newAVPL, glm::vec3 direction, float pdf)
{
	glm::vec3 pred_pos = pred->GetPosition();

	const glm::vec3 pred_norm = glm::normalize(pred->GetOrientation());
	direction = glm::normalize(direction);

	const float epsilon = 0.005f;
	Ray ray(pred_pos + epsilon * direction, direction);
		
	Intersection intersection_kd, intersection_simple;
	float t_kd = 0.f, t_simple = 0.f;
	bool inter_kd = IntersectRayScene(ray, &t_kd, &intersection_kd, Triangle::FRONT_FACE);
		
	/*
	bool inter_simple = IntersectRaySceneSimple(ray, &t_simple, &intersection_simple, isect_bfc);
	if(inter_kd != inter_simple)
		std::cout << "KDTree intersection and naive intersection differ!" << std::endl;
	else if(intersection_kd.GetPosition() != intersection_simple.GetPosition())
		std::cout << "KDTree intersection point and naive intersection point differ!" << std::endl;
	*/

	Intersection intersection = intersection_kd;
	float t = t_kd;
	if(inter_kd)
	{
		// gather information for the new VPL
		glm::vec3 norm = intersection.getTriangle()->getNormal();
		glm::vec3 pos = intersection.getPosition() - 0.1f * norm;
		uint index = intersection.getTriangle()->getMaterialIndex();
		if(index > 100)
			std::cout << "material index uob" << std::endl;
		
		MATERIAL* mat = m_materialBuffer->GetMaterial(index);
		
		const float cos_theta = glm::dot(pred_norm, direction);

		glm::vec3 contrib = pred->GetRadiance(direction) * cos_theta / pdf;	
		
		float coneFactor = m_confManager->GetConfVars()->ConeFactor;
		const int clamp_cone_mode = m_confManager->GetConfVars()->ClampConeMode;
		if(m_confManager->GetConfVars()->ClampConeMode == 1)
		{
			const float cone_min = M_PI / (M_PI/2.f - acos(glm::dot(norm, -direction)));
			coneFactor = std::max(cone_min, m_confManager->GetConfVars()->ConeFactor);
		}
		
		const float area = 2 * M_PI * ( 1 - cos(M_PI/coneFactor) );

		glm::vec3 antiradiance = 1.f/float(m_confManager->GetConfVars()->NumAdditionalAVPLs + 1.f) * contrib / area;

		AVPL avpl(pos, norm, contrib, antiradiance, direction, coneFactor, pred->GetBounce() + 1,
			intersection.getTriangle()->getMaterialIndex(), m_materialBuffer.get(), m_confManager);
		*newAVPL = avpl;
		return true;
	}

	newAVPL = 0;
	return false;
}

void Scene::CreatePaths(std::vector<AVPL>& avpls_res, std::vector<AVPL>& allAVPLs, std::vector<AVPL>& isAVPLs, bool profile, uint numPaths)
{
	avpls_res.reserve(numPaths * 4);
	allAVPLs.reserve(numPaths * 4);	
	isAVPLs.reserve(numPaths * 4);

	for(uint i = 0; i < numPaths; ++i)
	{
		CreatePath(avpls_res);
	}
	m_NumCreatedAVPLs += uint(avpls_res.size());
}

void Scene::CreatePath(std::vector<AVPL>& path) 
{
	m_NumLightPaths++;

	int currentBounce = 0;
	
	AVPL pred, succ;
		
	// create new primary light on light source
	CreatePrimaryAVPL(&pred);
	
	path.push_back(pred);
	
	currentBounce++;

	// follow light path until it is terminated
	// by RR with termination probability 1-rrProb
	bool terminate = false;
	int bl = m_confManager->GetConfVars()->LimitBounces;
	const float rrProb = (bl == -1 ? 0.8f : 1.0f);
	while(!terminate)
	{
		// decide whether to terminate path
		float rand_01 = glm::linearRand(0.f, 1.f);

		if(bl == -1 ? rand_01 > rrProb : currentBounce > bl)
		{
			// create path-finishing Anti-VPLs
			CreateAVPLs(&pred, path, m_confManager->GetConfVars()->NumAdditionalAVPLs);

			AVPL avpl;
			if(CreateAVPL(&pred, &avpl))
			{
				avpl.ScaleIncidentRadiance(0.f);
				path.push_back(avpl);
			}
			
			terminate = true;
		}
		else
		{
			// create additional avpls
			CreateAVPLs(&pred, path, m_confManager->GetConfVars()->NumAdditionalAVPLs);

			// follow the path with cos-sampled direction (importance sample diffuse surface)
			// if the ray hits geometry
			if(CreateAVPL(&pred, &succ))
			{
				succ.ScaleIncidentRadiance(1.f / rrProb);
				path.push_back(succ);

				pred = succ;
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
}

void Scene::CreateAVPLs(AVPL* pred, std::vector<AVPL>& path, int nAVPLs)
{
	glm::vec3 pred_norm = pred->GetOrientation();

	std::vector<glm::vec3> directions;
	std::vector<float> pdfs;
	GetStratifiedDirections(directions, pdfs, nAVPLs, pred_norm, 1);

	for(int i = 0; i < nAVPLs; ++i)
	{
		float pdf = pdfs[i]; //0.f; //
		glm::vec3 direction = directions[i]; // GetRandomSampleDirectionCosCone(pred_norm, Rand01(), Rand01(), pdf, 1);
		
		if(pdf <= 0.f)
		{
			std::cout << "pdf <= 0.f" << std::endl;
			continue;
		}

		AVPL avpl;
		if(ContinueAVPLPath(pred, &avpl, direction, pdf))		
		{
			avpl.ScaleIncidentRadiance(0.f);
			avpl.SetColor(glm::vec3(1.f, 0.f, 0.f));
			path.push_back(avpl);
		}
	}
}

void Scene::CreatePrimaryVpls(std::vector<AVPL>& avpls, int numVpls)
{
	for(int i = 0; i < numVpls; ++i)
	{
		AVPL avpl;
		if(CreatePrimaryAVPL(&avpl))
			avpls.push_back(avpl);
	}
}

bool Scene::IntersectRaySceneSimple(const Ray& ray, float* t, Intersection *pIntersection,  Triangle::IsectMode isectMode)
{	
	float t_best = std::numeric_limits<float>::max();
	Intersection isect_best;
	bool hasIntersection = false;

	for(uint i = 0; i < m_primitives.size(); ++i)
	{
		float t_temp = 0.f;
		Intersection isect_temp;
		if(m_primitives[i].intersect(ray, &t_temp, &isect_temp, isectMode))
		{
			if(t_temp < t_best && t_temp > 0) {
				t_best = t_temp;
				isect_best = isect_temp;
				hasIntersection = true;
			}
		}
	}

	*pIntersection = isect_best;
	*t = t_best;
		
	// no intersections on light sources
	if(hasIntersection) {
		if(glm::length(GetMaterialBuffer()->GetMaterial(pIntersection->getTriangle()->getMaterialIndex())->emissive) > 0.f) {
			hasIntersection =  false;
		}
	}

	return hasIntersection;
}

bool Scene::IntersectRayScene(const Ray& ray, float* t, Intersection *pIntersection,  Triangle::IsectMode isectMode)
{	
	bool intersect = m_kdTreeAccelerator->intersect(ray, t, pIntersection, isectMode);

	// no intersections on light sources
	if(intersect) {
		if(glm::length(GetMaterialBuffer()->GetMaterial(pIntersection->getTriangle()->getMaterialIndex())->emissive) > 0.f) {
			intersect =  false;
		}
	}

	return intersect;
}

void Scene::LoadSimpleScene()
{
	ClearScene();

	m_models.emplace_back(std::unique_ptr<CModel>(
		new CModel("twoplanes", "obj", m_materialBuffer.get())));
	m_models.back()->SetWorldTransform(glm::scale(glm::vec3(2.f, 2.f, 2.f)));

	m_camera->Init(0, glm::vec3(-9.2f, 5.7f, 6.75f), 
		glm::vec3(0.f, -2.f, 0.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);
	
	m_areaLight.reset(new AreaLight(0.5f, 0.5f, 
		glm::vec3(0.0f, 4.f, 0.0f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		m_materialBuffer.get()));

	m_areaLight->SetRadiance(glm::vec3(120.f, 120.f, 120.f));
	InitKdTree();
}

void Scene::LoadCornellBox()
{
	ClearScene();

	m_models.emplace_back(std::unique_ptr<CModel>(
		new CModel("cb-diffuse", "obj", m_materialBuffer.get())));
	m_models.back()->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));

	m_referenceImage.reset(new CReferenceImage(m_camera->GetWidth(), m_camera->GetHeight()));
	m_referenceImage->LoadFromFile("References/cb-diffuse-closeup-ref.hdr", true);

	m_camera->Init(0, glm::vec3(278.f, 273.f, -650.f), 
		glm::vec3(278.f, 273.f, -649.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_camera->Init(1, glm::vec3(128.5f, 42.9f, 8.9f), 
		glm::vec3(128.2f, 42.6f, 9.8f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_camera->Init(2, glm::vec3(278.f, 273.f, -500.f), 
		glm::vec3(278.f, 273.f, -499.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_camera->Init(3, glm::vec3(278.f, 273.f, -700.f), 
		glm::vec3(278.f, 273.f, -699.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_camera->Init(4, glm::vec3(300.1f, 293.f, 78.4f), 
		glm::vec3(300.1f, 291.0f, 78.5f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_camera->UseCameraConfig(0);
	
	glm::vec3 areaLightFrontDir = glm::vec3(0.0f, -1.0f, 0.0f);
	glm::vec3 areaLightPosition = glm::vec3(278.f, 548.78999f, 279.5f);
	
	m_confManager->GetConfVars()->AreaLightFrontDirection[0] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_confManager->GetConfVars()->AreaLightFrontDirection[1] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_confManager->GetConfVars()->AreaLightFrontDirection[2] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_confManager->GetConfVars()->AreaLightPosX = m_confManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_confManager->GetConfVars()->AreaLightPosY = m_confManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_confManager->GetConfVars()->AreaLightPosZ = m_confManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	float AreaLightRadianceScale = 150;
	m_confManager->GetConfVars()->AreaLightRadianceScale = m_confManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;

	m_areaLight.reset(new AreaLight(130.0f, 105.0f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_materialBuffer.get()));

	m_areaLight->SetRadiance(glm::vec3(100.f, 100.f, 100.f));

	m_confManager->GetConfVars()->UseIBL = m_confManager->GetConfVarsGUI()->UseIBL = 0;
		
	InitKdTree();
}

void Scene::LoadCornellEmpty()
{
	ClearScene();

	m_models.emplace_back(std::unique_ptr<CModel>(
		new CModel("cb-empty", "obj", m_materialBuffer.get())));
	m_models.back()->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));

	m_referenceImage.reset(new CReferenceImage(m_camera->GetWidth(), m_camera->GetHeight()));
	m_referenceImage->LoadFromFile("References/cb-diffuse-clamped-indirect.hdr", true);

	m_camera->Init(0, glm::vec3(386.7f, 165.9f, -4.6f), 
		glm::vec3(387.0f, 165.4f, -3.7f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_camera->UseCameraConfig(0);
		
	glm::vec3 areaLightFrontDir = glm::vec3(1.0f, 0.0f, 1.0f);
	glm::vec3 areaLightPosition = glm::vec3(450.f, 5.00f, 279.5f);
	
	m_confManager->GetConfVars()->AreaLightFrontDirection[0] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_confManager->GetConfVars()->AreaLightFrontDirection[1] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_confManager->GetConfVars()->AreaLightFrontDirection[2] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_confManager->GetConfVars()->AreaLightPosX = m_confManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_confManager->GetConfVars()->AreaLightPosY = m_confManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_confManager->GetConfVars()->AreaLightPosZ = m_confManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	float AreaLightRadianceScale = 10.f;
	m_confManager->GetConfVars()->AreaLightRadianceScale = m_confManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;

	m_areaLight.reset(new AreaLight(50.0f, 5.0f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_materialBuffer.get()));

	m_areaLight->SetRadiance(AreaLightRadianceScale * glm::vec3(100.f, 100.f, 100.f));

	m_confManager->GetConfVars()->UseIBL = m_confManager->GetConfVarsGUI()->UseIBL = 0;
		
	InitKdTree();
}

void Scene::LoadBuddha()
{
	ClearScene();

	m_models.emplace_back(std::unique_ptr<CModel>(
		new CModel("cb-buddha-specular", "obj", m_materialBuffer.get())));
	m_models.back()->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));

	m_referenceImage.reset(new CReferenceImage(m_camera->GetWidth(), m_camera->GetHeight()));
	m_referenceImage->LoadFromFile("References/cb-buddha-full-noclamp.hdr", true);

	m_camera->Init(0, glm::vec3(278.f, 273.f, -800.f), 
		glm::vec3(278.f, 273.f, -799.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_camera->Init(1, glm::vec3(660.97f, 363.99f, -126.34f), 
		glm::vec3(660.3f, 363.7f, -125.7f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_camera->Init(2, glm::vec3(278.f, 273.f, -500.f), 
		glm::vec3(278.f, 273.f, -499.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_camera->UseCameraConfig(2);
	
	glm::vec3 areaLightFrontDir = glm::vec3(0.0f, -1.0f, 0.0f);
	glm::vec3 areaLightPosition = glm::vec3(278.f, 548.78999f, 279.5f);
	
	m_confManager->GetConfVars()->AreaLightFrontDirection[0] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_confManager->GetConfVars()->AreaLightFrontDirection[1] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_confManager->GetConfVars()->AreaLightFrontDirection[2] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_confManager->GetConfVars()->AreaLightPosX = m_confManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_confManager->GetConfVars()->AreaLightPosY = m_confManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_confManager->GetConfVars()->AreaLightPosZ = m_confManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	float AreaLightRadianceScale = 100;
	m_confManager->GetConfVars()->AreaLightRadianceScale = m_confManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;

	m_areaLight.reset(new AreaLight(130.0f, 105.0f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_materialBuffer.get()));

	m_areaLight->SetRadiance(glm::vec3(100.f, 100.f, 100.f));

	m_confManager->GetConfVars()->UseIBL = m_confManager->GetConfVarsGUI()->UseIBL = 0;
		
	InitKdTree();
}

void Scene::LoadConferenceRoom()
{
	ClearScene();

	m_models.emplace_back(std::unique_ptr<CModel>(
		new CModel("conference-3", "obj", m_materialBuffer.get())));
	m_models.back()->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));
		
	m_camera->Init(0, glm::vec3(-130.8f, 304.3f, 21.6f), 
		glm::vec3(-129.9f, 304.0f, 21.2f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);
	
	m_camera->UseCameraConfig(0);
	
	glm::vec3 areaLightFrontDir = glm::vec3(0.0f, -1.0f, 0.0f);
	glm::vec3 areaLightPosition = glm::vec3(450.f, 615.f, -128.5f);
	
	m_confManager->GetConfVars()->AreaLightFrontDirection[0] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_confManager->GetConfVars()->AreaLightFrontDirection[1] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_confManager->GetConfVars()->AreaLightFrontDirection[2] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_confManager->GetConfVars()->AreaLightPosX = m_confManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_confManager->GetConfVars()->AreaLightPosY = m_confManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_confManager->GetConfVars()->AreaLightPosZ = m_confManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	float AreaLightRadianceScale = 80;
	m_confManager->GetConfVars()->AreaLightRadianceScale = m_confManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;

	m_areaLight.reset(new AreaLight(130.0f, 105.0f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_materialBuffer.get()));

	m_areaLight->SetRadiance(AreaLightRadianceScale * glm::vec3(1.f, 1.f, 1.f));

	m_confManager->GetConfVars()->UseIBL = m_confManager->GetConfVarsGUI()->UseIBL = 0;
		
	InitKdTree();
}

void Scene::LoadSibernik()
{
	ClearScene();

	float scale = 100.f;

	m_models.emplace_back(std::unique_ptr<CModel>(
		new CModel("sibenik", "obj", m_materialBuffer.get())));
	m_models.back()->SetWorldTransform(glm::scale(scale * glm::vec3(1.f, 1.f, 1.f)));

	m_camera->Init(0, scale * glm::vec3(-16.f, -5.f, 0.f), 
		scale * glm::vec3(-15.f, -5.5f, 0.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.f);
			
	glm::vec3 areaLightFrontDir = glm::vec3(0.f, -1.f, 0.f);
	glm::vec3 areaLightPosition = scale * glm::vec3(0.f, 2.f, 0.f);
	
	m_camera->Init(1, areaLightPosition, 
		areaLightPosition + areaLightFrontDir,
		glm::vec3(1.f, 0.f, 0.f),
		2.f);

	m_camera->Init(2, scale * 0.01f * glm::vec3(-269.1f, -1266.1f, 239.3f), 
		scale * 0.01f * glm::vec3(-206.1f, -1308.4f, 314.3f),
		glm::vec3(0.f, 1.f, 0.f),
		2.f);

	m_camera->Init(3, scale * 0.01f * glm::vec3(-1855.f, -923.f, 0.f), 
		scale * 0.01f * glm::vec3(-1854.f, -923.f, 0.0f),
		glm::vec3(0.f, 1.f, 0.f),
		2.f);

	m_camera->UseCameraConfig(3);

	m_confManager->GetConfVars()->AreaLightFrontDirection[0] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_confManager->GetConfVars()->AreaLightFrontDirection[1] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_confManager->GetConfVars()->AreaLightFrontDirection[2] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_confManager->GetConfVars()->AreaLightPosX = m_confManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_confManager->GetConfVars()->AreaLightPosY = m_confManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_confManager->GetConfVars()->AreaLightPosZ = m_confManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	m_confManager->GetConfVars()->ClampConeMode = m_confManager->GetConfVarsGUI()->ClampConeMode = 0;

	float AreaLightRadianceScale = 3000.f;
	m_confManager->GetConfVars()->AreaLightRadianceScale = m_confManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;
	
	m_areaLight.reset(new AreaLight(scale * 0.25f, scale * 0.25f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_materialBuffer.get()));
	
	m_areaLight->SetRadiance(glm::vec3(AreaLightRadianceScale));

	InitKdTree();
}

void Scene::LoadCornellBoxDragon()
{
	ClearScene();

	m_models.emplace_back(std::unique_ptr<CModel>(
		new CModel("cornell-box-dragon43k", "obj", m_materialBuffer.get())));
	m_models.back()->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));

	m_camera->Init(0, glm::vec3(0.0f, 1.0f, 3.5f), 
		glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		1.0f);

	glm::vec3 areaLightFrontDir = glm::vec3(0.f, -1.f, 0.f);
	glm::vec3 areaLightPosition = glm::vec3(-0.005f, 1.97f, 0.19f);
	
	m_confManager->GetConfVars()->AreaLightFrontDirection[0] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_confManager->GetConfVars()->AreaLightFrontDirection[1] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_confManager->GetConfVars()->AreaLightFrontDirection[2] = m_confManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_confManager->GetConfVars()->AreaLightPosX = m_confManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_confManager->GetConfVars()->AreaLightPosY = m_confManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_confManager->GetConfVars()->AreaLightPosZ = m_confManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	float AreaLightRadianceScale = 10;
	m_confManager->GetConfVars()->AreaLightRadianceScale = m_confManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;

	m_areaLight.reset(new AreaLight(0.47f, 0.38f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_materialBuffer.get()));
	
	m_areaLight->SetIntensity(glm::vec3(AreaLightRadianceScale));

	InitKdTree();
}


void Scene::InitKdTree()
{
	std::vector<std::unique_ptr<CModel>>::iterator it_models;
	for (it_models = m_models.begin(); it_models < m_models.end(); it_models++ )
	{
		CModel* model = (*it_models).get();

		std::vector<CSubModel*> subModels = model->GetSubModels();
		std::vector<CSubModel*>::iterator it_subModels;
		for (it_subModels = subModels.begin(); it_subModels < subModels.end(); it_subModels++ )
		{
			CSubModel* subModel = *it_subModels;
			std::vector<Triangle> triangles = subModel->GetTrianglesWS();
			for(uint i = 0; i < triangles.size(); ++i)
			{
				m_primitives.push_back(triangles[i]);
			}
		}
	}

	// add area light source
	std::vector<Triangle> triangles;
	m_areaLight->GetTrianglesWS(triangles);
	for(uint i = 0; i < triangles.size(); ++i)
	{
		m_primitives.push_back(triangles[i]);
	}
	
	m_kdTreeAccelerator.reset(new KdTreeAccelerator(m_primitives, 80, 1, 5, 0));
	CTimer timer(CTimer::CPU);
	timer.Start();
	m_kdTreeAccelerator->buildTree();
	timer.Stop("BuildAccelerationTree");
}

void Scene::ReleaseKdTree()
{
	m_kdTreeAccelerator.reset(nullptr);	
	m_primitives.clear();
}

void Scene::UpdateAreaLights()
{
	if(!m_areaLight)
		return; 
	
	glm::vec3 pos = glm::vec3(
		m_confManager->GetConfVars()->AreaLightPosX, 
		m_confManager->GetConfVars()->AreaLightPosY, 
		m_confManager->GetConfVars()->AreaLightPosZ); 
	m_areaLight->SetCenterPosition(pos);

	glm::vec3 front = glm::vec3(
		m_confManager->GetConfVars()->AreaLightFrontDirection[0],
		m_confManager->GetConfVars()->AreaLightFrontDirection[1],
		m_confManager->GetConfVars()->AreaLightFrontDirection[2]);
	m_areaLight->SetFrontDirection(front);

	m_areaLight->SetRadiance(glm::vec3(m_confManager->GetConfVars()->AreaLightRadianceScale));
}

bool Scene::Visible(const SceneSample& ss1, const SceneSample& ss2)
{
	glm::vec3 direction12 = glm::normalize(ss2.position - ss1.position);
		
	if(glm::dot(ss1.normal, direction12) <= 0.f || glm::dot(ss2.normal, -direction12) <= 0.f)
		return false;

	glm::vec3 ray_origin = ss1.position + EPSILON * direction12;
	Ray r(ray_origin, direction12);
	float dist = glm::length(ss2.position - ray_origin);

	float t = 0.f;
	Intersection intersection;
	bool isect = IntersectRaySceneSimple(r, &t, &intersection,  Triangle::FRONT_FACE);
			
	const float big = std::max(dist, t);
	const float small = std::min(dist, t);

	const float temp = small/big;

	if(isect && temp > 0.99f ) {
		return true;
	}

	return false;
}

void Scene::SampleLightSource(SceneSample& ss)
{
	float pdf = 0.f;
	ss.position = m_areaLight->SamplePos(pdf);
	ss.normal = m_areaLight->GetFrontDirection();
	ss.pdf = pdf;
	ss.materialIndex = m_areaLight->GetMaterialIndex();

	if(!(pdf > 0.f))
		std::cout << "Warning: pdf is 0" << std::endl;
}
MATERIAL* Scene::GetMaterial(const SceneSample& ss)
{
	return m_materialBuffer->GetMaterial(ss.materialIndex);
}
