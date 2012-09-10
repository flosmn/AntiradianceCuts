#include "Scene.h"

typedef unsigned int uint;

#include "Defines.h"
#include "Structs.h"

#include "AVPL.h"
#include "AreaLight.h"
#include "CCamera.h"
#include "CKdTreeAccelerator.h"
#include "CPrimitive.h"
#include "CConfigManager.h"
#include "CTimer.h"
#include "CAVPLImportanceSampling.h"
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

const float EPSILON = 0.005f;

Scene::Scene(CCamera* _camera, CConfigManager* pConfManager, COCLContext* pOCLContext)
{
	m_Camera = _camera;
	m_pConfManager = pConfManager;
	m_CurrentBounce = 0;
	m_pKdTreeAccelerator = 0;
	m_pAVPLImportanceSampling = 0;
	m_AreaLight = 0;

	m_NumCreatedAVPLs = 0;
	m_NumAVPLsAfterIS = 0;
	m_HasLightSource = true;
	m_pMaterialBuffer = new CMaterialBuffer(pOCLContext);
	m_pReferenceImage = 0;
}

Scene::~Scene()
{
	SAFE_DELETE(m_AreaLight);
	SAFE_DELETE(m_pMaterialBuffer);
	SAFE_DELETE(m_pReferenceImage);
}

bool Scene::Init()
{
	m_AreaLight = 0;
	m_NumLightPaths = 0;

	return true;
}

void Scene::Release()
{
	if(m_AreaLight)
		m_AreaLight->Release();
	if(m_pReferenceImage)
		m_pReferenceImage->Release();

	ClearScene();

	ReleaseKdTree();
}

void Scene::ClearScene() 
{
	std::vector<CModel*>::iterator it;

	for ( it=m_Models.begin() ; it < m_Models.end(); it++ )
	{
		CModel* model = *it;
		model->Release();
		SAFE_DELETE(model);
	}

	m_Models.clear();

	ClearLighting();
}

 void Scene::ClearLighting()
{			
	m_CurrentBounce = 0;
	m_NumLightPaths = 0;

	m_NumAVPLsAfterIS = 0;
	m_NumCreatedAVPLs = 0;
}

void Scene::DrawScene(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBMaterial)
{
	std::vector<CModel*>::iterator it;

	for ( it=m_Models.begin() ; it < m_Models.end(); it++ )
	{
		(*it)->Draw(m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix(), pUBTransform, pUBMaterial);
	}
}

void Scene::DrawScene(COGLUniformBuffer* pUBTransform)
{
	std::vector<CModel*>::iterator it;

	for ( it=m_Models.begin() ; it < m_Models.end(); it++ )
	{
		(*it)->Draw(m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix(), pUBTransform);
	}
}

void Scene::DrawScene(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* pUBTransform)
{
	std::vector<CModel*>::iterator it;

	for ( it=m_Models.begin() ; it < m_Models.end(); it++ )
	{
		(*it)->Draw(mView, mProj, pUBTransform);
	}
}

void Scene::DrawAreaLight(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight)
{
	if(m_AreaLight)
		m_AreaLight->Draw(m_Camera, pUBTransform, pUBAreaLight);
}

void Scene::DrawAreaLight(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight, glm::vec3 color)
{
	if(m_AreaLight)
		m_AreaLight->Draw(m_Camera, pUBTransform, pUBAreaLight, color);
}

bool Scene::CreateAVPL(AVPL* pred, AVPL* newAVPL)
{
	if(!m_AreaLight)
	{
		newAVPL = 0;
		return false;
	}
	
	bool mis = m_pConfManager->GetConfVars()->UseMIS == 1 ? true : false;

	float pdf = 0.f;
	glm::vec3 direction = SamplePhong(pred->GetDirection(), pred->GetOrientation(), 
		m_pMaterialBuffer->GetMaterial(pred->GetMaterialIndex()), pdf, mis);
	
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
	if(!m_AreaLight)
	{
		newAVPL = 0;
		return false;
	}
	
	// create VPL on light source
	float pdf;
	glm::vec3 pos = m_AreaLight->SamplePos(pdf);

	glm::vec3 ori = m_AreaLight->GetFrontDirection();
	glm::vec3 I = m_AreaLight->GetRadiance();

	if(!(pdf > 0.f))
		std::cout << "Warning: pdf is 0" << std::endl;

	AVPL avpl(pos, ori, I / pdf, glm::vec3(0), glm::vec3(0), m_pConfManager->GetConfVars()->ConeFactor, 
		0, m_AreaLight->GetMaterialIndex(), m_pMaterialBuffer, m_pConfManager);
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
	bool inter_kd = IntersectRayScene(ray, &t_kd, &intersection_kd, CPrimitive::FRONT_FACE);
		
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
		glm::vec3 pos = intersection.GetPosition();
		glm::vec3 norm = intersection.GetPrimitive()->GetNormal();
		uint index = intersection.GetPrimitive()->GetMaterialIndex();
		if(index > 100)
			std::cout << "material index uob" << std::endl;
		
		MATERIAL* mat = m_pMaterialBuffer->GetMaterial(index);
		
		const float cos_theta = glm::dot(pred_norm, direction);

		glm::vec3 contrib = pred->GetRadiance(direction) * cos_theta / pdf;	
		
		const float coneFactor = m_pConfManager->GetConfVars()->ConeFactor;
		//const float cone_min = m_pConfManager->GetConfVars()->ClampCone ? PI / (PI/2.f - acos(glm::dot(norm, -direction))) : 0.f;
		//const float coneFactor = std::max(cone_min, m_pConfManager->GetConfVars()->ConeFactor);
		
		const float area = 2 * PI * ( 1 - cos(PI/coneFactor) );

		glm::vec3 antiradiance = 1.f / float(m_pConfManager->GetConfVars()->NumAdditionalAVPLs + 1) * contrib / area;

		AVPL avpl(pos, norm, contrib, antiradiance, direction, coneFactor, pred->GetBounce() + 1,
			intersection.GetMaterialIndex(), m_pMaterialBuffer, m_pConfManager);
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

	if(!m_pConfManager->GetConfVars()->UseAVPLImportanceSampling)
	{
		for(uint i = 0; i < numPaths; ++i)
		{
			CreatePath(avpls_res);
		}
		m_NumCreatedAVPLs += uint(avpls_res.size());

		if(m_pConfManager->GetConfVars()->CollectAVPLs)
		{
			std::copy(avpls_res.begin(), avpls_res.end(), std::back_inserter(allAVPLs));
		}
	}
	else
	{
		std::vector<AVPL> avpls_temp;
		avpls_temp.reserve(numPaths * 10);

		for(uint i = 0; i < numPaths; ++i)
		{
			CreatePath(avpls_temp);
		}
		m_NumCreatedAVPLs += uint(avpls_temp.size());

		CTimer timer(CTimer::CPU);
		timer.Start();
		for(int i = 0; i < avpls_temp.size(); ++i)
		{
			if(avpls_temp[i].GetBounce() > 0)
			{
				float scale = 1.f;
				
				if(ImportanceSampling(avpls_temp[i], &scale))
				{
					avpls_temp[i].ScaleAntiradiance(scale);
					avpls_temp[i].ScaleIncidentRadiance(scale);
					avpls_res.push_back(avpls_temp[i]);
				}
			}
			else
			{
				avpls_res.push_back(avpls_temp[i]);
			}
		}

		if(m_pConfManager->GetConfVars()->CollectAVPLs)
		{
			std::copy(avpls_temp.begin(), avpls_temp.end(), std::back_inserter(allAVPLs));
		}
		if(m_pConfManager->GetConfVars()->CollectISAVPLs)
		{
			std::copy(avpls_res.begin(), avpls_res.end(), std::back_inserter(isAVPLs));
		}
								
		m_NumAVPLsAfterIS += uint(avpls_res.size());
		
		avpls_temp.clear();
	}
}

bool Scene::ImportanceSampling(AVPL& avpl, float* scale)
{
	switch(m_pConfManager->GetConfVars()->ISMode)
	{
	case 0:
		return m_pAVPLImportanceSampling->EvaluateAVPLImportance0(avpl, scale);
	case 1:
		return m_pAVPLImportanceSampling->EvaluateAVPLImportance1(avpl, scale);
	case 2:
		return m_pAVPLImportanceSampling->EvaluateAVPLImportance2(avpl, scale);
	}
	return false;
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
	int bl = m_pConfManager->GetConfVars()->LimitBounces;
	const float rrProb = (bl == -1 ? 0.8f : 1.0f);
	while(!terminate)
	{
		// decide whether to terminate path
		float rand_01 = glm::linearRand(0.f, 1.f);

		if(bl == -1 ? rand_01 > rrProb : currentBounce > bl)
		{
			// create path-finishing Anti-VPLs
			CreateAVPLs(&pred, path, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);

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
			CreateAVPLs(&pred, path, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);

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

	for(int i = 0; i < nAVPLs; ++i)
	{
		
		float pdf;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(pred_norm, Rand01(), Rand01(), pdf, 1);
		
		AVPL avpl;
		if(ContinueAVPLPath(pred, &avpl, direction, pdf))		
		{
			avpl.ScaleIncidentRadiance(0.f);
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

bool Scene::IntersectRaySceneSimple(const Ray& ray, float* t, Intersection *pIntersection, CPrimitive::IsectMode isectMode)
{	
	float t_best = std::numeric_limits<float>::max();
	Intersection isect_best;
	bool hasIntersection = false;

	for(uint i = 0; i < m_Primitives.size(); ++i)
	{
		float t_temp = 0.f;
		Intersection isect_temp;
		if(m_Primitives[i]->Intersect(ray, &t_temp, &isect_temp, isectMode))
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
		if(glm::length(GetMaterialBuffer()->GetMaterial(pIntersection->GetMaterialIndex())->emissive) > 0.f) {
			hasIntersection =  false;
		}
	}

	return hasIntersection;
}

bool Scene::IntersectRayScene(const Ray& ray, float* t, Intersection *pIntersection, CPrimitive::IsectMode isectMode)
{	
	bool intersect = m_pKdTreeAccelerator->Intersect(ray, t, pIntersection, isectMode);

	// no intersections on light sources
	if(intersect) {
		if(glm::length(GetMaterialBuffer()->GetMaterial(pIntersection->GetMaterialIndex())->emissive) > 0.f) {
			intersect =  false;
		}
	}

	return intersect;
}

void Scene::LoadSimpleScene()
{
	ClearScene();

	CModel* model = new CModel();
	model->Init("twoplanes", "obj", m_pMaterialBuffer);
	model->SetWorldTransform(glm::scale(glm::vec3(2.f, 2.f, 2.f)));
		
	m_Models.push_back(model);

	m_Camera->Init(0, glm::vec3(-9.2f, 5.7f, 6.75f), 
		glm::vec3(0.f, -2.f, 0.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);
	
	m_AreaLight = new AreaLight(0.5f, 0.5f, 
		glm::vec3(0.0f, 4.f, 0.0f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		m_pMaterialBuffer);

	m_AreaLight->SetRadiance(glm::vec3(120.f, 120.f, 120.f));
	
	m_AreaLight->Init();

	InitKdTree();
}

void Scene::LoadCornellBox()
{
	ClearScene();

	CModel* model = new CModel();
	model->Init("cb-diffuse-hightessellation", "obj", m_pMaterialBuffer);
	model->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));

	m_pReferenceImage = new CReferenceImage(m_Camera->GetWidth(), m_Camera->GetHeight());
	m_pReferenceImage->LoadFromFile("References/cb-diffuse-clamped.hdr", true);

	m_Models.push_back(model);

	m_Camera->Init(0, glm::vec3(278.f, 273.f, -800.f), 
		glm::vec3(278.f, 273.f, -799.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_Camera->Init(1, glm::vec3(128.5f, 42.9f, 8.9f), 
		glm::vec3(128.2f, 42.6f, 9.8f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_Camera->Init(2, glm::vec3(278.f, 273.f, -500.f), 
		glm::vec3(278.f, 273.f, -499.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	m_Camera->UseCameraConfig(0);
	
	glm::vec3 areaLightFrontDir = glm::vec3(0.0f, -1.0f, 0.0f);
	glm::vec3 areaLightPosition = glm::vec3(278.f, 548.78999f, 279.5f);
	
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[0] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[1] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[2] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_pConfManager->GetConfVars()->AreaLightPosX = m_pConfManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_pConfManager->GetConfVars()->AreaLightPosY = m_pConfManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_pConfManager->GetConfVars()->AreaLightPosZ = m_pConfManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	float AreaLightRadianceScale = 100;
	m_pConfManager->GetConfVars()->AreaLightRadianceScale = m_pConfManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;

	m_AreaLight = new AreaLight(130.0f, 105.0f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_pMaterialBuffer);

	m_AreaLight->Init();

	m_AreaLight->SetRadiance(glm::vec3(100.f, 100.f, 100.f));

	m_pConfManager->GetConfVars()->UseIBL = m_pConfManager->GetConfVarsGUI()->UseIBL = 0;
		
	InitKdTree();
}

void Scene::LoadBuddha()
{
	ClearScene();

	float scale = 300.f; 

	CModel* model = new CModel();
	model->Init("dragon43k", "obj", m_pMaterialBuffer);
	model->SetWorldTransform(glm::scale(scale * glm::vec3(1.f, 1.f, 1.f)));

	m_Models.push_back(model);

	m_Camera->Init(0, glm::vec3(0.f, 273.f, -800.f), 
		glm::vec3(0.f, 273.f, -799.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);

	glm::vec3 areaLightFrontDir = glm::vec3(0.0f, -1.0f, 0.0f);
	glm::vec3 areaLightPosition = glm::vec3(0.f, 800.0f, 0.f);
	
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[0] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[1] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[2] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_pConfManager->GetConfVars()->AreaLightPosX = m_pConfManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_pConfManager->GetConfVars()->AreaLightPosY = m_pConfManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_pConfManager->GetConfVars()->AreaLightPosZ = m_pConfManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	float AreaLightRadianceScale = 100;
	m_pConfManager->GetConfVars()->AreaLightRadianceScale = m_pConfManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;

	m_AreaLight = new AreaLight(140.0f, 100.0f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_pMaterialBuffer);

	m_AreaLight->SetRadiance(glm::vec3(AreaLightRadianceScale));

	m_AreaLight->Init();

	m_pConfManager->GetConfVars()->UseIBL = m_pConfManager->GetConfVarsGUI()->UseIBL = 1;
	//m_HasLightSource = false;

	InitKdTree();
}

void Scene::LoadSibernik()
{
	ClearScene();

	float scale = 100.f;

	CModel* model = new CModel();
	model->Init("sibenik", "obj", m_pMaterialBuffer);
	model->SetWorldTransform(glm::scale(glm::vec3(100.f, 100.f, 100.f)));

	m_Models.push_back(model);

	m_Camera->Init(0, scale * glm::vec3(-16.f, -5.f, 0.f), 
		scale * glm::vec3(-15.f, -5.5f, 0.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.f);
			
	glm::vec3 areaLightFrontDir = glm::vec3(0.f, -1.f, 0.f);
	glm::vec3 areaLightPosition = scale * glm::vec3(0.f, 2.f, 0.f);
	
	m_Camera->Init(1, areaLightPosition, 
		areaLightPosition + areaLightFrontDir,
		glm::vec3(1.f, 0.f, 0.f),
		2.f);

	m_Camera->Init(2, glm::vec3(-269.1f, -1266.1f, 239.3f), 
		glm::vec3(-206.1f, -1308.4f, 314.3f),
		glm::vec3(0.f, 1.f, 0.f),
		2.f);

	m_pConfManager->GetConfVars()->AreaLightFrontDirection[0] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[1] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[2] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_pConfManager->GetConfVars()->AreaLightPosX = m_pConfManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_pConfManager->GetConfVars()->AreaLightPosY = m_pConfManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_pConfManager->GetConfVars()->AreaLightPosZ = m_pConfManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	float AreaLightRadianceScale = 3000.f;
	m_pConfManager->GetConfVars()->AreaLightRadianceScale = m_pConfManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;
	
	m_AreaLight = new AreaLight(scale * 0.25f, scale * 0.25f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_pMaterialBuffer);
	
	m_AreaLight->SetRadiance(glm::vec3(AreaLightRadianceScale));

	m_AreaLight->Init();

	InitKdTree();
}

void Scene::LoadCornellBoxDragon()
{
	ClearScene();

	CModel* model = new CModel();
	model->Init("cornell-box-dragon43k", "obj", m_pMaterialBuffer);
	model->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));

	m_Models.push_back(model);
	
	m_Camera->Init(0, glm::vec3(0.0f, 1.0f, 3.5f), 
		glm::vec3(0.0f, 1.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f),
		1.0f);

	glm::vec3 areaLightFrontDir = glm::vec3(0.f, -1.f, 0.f);
	glm::vec3 areaLightPosition = glm::vec3(-0.005f, 1.97f, 0.19f);
	
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[0] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[0] = areaLightFrontDir.x;
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[1] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[1] = areaLightFrontDir.y;
	m_pConfManager->GetConfVars()->AreaLightFrontDirection[2] = m_pConfManager->GetConfVarsGUI()->AreaLightFrontDirection[2] = areaLightFrontDir.z;

	m_pConfManager->GetConfVars()->AreaLightPosX = m_pConfManager->GetConfVarsGUI()->AreaLightPosX = areaLightPosition.x;
	m_pConfManager->GetConfVars()->AreaLightPosY = m_pConfManager->GetConfVarsGUI()->AreaLightPosY = areaLightPosition.y;
	m_pConfManager->GetConfVars()->AreaLightPosZ = m_pConfManager->GetConfVarsGUI()->AreaLightPosZ = areaLightPosition.z;

	float AreaLightRadianceScale = 10;
	m_pConfManager->GetConfVars()->AreaLightRadianceScale = m_pConfManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;

	m_AreaLight = new AreaLight(0.47f, 0.38f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		m_pMaterialBuffer);
	
	m_AreaLight->SetIntensity(glm::vec3(AreaLightRadianceScale));

	m_AreaLight->Init();
		
	InitKdTree();
}


void Scene::InitKdTree()
{
	std::vector<CModel*>::iterator it_models;
	for (it_models = m_Models.begin(); it_models < m_Models.end(); it_models++ )
	{
		CModel* model = *it_models;

		std::vector<CSubModel*> subModels = model->GetSubModels();
		std::vector<CSubModel*>::iterator it_subModels;
		for (it_subModels = subModels.begin(); it_subModels < subModels.end(); it_subModels++ )
		{
			CSubModel* subModel = *it_subModels;
			std::vector<CTriangle*> triangles = subModel->GetTrianglesWS();
			for(uint i = 0; i < triangles.size(); ++i)
			{
				m_Primitives.push_back(triangles[i]);
			}
		}
	}

	// add area light source
	std::vector<CTriangle*> triangles;
	m_AreaLight->GetTrianglesWS(triangles);
	for(uint i = 0; i < triangles.size(); ++i)
	{
		m_Primitives.push_back(triangles[i]);
	}
	
	m_pKdTreeAccelerator = new CKdTreeAccelerator(m_Primitives, 80, 1, 5, 0);
	CTimer timer(CTimer::CPU);
	timer.Start();
	m_pKdTreeAccelerator->BuildTree();
	timer.Stop("BuildAccelerationTree");
}

void Scene::ReleaseKdTree()
{
	if(m_pKdTreeAccelerator)
	{
		delete m_pKdTreeAccelerator;
		m_pKdTreeAccelerator = 0;
	}
	
	m_Primitives.clear();
}

void Scene::UpdateAreaLights()
{
	if(!m_AreaLight)
		return; 
	
	glm::vec3 pos = glm::vec3(
		m_pConfManager->GetConfVars()->AreaLightPosX, 
		m_pConfManager->GetConfVars()->AreaLightPosY, 
		m_pConfManager->GetConfVars()->AreaLightPosZ); 
	m_AreaLight->SetCenterPosition(pos);

	glm::vec3 front = glm::vec3(
		m_pConfManager->GetConfVars()->AreaLightFrontDirection[0],
		m_pConfManager->GetConfVars()->AreaLightFrontDirection[1],
		m_pConfManager->GetConfVars()->AreaLightFrontDirection[2]);
	m_AreaLight->SetFrontDirection(front);

	m_AreaLight->SetRadiance(glm::vec3(m_pConfManager->GetConfVars()->AreaLightRadianceScale));
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
	bool isect = IntersectRaySceneSimple(r, &t, &intersection, CPrimitive::FRONT_FACE);
			
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
	ss.position = m_AreaLight->SamplePos(pdf);
	ss.normal = m_AreaLight->GetFrontDirection();
	ss.pdf = pdf;
	ss.materialIndex = m_AreaLight->GetMaterialIndex();

	if(!(pdf > 0.f))
		std::cout << "Warning: pdf is 0" << std::endl;
}

MATERIAL* Scene::GetMaterial(const SceneSample& ss)
{
	return m_pMaterialBuffer->GetMaterial(ss.materialIndex);
}