#include "Scene.h"

typedef unsigned int uint;

#include "Defines.h"
#include "Structs.h"

#include "AVPL.h"
#include "AreaLight.h"
#include "CCamera.h"
#include "CKdTreeAccelerator.h"
#include "CPbrtKdTreeAccelerator.h"
#include "CPrimitive.h"

#include "OGLResources\COGLUniformBuffer.h"

#include "MeshResources\CMesh.h"
#include "MeshResources\CModel.h"
#include "MeshResources\CSubModel.h"

#include "MeshResources\CMeshMaterial.h"

#include <set>
#include <iostream>
#include <algorithm>

Scene::Scene(CCamera* _camera)
{
	m_Camera = _camera;
	m_CurrentBounce = 0;
	m_pKdTreeAccelerator = 0;
	m_pPbrtKdTreeAccelerator = 0;
}

Scene::~Scene()
{
	SAFE_DELETE(m_AreaLight);
}

bool Scene::Init()
{
	m_AreaLight = 0;
	m_NumLightPaths = 0;

	return true;
}

void Scene::Release()
{
	m_AreaLight->Release();
	
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
	m_AreaLight->Draw(m_Camera, pUBTransform, pUBAreaLight);
}

AVPL* Scene::CreateAVPL(AVPL* predecessor, int N, int nAdditionalAVPLs)
{
	AVPL* avpl = 0;
	if(predecessor == 0)
	{
		// create VPL on light source
		float pdf;
		glm::vec3 pos = m_AreaLight->SamplePos(pdf);
		glm::vec3 ori = m_AreaLight->GetFrontDirection();
		glm::vec3 I = m_AreaLight->GetRadiance();

		if(!(pdf > 0.f))
			return avpl;

		avpl = new AVPL(pos, ori, I / pdf, glm::vec3(0), glm::vec3(0), 0);
	}
	else
	{
		float pdf = 0.f;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(predecessor->GetOrientation(), Rand01(), Rand01(), pdf, 1);
		
		avpl = ContinueAVPLPath(predecessor, direction, pdf, N, nAdditionalAVPLs);
	}
	return avpl;
}

AVPL* Scene::ContinueAVPLPath(AVPL* pred, glm::vec3 direction, float pdf, int N, int nAdditionalAVPLs)
{
	AVPL* avpl = 0;

	glm::vec3 pred_pos = pred->GetPosition();

	const glm::vec3 pred_norm = glm::normalize(pred->GetOrientation());
	direction = glm::normalize(direction);

	const float epsilon = 0.005f;
	Ray ray(pred_pos + epsilon * direction, direction);
		
	Intersection intersection;
	float t = 0;
	if(m_pKdTreeAccelerator->Intersect(ray, &t, &intersection))
	{
		// gather information for the new VPL
		glm::vec3 pos = intersection.GetPosition();
		glm::vec3 norm = intersection.GetPrimitive()->GetNormal();
		glm::vec3 rho = glm::vec3(intersection.GetPrimitive()->GetModel()->GetMaterial().diffuseColor);
				
		glm::vec3 intensity = rho/PI * pred->GetIntensity(glm::normalize(direction)) / pdf;	
		
		const float area = 2 * PI * ( 1 - cos(PI/float(N)) );
		
		glm::vec3 antiintensity = 1.f / float(nAdditionalAVPLs + 1) * pred->GetIntensity(glm::normalize(direction)) / (pdf * area);
			
		avpl = new AVPL(pos, norm, intensity, antiintensity, direction, pred->GetBounce() + 1);
	}

	return avpl;
}

std::vector<AVPL*> Scene::CreatePaths(uint numPaths, int N, int nAdditionalAVPLs)
{
	std::vector<AVPL*> avpls;
	for(uint i = 0; i < numPaths; ++i)
	{
		std::vector<AVPL*> path = CreatePath(N, nAdditionalAVPLs);
		for(uint j = 0; j < path.size(); ++j)
			avpls.push_back(path[j]);
	}

	return avpls;
}

std::vector<AVPL*> Scene::CreatePath(int N, int nAdditionalAVPLs) 
{
	m_NumLightPaths++;

	m_CurrentBounce = 0;
	std::vector<AVPL*> path;
	
	AVPL *pred, *succ;
		
	// create new primary light on light source
	pred = CreateAVPL(0, N, nAdditionalAVPLs);
	path.push_back(pred);
	
	m_CurrentBounce++;
	
	// follow light path until it is terminated
	// by RR with termination probability 1-rrProb
	bool terminate = false;
	const float rrProb = 0.8f;
	while(!terminate)
	{
		// decide whether to terminate path
		float rand_01 = glm::linearRand(0.f, 1.f);
		if(rand_01 > rrProb)
		{
			// create path-finishing Anti-VPLs
			CreateAVPLs(pred, path, N, nAdditionalAVPLs);

			AVPL* avpl = CreateAVPL(pred, N, nAdditionalAVPLs);
			if(avpl != 0)
			{
				avpl->SetIntensity(glm::vec3(0.f));
				path.push_back(avpl);
			}
			
			terminate = true;
		}
		else
		{
			// create additional avpls
			CreateAVPLs(pred, path, N, nAdditionalAVPLs);

			// follow the path with cos-sampled direction (importance sample diffuse surface)
			// if the ray hits geometry
			succ = CreateAVPL(pred, N, nAdditionalAVPLs);
			if(succ != 0)
			{
				succ->SetIntensity(succ->GetIntensity(succ->GetOrientation()) / rrProb);
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

		m_CurrentBounce++;
	}

	return path;
}

void Scene::CreateAVPLs(AVPL* pred, std::vector<AVPL*>& path, int N, int nAVPLs)
{
	glm::vec3 pred_norm = pred->GetOrientation();

	for(int i = 0; i < nAVPLs; ++i)
	{
		float pdf;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(pred_norm, Rand01(), Rand01(), pdf, 1);
		
		AVPL* avpl = ContinueAVPLPath(pred, direction, pdf, N, nAVPLs);
		if(avpl != 0)
		{
			avpl->SetIntensity(glm::vec3(0.f));
			path.push_back(avpl);
		}
	}
}

std::vector<AVPL*> Scene::CreatePrimaryVpls(int numVpls)
{
	std::vector<AVPL*> vpls;
	for(int i = 0; i < numVpls; ++i)
	{
		AVPL* avpl = CreateAVPL(0, 30, 0);
		avpl->SetIntensity(1.f / float(numVpls) * avpl->GetIntensity(avpl->GetOrientation()));
		vpls.push_back(avpl);
	}

	return vpls;
}

bool Scene::IntersectRaySceneSimple(const Ray& ray, float* t, Intersection *pIntersection)
{	
	float t_best = std::numeric_limits<float>::max();
	Intersection isect_best;
	bool hasIntersection = false;

	for(uint i = 0; i < m_Primitives.size(); ++i)
	{
		float t_temp = 0.f;
		Intersection isect_temp;
		if(m_Primitives[i]->Intersect(ray, &t_temp, &isect_temp))
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
		
	return hasIntersection;
}

void Scene::LoadSimpleScene()
{
	ClearScene();

	CModel* model = new CModel();
	model->Init("twoplanes");
	model->SetWorldTransform(glm::scale(glm::vec3(2.f, 2.f, 2.f)));
		
	m_Models.push_back(model);

	m_Camera->Init(glm::vec3(-9.2f, 5.7f, 6.75f), 
		glm::vec3(0.f, -2.f, 0.f),
		2.0f);
	
	m_AreaLight = new AreaLight(0.5f, 0.5f, 
		glm::vec3(0.0f, 4.f, 0.0f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f));

	m_AreaLight->SetRadiance(glm::vec3(120.f, 120.f, 120.f));
	
	m_AreaLight->Init();

	InitKdTree();
}

void Scene::LoadCornellBox()
{
	ClearScene();

	CModel* model = new CModel();
	model->Init("cornellorg-boxes");
	model->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));

	m_Models.push_back(model);

	m_Camera->Init(glm::vec3(278.f, 273.f, -800.f), 
		glm::vec3(278.f, 273.f, 270.f),
		2.0f);
	
	m_AreaLight = new AreaLight(140.0f, 100.0f, 
		glm::vec3(270.f, 550.0f, 280.f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f));
	
	m_AreaLight->SetRadiance(glm::vec3(100.f, 100.f, 100.f));

	m_AreaLight->Init();

	InitKdTree();
}

void Scene::LoadSibernik()
{
	ClearScene();

	CModel* model = new CModel();
	model->Init("sibmaxexp");
	model->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));

	m_Models.push_back(model);

	m_Camera->Init(glm::vec3(-16.f, -5.f, 0.f), 
		glm::vec3(0.f, -10.f, 0.f),
		1.0f);
	
	m_AreaLight = new AreaLight(0.25f, 0.25f, 
		glm::vec3(0.f, 2.0f, 0.f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f));
	
	m_AreaLight->SetRadiance(glm::vec3(8000.f, 8000.f, 8000.f));

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

	m_pPbrtKdTreeAccelerator = new CPbrtKdTreeAccel(m_Primitives, 80, 1, 5, 0);
	
	m_pKdTreeAccelerator = new CKdTreeAccelerator(m_Primitives, 80, 1, 5, 0);
	m_pKdTreeAccelerator->BuildTree();
}

void Scene::ReleaseKdTree()
{
	if(m_pKdTreeAccelerator)
	{
		delete m_pKdTreeAccelerator;
		m_pKdTreeAccelerator = 0;
	}

	if(m_pPbrtKdTreeAccelerator)
	{
		delete m_pPbrtKdTreeAccelerator;
		m_pPbrtKdTreeAccelerator = 0;
	}

	m_Primitives.clear();
}