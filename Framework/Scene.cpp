#include "Scene.h"

typedef unsigned int uint;

#include "Defines.h"
#include "Structs.h"

#include "AVPL.h"
#include "AreaLight.h"
#include "Camera.h"

#include "CGLResources\CGLUniformBuffer.h"

#include "CMeshResources\CMesh.h"
#include "CMeshResources\CModel.h"
#include "CMeshResources\CSubModel.h"

#include <set>
#include <iostream>
#include <algorithm>

Scene::Scene(Camera* _camera)
{
	m_Camera = _camera;
	m_CurrentBounce = 0;
	m_pPlaneHammersleySamples = 0;
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

void Scene::DrawScene(CGLUniformBuffer* pUBTransform, CGLUniformBuffer* pUBMaterial)
{
	std::vector<CModel*>::iterator it;

	for ( it=m_Models.begin() ; it < m_Models.end(); it++ )
	{
		(*it)->Draw(m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix(), pUBTransform, pUBMaterial);
	}
}

void Scene::DrawScene(CGLUniformBuffer* pUBTransform)
{
	std::vector<CModel*>::iterator it;

	for ( it=m_Models.begin() ; it < m_Models.end(); it++ )
	{
		(*it)->Draw(m_Camera->GetViewMatrix(), m_Camera->GetProjectionMatrix(), pUBTransform);
	}
}

void Scene::DrawScene(const glm::mat4& mView, const glm::mat4& mProj, CGLUniformBuffer* pUBTransform)
{
	std::vector<CModel*>::iterator it;

	for ( it=m_Models.begin() ; it < m_Models.end(); it++ )
	{
		(*it)->Draw(mView, mProj, pUBTransform);
	}
}

void Scene::DrawAreaLight(CGLUniformBuffer* pUBTransform, CGLUniformBuffer* pUBAreaLight)
{
	m_AreaLight->Draw(m_Camera, pUBTransform, pUBAreaLight);
}

AVPL* Scene::CreateAVPL(AVPL* predecessor, int N, int nAdditionalAVPLs, bool useHammersley)
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
		
		avpl = ContinueAVPLPath(predecessor, direction, pdf, N, nAdditionalAVPLs, useHammersley);
	}
	return avpl;
}

AVPL* Scene::ContinueAVPLPath(AVPL* pred, glm::vec3 direction, float pdf, int N, int nAdditionalAVPLs, bool useHammersley)
{
	AVPL* avpl = 0;

	glm::vec3 pred_pos = pred->GetPosition();

	const glm::vec3 pred_norm = glm::normalize(pred->GetOrientation());
	direction = glm::normalize(direction);

	const float epsilon = 0.0005f;
	Ray ray(pred_pos + epsilon * direction + epsilon * pred_norm, direction);
		
	Intersection intersection;
	if(IntersectRayScene(ray, intersection)) 
	{
		// gather information for the new VPL
		glm::vec3 pos = intersection.GetPoint();
		glm::vec3 norm = intersection.GetTriangle().GetNormal();
		glm::vec3 rho = glm::vec3(intersection.GetModel()->GetMaterial().diffuseColor);

		const float cos_theta_yz = glm::dot(glm::normalize(pred_norm), glm::normalize(direction));

		glm::vec3 intensity = rho/PI * cos_theta_yz/pdf * pred->GetIntensity();	
		
		const float area = 2 * PI * ( 1 - cos(PI/float(N+1)) );
		
		glm::vec3 antiintensity = 1.f / float(nAdditionalAVPLs + 1) * pred->GetIntensity() * cos_theta_yz / (pdf * area);
			
		avpl = new AVPL(pos, norm, intensity, antiintensity, direction, pred->GetBounce() + 1);	
	}

	return avpl;
}

std::vector<AVPL*> Scene::CreatePaths(uint numPaths, int N, int nAdditionalAVPLs, bool useHammersley)
{
	std::vector<AVPL*> avpls;
	for(uint i = 0; i < numPaths; ++i)
	{
		std::vector<AVPL*> path = CreatePath(N, nAdditionalAVPLs, useHammersley);
		for(uint j = 0; j < path.size(); ++j)
			avpls.push_back(path[j]);
	}

	return avpls;
}

std::vector<AVPL*> Scene::CreatePath(int N, int nAdditionalAVPLs, bool useHammersley) 
{
	m_NumLightPaths++;

	m_CurrentBounce = 0;
	std::vector<AVPL*> path;
	
	AVPL *pred, *succ;
		
	// create new primary light on light source
	pred = CreateAVPL(0, N, nAdditionalAVPLs, useHammersley);
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
			CreateAVPLs(pred, path, N, nAdditionalAVPLs, useHammersley);

			AVPL* avpl = CreateAVPL(pred, N, nAdditionalAVPLs, useHammersley);
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
			CreateAVPLs(pred, path, N, nAdditionalAVPLs, useHammersley);

			// follow the path with cos-sampled direction (importance sample diffuse surface)
			// if the ray hits geometry
			succ = CreateAVPL(pred, N, nAdditionalAVPLs, useHammersley);
			if(succ != 0)
			{
				succ->SetIntensity(succ->GetIntensity() / rrProb);
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

void Scene::CreateAVPLs(AVPL* pred, std::vector<AVPL*>& path, int N, int nAVPLs, bool useHammersley)
{
	glm::vec3 pred_norm = pred->GetOrientation();

	for(int i = 0; i < nAVPLs; ++i)
	{
		float pdf;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(pred_norm, Rand01(), Rand01(), pdf, 1);

		AVPL* avpl = ContinueAVPLPath(pred, direction, pdf, N, nAVPLs, useHammersley);
		if(avpl != 0)
		{
			avpl->SetIntensity(glm::vec3(0.f));
			path.push_back(avpl);
		}
	}
}

bool Scene::IntersectRayScene(Ray ray, Intersection &intersection)
{	
	float t = 1000000.0f;
	bool hasIntersection = false;;
	
	std::vector<CModel*>::iterator it_models;
	for (it_models = m_Models.begin(); it_models < m_Models.end(); it_models++ )
	{
		CModel* model = *it_models;

		std::vector<CSubModel*> subModels = model->GetSubModels();
		std::vector<CSubModel*>::iterator it_subModels;
		
		for (it_subModels = subModels.begin(); it_subModels < subModels.end(); it_subModels++ )
		{
			CSubModel* subModel = *it_subModels;

			std::vector<Triangle*> triangles = subModel->GetTriangles();
			glm::mat4 transform = model->GetWorldTransform();

			std::vector<Triangle*>::iterator it_triangle;
			for(it_triangle = triangles.begin(); it_triangle < triangles.end(); it_triangle++)
			{
				Triangle triangle = (*it_triangle)->GetTransformedTriangle(transform);
			
				bool intersectionBB = IntersectWithBB(triangle, ray);
				if(intersectionBB)
				{
					glm::vec3 v0 = triangle.GetPoints()[0];
					glm::vec3 v1 = triangle.GetPoints()[1];
					glm::vec3 v2 = triangle.GetPoints()[2];

					glm::vec3 origin = ray.GetOrigin();
					glm::vec3 direction = glm::normalize(ray.GetDirection());

					float temp = -1.0f;
					if(IntersectRayTriangle(origin, direction, v0, v1, v2, temp))
					{
						if(temp < t && temp > 0) {
							t = temp;
							glm::vec3 position = origin + t * direction;
							intersection = Intersection(subModel, triangle, position);
							hasIntersection = true;
						}
					}
					if(IntersectRayTriangle(origin, direction, v0, v2, v1, temp))
					{
						if(temp < t && temp > 0) {
							t = temp;
							glm::vec3 position = origin + t * direction;
							intersection = Intersection(subModel, triangle, position);
							hasIntersection = true;
						}
					}
				}
			}
		}
	}
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

	m_AreaLight->SetFlux(glm::vec3(120.f, 120.f, 120.f));
	
	m_AreaLight->Init();
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
}

void Scene::CreatePlaneHammersleySamples(int i)
{
	if(m_pPlaneHammersleySamples)
		SAFE_DELETE_ARRAY(m_pPlaneHammersleySamples);

	m_pPlaneHammersleySamples = new float[2* i];

	PlaneHammersley(m_pPlaneHammersleySamples, i);
}