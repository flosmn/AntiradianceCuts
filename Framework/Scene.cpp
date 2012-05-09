#include "Scene.h"

typedef unsigned int uint;

#include "Defines.h"
#include "Structs.h"

#include "AreaLight.h"
#include "Light.h"
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
	m_BounceInfo = 0;
	m_pPlaneHammersleySamples = 0;
}

Scene::~Scene()
{
	SAFE_DELETE(m_AreaLight);
	SAFE_DELETE_ARRAY(m_BounceInfo);
}

bool Scene::Init()
{
	m_AreaLight = 0;
	m_MaxBounceInfo = 10;
	m_BounceInfo = new int[m_MaxBounceInfo];
	for(int i = 0; i < m_MaxBounceInfo; ++i) {
		m_BounceInfo[i] = 0;
	}
	m_MaxVPLFlow = glm::vec3(0.f);
	m_MaxVPLFlowBounce = 0;
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
	if(m_BounceInfo != 0)
		delete [] m_BounceInfo;
	
	m_BounceInfo = new int[m_MaxBounceInfo];
	memset(m_BounceInfo, 0, m_MaxBounceInfo * sizeof(int));
				
	m_MaxVPLFlow = glm::vec3(0.f);
	m_MaxVPLFlowBounce = 0;
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

void Scene::DrawAreaLight(CGLUniformBuffer* pUBTransform)
{
	m_AreaLight->Draw(m_Camera, pUBTransform);
}

Light* Scene::CreateLight(Light* tail, int N, int nAdditionalAVPLs, bool useHammersley)
{
	Light* head = 0;
	if(tail == 0)
	{
		// create VPL on light source
		float pdf;
		glm::vec3 pos = m_AreaLight->SamplePos(pdf);
		glm::vec3 ori = m_AreaLight->GetFrontDirection();
		glm::vec3 rad = m_AreaLight->GetRadiance();

		if(!(pdf > 0.f))
			return head;

		head = new Light(pos, ori, rad / pdf, glm::vec3(0), glm::vec3(0), glm::vec3(0), 0);
	}
	else
	{
		// create new path segment starting from tail with cos-sampled direction 
		// (importance sample diffuse surface)
		float pdf = 0.f;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(tail->GetOrientation(), Rand01(), Rand01(), pdf, 1);
		
		head = CreateLight(tail, direction, pdf, N, nAdditionalAVPLs, useHammersley);
	}
	return head;
}

Light* Scene::CreateLight(Light* tail, glm::vec3 direction, float pdf, int N, int nAdditionalAVPLs, bool useHammersley)
{
	Light* head = 0;

	glm::vec3 origin = tail->GetPosition();

	const float epsilon = 0.0005f;
	Ray ray(origin + epsilon * direction + epsilon * tail->GetOrientation(), direction);
		
	Intersection intersection;
	if(IntersectRayScene(ray, intersection)) 
	{
		// gather information for the new VPL
		glm::vec3 pos = intersection.GetPoint();
		glm::vec3 ori = intersection.GetTriangle().GetNormal();
		glm::vec3 rho = glm::vec3(intersection.GetModel()->GetMaterial().diffuseColor);

		float cos_theta = glm::dot(glm::normalize(ori), glm::normalize(-direction));
		
		glm::vec3 contrib = rho/PI * cos_theta/pdf * tail->GetContrib();
		
		glm::vec3 src_pos = tail->GetPosition();
		glm::vec3 src_orientation = tail->GetOrientation();
		
		const float area = 2 * PI * ( 1 - cos(PI/float(N+1)) );
		const float dist = glm::length(pos - src_pos);

		glm::vec3 src_contrib = 1.f / float(nAdditionalAVPLs + 1) * tail->GetContrib() / (pdf * area);
		
		if(pdf < 0.00001)
		{
			std::cout << "pdf small, discard vpl. pdf:" << pdf << std::endl;
			return 0;
		}
		else if(glm::length(contrib) > 10000.f)
		{
			std::cout << "contrib high, discard vpl. contrib: " << AsString(contrib) << std::endl;
			return 0;
		}
		else if(glm::length(src_contrib) > 100000.f)
		{
			std::cout << "antiradiance high, discard vpl. antirad: " << AsString(src_contrib) << std::endl;
			return 0;
		}
		else{
			head = new Light(pos, ori, contrib, src_pos, src_orientation, src_contrib, tail->GetBounce() + 1);
		}
	}

	return head;
}

std::vector<Light*> Scene::CreatePaths(uint numPaths, int N, int nAdditionalAVPLs, bool useHammersley)
{
	std::vector<Light*> lights;
	for(uint i = 0; i < numPaths; ++i)
	{
		std::vector<Light*> path = CreatePath(N, nAdditionalAVPLs, useHammersley);
		for(uint j = 0; j < path.size(); ++j)
			lights.push_back(path[j]);
	}

	return lights;
}

std::vector<Light*> Scene::CreatePath(int N, int nAdditionalAVPLs, bool useHammersley) 
{
	m_NumLightPaths++;

	m_CurrentBounce = 0;
	std::vector<Light*> path;
	
	Light *tail, *head;
		
	// create new primary light on light source
	tail = CreateLight(0, N, nAdditionalAVPLs, useHammersley);
	path.push_back(tail);
	
	if(m_CurrentBounce < m_MaxBounceInfo)
		m_BounceInfo[m_CurrentBounce]++;
	m_CurrentBounce++;
	
	// follow light path until it is terminated
	// by RR with termination probability 1-rrProb
	bool terminate = false;
	const float rrProb = 0.8f;
	while(!terminate)
	{
		// decide whether to terminate path
		float rand_01 = glm::linearRand(0.f, 1.f);
		if(rand_01 > rrProb || m_CurrentBounce >= 3)
		{
			// create path-finishing Anti-VPLs
			CreateAVPLs(tail, path, N, nAdditionalAVPLs, useHammersley);

			Light* avpl = CreateLight(tail, N, nAdditionalAVPLs, useHammersley);
			if(avpl != 0)
			{
				avpl->SetContrib(glm::vec3(0.f));
				path.push_back(avpl);
			}
			
			terminate = true;
		}
		else
		{
			// create additional avpls
			CreateAVPLs(tail, path, N, nAdditionalAVPLs, useHammersley);

			// follow the path with cos-sampled direction (importance sample diffuse surface)
			// if the ray hits geometry
			head = CreateLight(tail, N, nAdditionalAVPLs, useHammersley);
			if(head != 0)
			{
				head->SetContrib(head->GetContrib() / rrProb);
				path.push_back(head);
				tail = head;
			}
			else
			{
				// if the ray hits no geometry the transpored energy
				// goes to nirvana and is lost
				terminate = true;
			}
		}

		if(m_CurrentBounce < m_MaxBounceInfo)
		m_BounceInfo[m_CurrentBounce]++;
		m_CurrentBounce++;
	}

	return path;
}

void Scene::CreateAVPLs(Light* tail, std::vector<Light*>& path, int N, int nAVPLs, bool useHammersley)
{
	glm::vec3 orientation = tail->GetOrientation();
	float deltaX = 0.f;
	float deltaY = 0.f;

	for(int i = 0; i < nAVPLs; ++i)
	{
		float pdf;
		glm::vec3 direction;

		if(useHammersley)
		{
			float u1 = fmod(deltaX + m_pPlaneHammersleySamples[2 * i + 0], 1.f);
			float u2 = fmod(deltaY + m_pPlaneHammersleySamples[2 * i + 1], 1.f);
			
			direction = GetRandomSampleDirectionCosCone(orientation, u1, u2, pdf, 1);
		}
		else
		{
			direction = GetRandomSampleDirectionCosCone(orientation, Rand01(), Rand01(), pdf, 1);
		}

		Light* avpl = CreateLight(tail, direction, pdf, N, nAVPLs, useHammersley);
		if(avpl != 0)
		{
			avpl->SetContrib(glm::vec3(0.f));
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
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(120.0f, 120.0f, 120.0f));
	
	m_AreaLight->Init();
}

void Scene::LoadCornellBox()
{
	ClearScene();

	CModel* model = new CModel();
	model->Init("cornell");
	model->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));

	m_Models.push_back(model);

	m_Camera->Init(glm::vec3(2.78f, 2.73f, 8.0f), 
		glm::vec3(2.8f, 2.73f, -2.8f),
		2.0f);
		
	m_AreaLight = new AreaLight(1.1f, 1.1f, 
		glm::vec3(2.75f, 5.49f, -2.75f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(120.0f, 120.0f, 120.0f));
	
	m_AreaLight->Init();
}

void Scene::Stats() {
	std::cout << "information abount VPLs: " << std::endl;
	std::cout << "primary lights: " << m_BounceInfo[0] << std::endl;
	for(int i = 1; i < m_MaxBounceInfo; ++i)
	{
		std::cout << i << ". indirection lights: " << m_BounceInfo[i] << std::endl;
	}
	std::cout << "max flow (VPL) : (" << m_MaxVPLFlow.r << ", " << m_MaxVPLFlow.g << ", " << m_MaxVPLFlow.b << ")" << std::endl;
	std::cout << "max flow (VPL) bounce : " << m_MaxVPLFlowBounce << std::endl;
}

void Scene::CreatePlaneHammersleySamples(int i)
{
	if(m_pPlaneHammersleySamples)
		SAFE_DELETE_ARRAY(m_pPlaneHammersleySamples);

	m_pPlaneHammersleySamples = new float[2* i];

	PlaneHammersley(m_pPlaneHammersleySamples, i);
}