#include "Scene.h"

#include "Defines.h"
#include "Structs.h"

#include "AreaLight.h"
#include "Light.h"
#include "Camera.h"

#include "CGLResources\CGLUniformBuffer.h"

#include "CMeshResources\CMesh.h"
#include "CMeshResources\CModel.h"
#include "CMeshResources\CSubModel.h"

#include <iostream>
#include <algorithm>

Scene::Scene(Camera* _camera)
{
	m_Camera = _camera;
	m_AvgReflectivity = glm::vec3(0.0f);
	m_CurrentBounce = 0;
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

	return true;
}

void Scene::Release()
{
	m_AreaLight->Release();

	std::vector<CModel*>::iterator it;
	for ( it=m_Models.begin() ; it < m_Models.end(); it++ )
	{
		CModel* model = *it;
		model->Release();
		SAFE_DELETE(model);
	}
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

	std::vector<Light*>::iterator it_light;

	for ( it_light=m_Lights.begin() ; it_light < m_Lights.end(); it_light++ )
	{
		SAFE_DELETE(*it_light);
	}

	m_Lights.clear();

	m_AvgReflectivity = glm::vec3(0.0f);
}

 void Scene::ClearLighting()
{
	std::vector<Light*>::iterator it_lights;

	/* crashes on pressing "c"
	for ( it_lights=m_Lights.begin() ; it_lights < m_Lights.end(); it_lights++ )
	{
		Light* light = *it_lights;
		if(light != NULL)
			delete light;
	}
	*/

	m_Lights.clear();
	m_Paths.clear();
	m_CurrentPath.clear();

	SAFE_DELETE_ARRAY(m_BounceInfo);
	m_BounceInfo = new int[m_MaxBounceInfo];
	for(int i = 0; i < m_MaxBounceInfo; ++i) {
		m_BounceInfo[i] = 0;
	}
	m_MaxVPLFlow = glm::vec3(0.f);
	m_MaxVPLFlowBounce = 0;
	m_CurrentBounce = 0;
}

void Scene::ClearPath()
{
	m_CurrentPath.clear();
	m_CurrentBounce = 0;
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

Light* Scene::CreateLight(Light* tail)
{
	Light* head = 0; 
	if(tail == 0)
	{
		// create new primary light source
		float pdf;
		head = m_AreaLight->GetNewPrimaryLight(pdf);
	}
	else
	{
		// create new path segment starting from tail with cos-sampled direction 
		// (importance sample diffuse surface)
		glm::vec3 origin = tail->GetPosition();
		float pdf;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(tail->GetOrientation(), pdf, 1);
		
		Ray ray(origin, direction);
		Intersection intersection;
		if(IntersectRayScene(ray, intersection)) 
		{
			// gather information for the new VPL
			glm::vec3 pos = intersection.GetPoint();
			glm::vec3 orientation = intersection.GetTriangle().GetNormal();
			glm::vec3 rho = glm::vec3(intersection.GetModel()->GetMaterial().diffuseColor);
			//glm::vec3 flux = rho/m_MeanRho * tail->GetFlux();
			glm::vec3 flux = glm::min(rho/m_MeanRho, glm::vec3(1.f, 1.f, 1.f)) * tail->GetFlux();
			
			glm::vec3 src_pos = tail->GetPosition();
			glm::vec3 src_orientation = tail->GetOrientation();
			
			float dist = glm::length(pos - src_pos);
			float cos_theta_i = glm::dot(glm::normalize(-direction), glm::normalize(orientation));
			
			glm::vec3 src_flux = 2 * PI * (dist * dist) / (cos_theta_i) * tail->GetFlux();
			
			head = new Light(pos, orientation, flux, src_pos, src_orientation, src_flux);
		}
	}

	if(!head) return head;

	// set debug info
	if(m_CurrentBounce == 1){
		head->SetDebugColor(glm::vec3(0.8, 0.2, 0.2));
	} else if(m_CurrentBounce == 2){
		head->SetDebugColor(glm::vec3(0.2, 0.8, 0.2));
	} else {
		head->SetDebugColor(glm::vec3(0.2, 0.2, 0.2));
	}
	if(m_CurrentBounce < m_MaxBounceInfo){
		m_BounceInfo[m_CurrentBounce]++;
	}

	m_CurrentPath.push_back(head);
	m_Lights.push_back(head);			
	m_CurrentBounce++;

	return head;
}

std::vector<Light*> Scene::CreatePath() 
{
	// first clear the information about and old path
	ClearPath();

	Light *tail, *head;
		
	// create new primary light on light source
	tail = CreateLight(0);
	m_CurrentPath.push_back(tail);
	m_Lights.push_back(tail);
	m_BounceInfo[0] ++;

	m_CurrentBounce = 1;

	// follow light path until it is terminated
	// by RR with termination probability 1-rho_avg
	while(true)
	{
		// decide whether to terminate path
		float rand_01 = glm::linearRand(0.f, 1.f);
		if(rand_01 > m_MeanRho)
		{
			// create finishing Anti-VPL
			Light* finish = CreateLight(tail);
			if(finish){
				finish->SetFlux(glm::vec3(0.f, 0.f, 0.f));
				finish->SetDebugColor(glm::vec3(0.f, 0.0f, 0.8f));
			}
			m_Paths.push_back(m_CurrentPath);
			return m_CurrentPath;
		}		
		
		// follow the path with cos-sampled direction (importance sample diffuse surface)
		// if the ray hits geometry
		head = CreateLight(tail);
		if(head)
		{
			tail = head;
		}
		else
		{
			// if the ray hits no geometry the transpored energy
			// goes to nirvana and is lost
			m_Paths.push_back(m_CurrentPath);
			return m_CurrentPath;
		}
	}
	return m_CurrentPath;
}

std::vector<Light*> Scene::CreatePathPBRT()
{
	float epsilon = 0.0005f;
	ClearPath();
	m_CurrentBounce = 0;
	
	// Follow path _i_ from light to create virtual lights
	float pdf;
	Light *tail = m_AreaLight->GetNewPrimaryLight(pdf);
	m_Lights.push_back(tail);
	m_CurrentPath.push_back(tail);

	glm::vec3 alpha = tail->GetFlux() /*/ pdf*/; // pdf already considered in radiance
	
	// Sample ray leaving light source for virtual light path
	glm::vec3 direction = GetRandomSampleDirectionCosCone(tail->GetOrientation(), pdf, 1);
	glm::vec3 origin = tail->GetPosition();
	    
	if (pdf == 0.f || alpha.length() == 0)
	{
		m_Paths.push_back(m_CurrentPath);
		return m_CurrentPath;
	}
	
	alpha /= pdf;
    
	Ray ray(origin + epsilon * direction + epsilon * tail->GetOrientation(), direction);
	Intersection intersection;
	while(IntersectRayScene(ray, intersection) && m_CurrentBounce < 1) 
	{
		m_CurrentBounce++;

		// Create virtual light and sample new ray for path
		glm::vec3 albedo = glm::vec3(intersection.GetModel()->GetMaterial().diffuseColor);
		glm::vec3 pos = intersection.GetPoint();
		glm::vec3 normal = intersection.GetTriangle().GetNormal();
		
		// Create virtual light at ray intersection point
        glm::vec3 contrib = albedo / PI * alpha;
		Light* head = new Light(pos, normal, contrib, tail->GetPosition(),
			tail->GetOrientation(), tail->GetFlux());
		m_Lights.push_back(head);
		m_CurrentPath.push_back(head);
		
		// Sample new ray direction and update weight for virtual light path
		float pdf;
		glm::vec3 direction = GetRandomSampleDirectionCosCone(head->GetOrientation(), pdf, 1);
		
		glm::vec3 fr = albedo;           
		if (fr.length() == 0 || pdf == 0.f)
		{
			head->SetFlux(glm::vec3(0));
			SetDebugColor(head, m_CurrentBounce);
			m_Paths.push_back(m_CurrentPath);
			return m_CurrentPath;
		}
        
		glm::vec3 contribScale = fr / pdf * glm::dot(direction, normal);
		
		// Possibly terminate virtual light path with Russian roulette
		float rrProb = std::min(1.f, 1.f/3.f * ( contribScale.r + contribScale.g + contribScale.b));
		
		float rand_01 = glm::linearRand(0.f, 1.f);
		if (rand_01 > rrProb || m_CurrentBounce > 4)
		{
			head->SetFlux(glm::vec3(0));
			SetDebugColor(head, m_CurrentBounce);
			m_Paths.push_back(m_CurrentPath);
			return m_CurrentPath;
		}
		
		alpha *= contribScale / rrProb;
		ray = Ray(pos + epsilon * direction + epsilon * normal, direction);

		SetDebugColor(head, m_CurrentBounce);
		tail = head;
	}

	// make last vpl to pure antiradiance vpl
	tail->SetFlux(glm::vec3(0));

	return m_CurrentPath;
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
		
	m_MeanRho = 0.5f;

	m_Models.push_back(model);

	m_Camera->Init(glm::vec3(-9.2f, 5.7f, 6.75f), 
		glm::vec3(0.f, -2.f, 0.f),
		2.0f);
	
	m_AreaLight = new AreaLight(0.5f, 0.5f, 
		glm::vec3(0.0f, 5.f, 0.0f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(1500.0f, 1500.0f, 1500.0f));
	
	m_AreaLight->Init();
}

void Scene::LoadCornellBox()
{
	ClearScene();

	CModel* model = new CModel();
	model->Init("cornell-fine");
	model->SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));
		
	m_MeanRho = 0.5f;

	m_Models.push_back(model);

	m_Camera->Init(glm::vec3(2.78f, 2.73f, 8.0f), 
		glm::vec3(2.8f, 2.73f, -2.8f),
		2.0f);
		
	m_AreaLight = new AreaLight(1.1f, 1.1f, 
		glm::vec3(2.75f, 5.5f, -2.75f), 
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

void Scene::SetDebugColor(Light* light, int bounce)
{
	if(bounce == 1){
		light->SetDebugColor(glm::vec3(0.8, 0.2, 0.2));
	} else if(bounce == 2){
		light->SetDebugColor(glm::vec3(0.2, 0.8, 0.2));
	} else if(bounce == 3) {
		light->SetDebugColor(glm::vec3(0.2, 0.2, 0.8));
	} else {
		light->SetDebugColor(glm::vec3(0.2, 0.2, 0.2));
	}
}