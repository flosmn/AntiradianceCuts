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
		head = m_AreaLight->GetNewPrimaryLight();
	}
	else
	{
		// create new path segment starting from tail with cos-sampled direction 
		// (importance sample diffuse surface)
		glm::vec3 origin = tail->GetPosition();
		glm::vec3 direction = GetRandomSampleDirectionCosCone(tail->GetOrientation(), 1);
		
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
			glm::vec3 src_flux = 4.f * PI * tail->GetFlux();		
			
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

void Scene::CreatePath() 
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
		if(m_CurrentBounce > 0.f)
		{
			// create finishing Anti-VPL
			Light* finish = CreateLight(tail);
			if(finish){
				finish->SetFlux(glm::vec3(0.f, 0.f, 0.f));
				finish->SetDebugColor(glm::vec3(0.f, 0.0f, 0.8f));
			}
			m_Paths.push_back(m_CurrentPath);
			return;
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
			return;
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
							intersection = Intersection(model, triangle, position);
							hasIntersection = true;
						}
					}
					if(IntersectRayTriangle(origin, direction, v0, v2, v1, temp))
					{
						if(temp < t && temp > 0) {
							t = temp;
							glm::vec3 position = origin + t * direction;
							intersection = Intersection(model, triangle, position);
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

	m_Camera->Init(glm::vec3(-8.f, 6.f, 6.f), 
		glm::vec3(0.f, 0.f, 0.f),
		2.0f);
	
	m_AreaLight = new AreaLight(1.0f, 1.0f, 
		glm::vec3(0.0f, 5.f, 0.0f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(150.0f, 150.0f, 150.0f));
	
	m_AreaLight->Init();
}


void Scene::LoadCornellBox()
{
	MATERIAL* white = new MATERIAL;
	white->diffuseColor = glm::vec4(1.f, 1.f, 1.f, 1.f);
	MATERIAL* red = new MATERIAL;
	red->diffuseColor = glm::vec4(0.5f, 0.f, 0.f, 1.f);
	MATERIAL* green = new MATERIAL;
	green->diffuseColor = glm::vec4(0.f, 0.5f, 0.f, 1.f);
	MATERIAL* gray = new MATERIAL;
	gray->diffuseColor = glm::vec4(0.5f, 0.5f, 0.5f, 1.f);

	ClearScene();

	float sum_area = 0.f;
	float area;
	glm::vec3 refl = glm::vec3(0.f);

	glm::mat4 translate = IdentityMatrix();
	glm::mat4 rotate = IdentityMatrix();
	glm::mat4 scale = IdentityMatrix();
	glm::mat4 normalize = glm::translate(0.5f, 0.5f, 0.5f) * glm::scale(0.5f, 0.5f, 0.5f);

	translate = glm::translate(glm::vec3(3.68f, 0.825f, -1.69f));
	rotate = glm::rotate(Rad2Deg(-0.29f), glm::vec3(0.0f, 1.0f, 0.0f));
	scale = glm::scale(glm::vec3(0.825f, 0.825f, 0.825f));
	CModel* smallBox = new CModel();
	smallBox->Init(new CCubeMesh());
	smallBox->SetWorldTransform(translate * rotate * scale);
	smallBox->SetMaterial(*gray);

	area = 0.825f * 6 * 2;
	sum_area += area;
	refl += area * glm::vec3(0.5f, 0.5f, 0.5f);
	
	translate = glm::translate(glm::vec3(1.85f, 1.65f, -3.51f));
	rotate = glm::rotate(Rad2Deg(-1.27f), glm::vec3(0.0f, 1.0f, 0.0f));
	scale = glm::scale(glm::vec3(0.825f, 1.65f, 0.825f));
	CModel* largeBox = new CModel();
	largeBox->Init(new CCubeMesh());
	largeBox->SetWorldTransform(translate * rotate * scale);
	largeBox->SetMaterial(*gray);
	
	area = 0.825f * 4 * 2 + 1.65f * 4 * 1;
	sum_area += area;
	refl += area * glm::vec3(0.5f, 0.5f, 0.5f);
	
	translate = glm::translate(glm::vec3(2.8f, 0.0f, -2.8f));
	rotate = glm::rotate(-90.0f, glm::vec3(1.0f, 0.0f, 0.0f));
	scale = glm::scale(glm::vec3(2.8f, 1.0f, 2.8f));
	CModel* floor = new CModel();
	floor->Init(new CQuadMesh());
	floor->SetWorldTransform(translate * scale * rotate);
	floor->SetMaterial(*white);
	area = 2.8f * 4.f;
	sum_area += area;
	refl += area * glm::vec3(1.0f, 1.0f, 1.0f);
	
	translate = glm::translate(glm::vec3(2.8f, 2.75f, -5.6f));
	scale = glm::scale(glm::vec3(2.8f, 2.75f, 1.0f));
	CModel* back = new CModel();
	back->Init(new CQuadMesh());
	back->SetWorldTransform(translate * scale);
	back->SetMaterial(*white);
	area = 2.8f * 2.f * 2.75f * 2.f;
	sum_area += area;
	refl += area * glm::vec3(1.0f, 1.0f, 1.0f);

	translate = glm::translate(glm::vec3(0.0f, 2.75f, -2.8f));
	rotate = glm::rotate(90.0f, glm::vec3(0.0f, 1.0f, 0.0f));
	scale = glm::scale(glm::vec3(1.0f, 2.75f, 2.8f));
	CModel* left = new CModel();
	left->Init(new CQuadMesh());
	left->SetWorldTransform(translate * scale * rotate);
	left->SetMaterial(*red);
	area = 2.8f * 2.f * 2.75f * 2.f;
	sum_area += area;
	refl += area * glm::vec3(0.5f, 0.0f, 0.0f);

	translate = glm::translate(glm::vec3(5.6f, 2.75f, -2.8f));
	rotate = glm::rotate(-90.0f, glm::vec3(0.0f, 1.0f, 0.0f));
	scale = glm::scale(glm::vec3(1.0f, 2.75f, 2.8f));
	CModel* right = new CModel();
	right->Init(new CQuadMesh());
	right->SetWorldTransform(translate * scale * rotate);
	right->SetMaterial(*green);
	area = 2.8f * 2.f * 2.75f * 2.f;
	sum_area += area;
	refl += area * glm::vec3(0.0f, 0.5f, 0.0f);

	translate = glm::translate(glm::vec3(2.8f, 5.5f, -2.8f));
	rotate = glm::rotate(90.0f, glm::vec3(1.0f, 0.0f, 0.0f));
	scale = glm::scale(glm::vec3(2.8f, 1.0f, 2.8f));
	CModel* ciel = new CModel();
	ciel->Init(new CQuadMesh());
	ciel->SetWorldTransform(translate * scale * rotate);
	ciel->SetMaterial(*white);
	area = 2.8f * 4.f;
	sum_area += area;
	refl += area * glm::vec3(1.0f, 1.0f, 1.0f);

	m_AvgReflectivity = refl * 1.0f/sum_area;
	m_MeanRho = 1.f/3.f * (m_AvgReflectivity.r + m_AvgReflectivity.g + m_AvgReflectivity.b);

	std::cout << "Average reflectivity: (" << m_AvgReflectivity.r << ", " << m_AvgReflectivity.g << ", " << m_AvgReflectivity.b << ")" << std::endl;

	m_Models.push_back(smallBox);
	m_Models.push_back(largeBox);
	m_Models.push_back(floor);
	m_Models.push_back(back);
	m_Models.push_back(left);
	m_Models.push_back(right);
	m_Models.push_back(ciel);

	m_Camera->Init(glm::vec3(2.78f, 2.73f, 8.0f), 
		glm::vec3(2.8f, 2.73f, -2.8f),
		2.0f);
	
	
	m_AreaLight = new AreaLight(1.3f, 1.05f, 
		glm::vec3(2.8f, 5.5f, -2.8f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(150.0f, 150.0f, 150.0f));
	
	/*
	m_AreaLight = new AreaLight(1.3f, 1.05f, 
		glm::vec3(5.6f, 2.75f, -2.8f), 
		glm::vec3(-1.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f), 
		glm::vec3(1.0f, 1.0f, 1.0f));
	*/
	
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