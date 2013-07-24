#include "Scene.h"

typedef unsigned int uint;

#include "Defines.h"
#include "Structs.h"

#include "Avpl.h"
#include "AreaLight.h"
#include "CCamera.h"
#include "KdTreeAccelerator.h"
#include "Triangle.h"
#include "CConfigManager.h"
#include "CTimer.h"
#include "CMaterialBuffer.h"
#include "CReferenceImage.h"

#include "Utils/stream.h"
#include "Utils/Util.h"

#include "OGLResources\COGLUniformBuffer.h"

#include "mesh.hpp"
#include "model.hpp"

#include <set>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <sstream>

#include <omp.h>

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/importer.hpp>


Scene::Scene(CCamera* camera, CConfigManager* confManager)
	: m_camera(camera), m_confManager(confManager)
{
	m_materialBuffer.reset(new CMaterialBuffer());

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
	m_meshes.clear();
	
	ClearLighting();
}

 void Scene::ClearLighting()
{			
	m_CurrentBounce = 0;
	m_NumLightPaths = 0;

	m_NumCreatedAVPLs = 0;
}
 
bool Scene::CreateAVPL(Avpl* pred, Avpl* newAVPL)
{
	float pdf = 0.f;
	glm::vec3 direction = SamplePhong(pred->getIncidentDirection(), pred->getNormal(), 
		m_materialBuffer->GetMaterial(pred->getMaterialIndex()), pdf);
	
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

	Avpl avpl;
	if(ContinueAVPLPath(pred, &avpl, direction, pdf))
	{
		*newAVPL = avpl;
		return true;
	}

	newAVPL = 0;
	return false;
}

bool Scene::CreatePrimaryAVPL(Avpl* newAVPL)
{
	if(!m_areaLight)
	{
		newAVPL = 0;
		return false;
	}
	
	// create VPL on light source
	float pdf;
	glm::vec3 pos = m_areaLight->samplePos(pdf);
	glm::vec3 normal = m_areaLight->getFrontDirection();
	glm::vec3 I = m_areaLight->getRadiance();

	if(!(pdf > 0.f))
		std::cout << "Warning: pdf is 0" << std::endl;

	Avpl avpl(pos, normal, I / pdf, glm::vec3(0), glm::vec3(0), 0, m_areaLight->getMaterialIndex());
	*newAVPL = avpl;

	return true;
}

bool Scene::ContinueAVPLPath(Avpl* pred, Avpl* newAVPL, glm::vec3 direction, float pdf)
{
	glm::vec3 pred_pos = pred->getPosition();

	const glm::vec3 pred_norm = glm::normalize(pred->getNormal());
	direction = glm::normalize(direction);

	Ray ray(pred_pos + EPS * direction, direction);
		
	Intersection intersection;
	float t = 0.f;
	if(IntersectRayScene(ray, &t, &intersection, Triangle::FRONT_FACE))
	{
		// gather information for the new VPL
		glm::vec3 norm = intersection.getTriangle()->getNormal();
		glm::vec3 pos = intersection.getPosition() - EPS * norm;
		uint index = intersection.getTriangle()->getMaterialIndex();
		if(index > 100)
			std::cout << "material index uob" << std::endl;
		
		const float cos_theta = glm::dot(pred_norm, direction);
		glm::vec3 contrib = pred->getIncidentRadiance() * cos_theta / pdf;

		if (pred->getBounce() > 0) {
			contrib *= glm::vec3(Phong(-pred->getIncidentDirection(), direction, 
				pred->getNormal(), m_materialBuffer->GetMaterial(pred->getMaterialIndex())));
		}
		
		glm::vec3 antiradiance = 1.f/float(m_confManager->GetConfVars()->NumAdditionalAVPLs + 1.f) * contrib;

		Avpl avpl(pos, norm, contrib, antiradiance, direction, pred->getBounce() + 1, intersection.getTriangle()->getMaterialIndex());
		*newAVPL = avpl;
		return true;
	}

	newAVPL = 0;
	return false;
}

void Scene::CreatePaths(std::vector<Avpl>& avpls_res, std::vector<Avpl>& allAVPLs, std::vector<Avpl>& isAVPLs, bool profile, uint numPaths)
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

void Scene::CreatePath(std::vector<Avpl>& path) 
{
	m_NumLightPaths++;

	int currentBounce = 0;
	
	Avpl pred, succ;
		
	// create new primary light on light source
	CreatePrimaryAVPL(&pred);
	
	path.push_back(pred);
	
	currentBounce++;

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
			CreateAVPLs(&pred, path, m_confManager->GetConfVars()->NumAdditionalAVPLs);

			Avpl avpl;
			if(CreateAVPL(&pred, &avpl))
			{
				avpl.setIncidentRadiance(glm::vec3(0.f));
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
				succ.setIncidentRadiance(1.f / rrProb * succ.getIncidentRadiance());
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

void Scene::CreateAVPLs(Avpl* pred, std::vector<Avpl>& path, int nAVPLs)
{
	glm::vec3 pred_norm = pred->getNormal();

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

		Avpl avpl;
		if(ContinueAVPLPath(pred, &avpl, direction, pdf))		
		{
			avpl.setIncidentRadiance(glm::vec3(0.f));
			path.push_back(avpl);
		}
	}
}

void Scene::CreatePrimaryVpls(std::vector<Avpl>& avpls, int numVpls)
{
	for(int i = 0; i < numVpls; ++i)
	{
		Avpl avpl;
		if(CreatePrimaryAVPL(&avpl))
			avpls.push_back(avpl);
	}
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

	loadSceneFromFile("twoplanes.obj");

	m_camera->Init(0, glm::vec3(-9.2f, 5.7f, 6.75f), 
		glm::vec3(0.f, -2.f, 0.f),
		glm::vec3(0.f, 1.f, 0.f),
		2.0f);
	
	m_areaLight.reset(new AreaLight(0.5f, 0.5f, 
		glm::vec3(0.0f, 4.f, 0.0f), 
		glm::vec3(0.0f, -1.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f),
		glm::vec3(120.f, 120.f, 120.f),
		m_materialBuffer.get()));

	initKdTree();
	calcBoundingBox();
}

void Scene::LoadCornellBox()
{
	ClearScene();

	loadSceneFromFile("cb-diffuse.obj");
	
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
		glm::vec3(100.f, 100.f, 100.f),
		m_materialBuffer.get()));
	
	initKdTree();
	calcBoundingBox();
}

void Scene::LoadCornellEmpty()
{
	ClearScene();

	loadSceneFromFile("cb_empty.obj");

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
		AreaLightRadianceScale * glm::vec3(100.f, 100.f, 100.f),
		m_materialBuffer.get()));
	
	initKdTree();
	calcBoundingBox();
}

void Scene::LoadBuddha()
{
	ClearScene();

	loadSceneFromFile("cb-buddha-specular.obj");
	
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
		glm::vec3(100.f, 100.f, 100.f),
		m_materialBuffer.get()));
	
	initKdTree();
	calcBoundingBox();
}

void Scene::LoadConferenceRoom()
{
	ClearScene();

	loadSceneFromFile("conference-3.obj");
	
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
		AreaLightRadianceScale * glm::vec3(1.f, 1.f, 1.f),
		m_materialBuffer.get()));
	
	initKdTree();
	calcBoundingBox();
}

void Scene::LoadSibernik()
{
	ClearScene();

	float scale = 100.f;

	loadSceneFromFile("sibenik.obj");

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

	float AreaLightRadianceScale = 3000.f;
	m_confManager->GetConfVars()->AreaLightRadianceScale = m_confManager->GetConfVarsGUI()->AreaLightRadianceScale = AreaLightRadianceScale;
	
	m_areaLight.reset(new AreaLight(scale * 0.25f, scale * 0.25f, 
		areaLightPosition, 
		areaLightFrontDir,
		Orthogonal(areaLightFrontDir),
		glm::vec3(AreaLightRadianceScale),
		m_materialBuffer.get()));
	
	initKdTree();
	calcBoundingBox();
}

void Scene::initKdTree()
{
	std::vector<Triangle> triangles;

	for (size_t i = 0; i < m_models.size(); ++i) {
		Mesh* mesh = m_models[i]->getMesh();
		glm::mat4 worldTransform = m_models[i]->getWorldTransform();
		int materialIndex = m_models[i]->getMaterial();

		std::vector<glm::vec3> const& positions = mesh->getPositions();
		std::vector<glm::uvec3> const& indices = mesh->getIndices();
		for (size_t j = 0; j < indices.size(); ++j) {
			glm::vec4 p0 =  glm::vec4(positions[indices[j].x], 1.f);
			glm::vec4 p1 =  glm::vec4(positions[indices[j].y], 1.f);
			glm::vec4 p2 =  glm::vec4(positions[indices[j].z], 1.f);
			Triangle triangle(glm::vec3(worldTransform*p0), 
							  glm::vec3(worldTransform*p1), 
							  glm::vec3(worldTransform*p2));
			triangle.setMaterialIndex(materialIndex);

			triangles.push_back(triangle);
		}
	}
	
	{ // add triangles from the area light source
		Mesh const* mesh = m_areaLight->getMesh();
		glm::mat4 worldTransform = m_areaLight->getWorldTransform();
		int materialIndex = m_areaLight->getMaterialIndex();

		std::vector<glm::vec3> const& positions = mesh->getPositions();
		std::vector<glm::uvec3> const& indices = mesh->getIndices();
		for (size_t j = 0; j < indices.size(); ++j) {
			glm::vec4 p0 =  glm::vec4(positions[indices[j].x], 1.f);
			glm::vec4 p1 =  glm::vec4(positions[indices[j].y], 1.f);
			glm::vec4 p2 =  glm::vec4(positions[indices[j].z], 1.f);
			Triangle triangle(glm::vec3(worldTransform*p0), 
							  glm::vec3(worldTransform*p1), 
							  glm::vec3(worldTransform*p2));
			triangle.setMaterialIndex(materialIndex);

			triangles.push_back(triangle);
		}
	}

	m_kdTreeAccelerator.reset(new KdTreeAccelerator(triangles, 80, 1, 5, 0));
	CTimer timer(CTimer::CPU);
	timer.Start();
	m_kdTreeAccelerator->buildTree();
	timer.Stop("BuildAccelerationTree");
}

void Scene::ReleaseKdTree()
{
	m_kdTreeAccelerator.reset(nullptr);
}

void Scene::UpdateAreaLights()
{
	if(!m_areaLight)
		return; 
	
	glm::vec3 pos = glm::vec3(
		m_confManager->GetConfVars()->AreaLightPosX, 
		m_confManager->GetConfVars()->AreaLightPosY, 
		m_confManager->GetConfVars()->AreaLightPosZ); 
	m_areaLight->setCenterPosition(pos);

	glm::vec3 front = glm::vec3(
		m_confManager->GetConfVars()->AreaLightFrontDirection[0],
		m_confManager->GetConfVars()->AreaLightFrontDirection[1],
		m_confManager->GetConfVars()->AreaLightFrontDirection[2]);
	m_areaLight->setFrontDirection(front);

	m_areaLight->setRadiance(glm::vec3(m_confManager->GetConfVars()->AreaLightRadianceScale));
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
	bool isect = IntersectRayScene(r, &t, &intersection,  Triangle::FRONT_FACE);
			
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
	ss.position = m_areaLight->samplePos(pdf);
	ss.normal = m_areaLight->getFrontDirection();
	ss.pdf = pdf;
	ss.materialIndex = m_areaLight->getMaterialIndex();

	if(!(pdf > 0.f))
		std::cout << "Warning: pdf is 0" << std::endl;
}

MATERIAL* Scene::GetMaterial(const SceneSample& ss)
{
	return m_materialBuffer->GetMaterial(ss.materialIndex);
}

void Scene::calcBoundingBox() 
{
	glm::vec3 min(std::numeric_limits<float>::max());
	glm::vec3 max(std::numeric_limits<float>::min());

	for (size_t i = 0; i < m_models.size(); ++i) {
		Mesh* mesh = m_models[i]->getMesh();
		glm::mat4 worldTransform = m_models[i]->getWorldTransform();
		
		std::vector<glm::vec3> const& positions = mesh->getPositions();
		for (size_t j = 0; j < positions.size(); ++j) {
			glm::vec3 pos =  glm::vec3(worldTransform * glm::vec4(positions[j], 1.f));
			min = glm::min(min, pos);
			max = glm::max(max, pos);
		}
	}
	m_bbox = BBox(min, max);
	std::cout << "scene bounding box: min=" << m_bbox.getMin() << ", max=" << m_bbox.getMax() << std::endl;
}

void Scene::loadSceneFromFile(std::string const& file) 
{
	std::cout << "Start loading model from file " << file << std::endl;
	CTimer timer(CTimer::CPU);
	timer.Start();
	
	// Create an instance of the Importer class
	Assimp::Importer importer;
	const aiScene* scene = NULL;
	
	//check if file exists
	std::stringstream ss;
	ss << "Resources\\" << file;

	std::ifstream fin(ss.str().c_str());
    if (!fin.fail()) {
        fin.close();
    } else {
		printf("Couldn't open file: %s\n", ss.str().c_str());
        printf("%s\n", importer.GetErrorString());
        return;
    }
 
	std::string pFile(ss.str());
	scene = importer.ReadFile(pFile, aiProcess_Triangulate | aiProcess_GenNormals  | aiProcess_FlipWindingOrder);
 
    // If the import failed, report it
    if (!scene) {
        printf("%s\n", importer.GetErrorString());
        return;
    }
 
    // Now we can access the file's contents.
    timer.Stop("Import with assimp");
	
	timer.Start();
    // For each mesh
    for (unsigned int n = 0; n < scene->mNumMeshes; ++n)
    {
        const struct aiMesh* mesh = scene->mMeshes[n];
 
		 // create material uniform buffer
        struct aiMaterial *mtl = scene->mMaterials[mesh->mMaterialIndex];
        
		aiColor4D d;
		aiString matName;
		float e = 0.f;

		std::string name = "noname";
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_NAME, matName))
			name = std::string(matName.C_Str());

        glm::vec4 diffuse = glm::vec4(0.f);
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_COLOR_DIFFUSE, d))
            diffuse = glm::vec4(d.r, d.g, d.b, d.a);

		glm::vec4 specular = glm::vec4(0.f);
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_COLOR_SPECULAR, d))
			specular = glm::vec4(d.r, d.g, d.b, d.a);

		float exponent = 0.f;
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_SHININESS, e))
			exponent = e;
		
		glm::vec4 emissive = glm::vec4(0.f);
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_COLOR_EMISSIVE, d))
            emissive = glm::vec4(d.r, d.g, d.b, d.a);
        
		MATERIAL mat;
		mat.emissive = emissive;
		mat.diffuse = diffuse;
		mat.specular = specular;
		mat.exponent = exponent;

		int materialIndex = m_materialBuffer->AddMaterial(name, mat);

        // create array with faces
        // have to convert from Assimp format to array
		std::vector<glm::uvec3> indices(mesh->mNumFaces);
        for (size_t i = 0; i < mesh->mNumFaces; ++i) {
            const struct aiFace* face = &mesh->mFaces[i];
			indices[i] = glm::uvec3(face->mIndices[0], face->mIndices[1], face->mIndices[2]);
        }

        // buffer for vertex positions
		std::vector<glm::vec3> positions(mesh->mNumVertices);
        if (mesh->HasPositions()) {			
			for (size_t i = 0; i < mesh->mNumVertices; ++i) {
				positions[i] = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y,
						mesh->mVertices[i].z);
			}
        }
 		
        // buffer for vertex normals
		std::vector<glm::vec3> normals(mesh->mNumVertices);
        if (mesh->HasNormals()) {
			for (size_t i = 0; i < mesh->mNumVertices; ++i)
				normals[i] = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
        }
 	
		m_meshes.emplace_back(std::unique_ptr<Mesh>(new Mesh(positions, normals, indices, materialIndex)));
		m_models.emplace_back(std::unique_ptr<Model>(new Model(m_meshes.back().get(), materialIndex, glm::mat4())));
    }
	timer.Stop("Creating models ");
}
