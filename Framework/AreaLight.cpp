#include "AreaLight.h"

#include "Macros.h"

#include "CCamera.h"

#include "Utils\ShaderUtil.h"
#include "Utils\Util.h"

#include "CTriangle.h"

#include "MeshResources\CMesh.h"
#include "MeshResources\CModel.h"
#include "MeshResources\CSubModel.h"

#include "OGLResources\COGLUniformBuffer.h"

#define NUM_QR_NUMBERS 1024

AreaLight::AreaLight(float _width, float _height, glm::vec3 _centerPosition, 
	glm::vec3 _frontDirection, glm::vec3 _upDirection)
{
	width = _width;
	height = _height;
	area = width * height;
	centerPosition = _centerPosition;
	frontDirection = _frontDirection;
	upDirection = _upDirection;
	intensity = glm::vec3(0.f);
	radiance = glm::vec3(0.f);
	flux = glm::vec3(0.f);

	m_PlaneHammersleyIndex = 0;
	m_pPlaneHammersleyNumbers = new float[2 * NUM_QR_NUMBERS];
	memset(m_pPlaneHammersleyNumbers, 0, 2 * NUM_QR_NUMBERS * sizeof(float));
	PlaneHammersley(m_pPlaneHammersleyNumbers, NUM_QR_NUMBERS);

	m_pAreaLightModel = new CModel();
}

AreaLight::~AreaLight()
{
	SAFE_DELETE(m_pAreaLightModel);
	SAFE_DELETE_ARRAY(m_pPlaneHammersleyNumbers);
}

bool AreaLight::Init()
{
	V_RET_FOF(m_pAreaLightModel->Init(new CQuadMesh()));
	
	UpdateWorldTransform();

	return true;
}

void AreaLight::Release()
{
	m_pAreaLightModel->Release();
}

void AreaLight::Update()
{
	MATERIAL m;
	m.emissive = glm::vec4(radiance, 1.f);
	m_pAreaLightModel->SetMaterial(m);

	UpdateWorldTransform();
}

void AreaLight::Draw(CCamera* camera, COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight)
{
	AREA_LIGHT arealight;
	arealight.radiance = glm::vec4(GetRadiance(), 1.f);
	pUBAreaLight->UpdateData(&arealight);

	m_pAreaLightModel->Draw(camera->GetViewMatrix(), camera->GetProjectionMatrix(), pUBTransform);
}

glm::vec3 AreaLight::SamplePos(float& pdf)
{
	glm::mat4 transform = GetWorldTransform();
	
	glm::vec2 samplePos = glm::linearRand(glm::vec2(-1, -1), glm::vec2(1, 1));
	glm::vec4 positionTemp = transform * glm::vec4(samplePos.x, samplePos.y, 0.0f, 1.0f);		
	glm::vec3 position	= glm::vec3(positionTemp /= positionTemp.w);		
	
	pdf = 1.f/area;
	return position;
}

glm::vec3 AreaLight::SampleDir(float& pdf, int order)
{
	float u1 = m_pPlaneHammersleyNumbers[2 * m_PlaneHammersleyIndex + 0];
	float u2 = m_pPlaneHammersleyNumbers[2 * m_PlaneHammersleyIndex + 1];

	glm::vec3 direction = GetRandomSampleDirectionCosCone(GetFrontDirection(), u1, u2, pdf, order);
	
	m_PlaneHammersleyIndex++;
	if(m_PlaneHammersleyIndex >= NUM_QR_NUMBERS) m_PlaneHammersleyIndex = 0;

	return direction;
}

glm::mat4 AreaLight::GetWorldTransform()
{ 
	return m_pAreaLightModel->GetWorldTransform(); 
}

void AreaLight::SetFlux(glm::vec3 _flux)
{
	flux = flux;

	intensity = flux / PI;
	radiance = intensity / area;
	Update();
}

void AreaLight::SetIntensity(glm::vec3 _intensity)
{
	intensity = _intensity;

	flux = intensity * PI;
	radiance = intensity / area;
	Update();
}

void AreaLight::SetRadiance(glm::vec3 _radiance)
{
	radiance = _radiance;

	intensity = radiance * area;
	flux = intensity * PI;	
	Update();
}

void AreaLight::SetCenterPosition(glm::vec3 pos)
{
	centerPosition = pos;
	UpdateWorldTransform();
}

void AreaLight::SetFrontDirection(glm::vec3 dir)
{
	frontDirection = dir;
	UpdateWorldTransform();
}

void AreaLight::UpdateWorldTransform()
{
	glm::mat4 scale = glm::scale(width/2.f, height/2.f, 1.0f);
	
	glm::mat4 position = glm::lookAt(centerPosition + 1.f * glm::normalize(frontDirection), centerPosition - frontDirection, upDirection);

	position = glm::inverse(position);
	m_pAreaLightModel->SetWorldTransform(position * scale);
}

 void AreaLight::GetTrianglesWS(std::vector<CTriangle*>& triangles)
{
	std::vector<CSubModel*> subModels = m_pAreaLightModel->GetSubModels();
	std::vector<CSubModel*>::iterator it_subModels;
	for (it_subModels = subModels.begin(); it_subModels < subModels.end(); it_subModels++ )
	{
		CSubModel* subModel = *it_subModels;
		std::vector<CTriangle*> t = subModel->GetTrianglesWS();
		for(uint i = 0; i < t.size(); ++i)
		{
			triangles.push_back(t[i]);
		}
	}
}