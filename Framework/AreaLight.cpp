#include "AreaLight.h"

#include "Macros.h"

#include "CCamera.h"

#include "Utils\ShaderUtil.h"
#include "Utils\Util.h"

#include "Triangle.h"
#include "CMaterialBuffer.h"

#include "MeshResources\CMesh.h"
#include "MeshResources\CModel.h"
#include "MeshResources\CSubModel.h"

#include "OGLResources\COGLUniformBuffer.h"

#define NUM_QR_NUMBERS 1024

AreaLight::AreaLight(float _width, float _height, glm::vec3 _centerPosition, 
	glm::vec3 _frontDirection, glm::vec3 _upDirection, CMaterialBuffer* pMaterialBuffer)
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

	m_areaLightModel.reset(new CModel(new CQuadMesh()));
	UpdateWorldTransform();

	m_pMaterialBuffer = pMaterialBuffer;
	m_MaterialIndex = 0;
}

AreaLight::~AreaLight()
{
}

void AreaLight::Update()
{
	if(m_MaterialIndex == 0)
	{
		MATERIAL m;
		m.emissive = glm::vec4(radiance, 1.f);
		m_MaterialIndex = m_pMaterialBuffer->AddMaterial("LightSource", m);
	}
	else
	{
		m_pMaterialBuffer->GetMaterial(m_MaterialIndex)->emissive =  glm::vec4(radiance, 1.f);
	}
	UpdateWorldTransform();
}

void AreaLight::Draw(CCamera* camera, COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight)
{
	AREA_LIGHT arealight;
	arealight.radiance = glm::vec4(GetRadiance(), 1.f);
	pUBAreaLight->UpdateData(&arealight);

	m_areaLightModel->Draw(camera->GetViewMatrix(), camera->GetProjectionMatrix(), pUBTransform);
}

void AreaLight::Draw(CCamera* camera, COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight, glm::vec3 color)
{
	AREA_LIGHT arealight;
	arealight.radiance = glm::vec4(color, 1.f);
	pUBAreaLight->UpdateData(&arealight);

	m_areaLightModel->Draw(camera->GetViewMatrix(), camera->GetProjectionMatrix(), pUBTransform);
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
	glm::vec2 u = glm::linearRand(glm::vec2(-1, -1), glm::vec2(1, 1));
	glm::vec3 direction = GetRandomSampleDirectionCosCone(
		GetFrontDirection(), u.x, u.y, pdf, order);

	return direction;
}

glm::mat4 AreaLight::GetWorldTransform()
{ 
	return m_areaLightModel->GetWorldTransform(); 
}

void AreaLight::SetFlux(glm::vec3 _flux)
{
	flux = flux;

	intensity = flux / M_PI;
	radiance = intensity / area;
	Update();
}

void AreaLight::SetIntensity(glm::vec3 _intensity)
{
	intensity = _intensity;

	flux = intensity * M_PI;
	radiance = intensity / area;
	Update();
}

void AreaLight::SetRadiance(glm::vec3 _radiance)
{
	radiance = _radiance;

	intensity = radiance * area;
	flux = intensity * M_PI;	
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
	glm::mat4 scale = glm::scale(glm::vec3(width/2.f, height/2.f, 1.0f));
	
	glm::mat4 position = glm::lookAt(centerPosition + 1.f * glm::normalize(frontDirection), centerPosition - frontDirection, upDirection);

	position = glm::inverse(position);
	m_areaLightModel->SetWorldTransform(position * scale);
}

 void AreaLight::GetTrianglesWS(std::vector<Triangle>& triangles)
{
	std::vector<CSubModel*> subModels = m_areaLightModel->GetSubModels();
	std::vector<CSubModel*>::iterator it_subModels;
	for (it_subModels = subModels.begin(); it_subModels < subModels.end(); it_subModels++ )
	{
		CSubModel* subModel = *it_subModels;
		std::vector<Triangle>& t = subModel->GetTrianglesWS();
		for(uint i = 0; i < t.size(); ++i)
		{
			t[i].setMaterialIndex(m_MaterialIndex);
			triangles.push_back(t[i]);
		}
	}
}
