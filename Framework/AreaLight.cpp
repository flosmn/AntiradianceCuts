#include "AreaLight.h"

#include "Macros.h"

#include "Camera.h"
#include "Light.h"

#include "CUtils\ShaderUtil.h"

#include "CMeshResources\CMesh.h"
#include "CMeshResources\CModel.h"

#include "CGLResources\CGLUniformBuffer.h"

AreaLight::AreaLight(float _width, float _height, glm::vec3 _centerPosition, 
										 glm::vec3 _frontDirection, glm::vec3 _upDirection,
										 glm::vec3 _flux)
{
	width = _width;
	height = _height;
	area = width * height;
	centerPosition = _centerPosition;
	frontDirection = _frontDirection;
	upDirection = _upDirection;
	flux = _flux;

	m_pAreaLightModel = new CModel();
}

AreaLight::~AreaLight()
{
	SAFE_DELETE(m_pAreaLightModel);
}

bool AreaLight::Init()
{
	drawAreaLightProgram = CreateProgram("Shaders\\DrawAreaLight.vert", "Shaders\\DrawAreaLight.frag");
	
	uniformModelMatrix = glGetUniformLocation(drawAreaLightProgram, "M");
	uniformViewMatrix = glGetUniformLocation(drawAreaLightProgram, "V");
	uniformProjectionMatrix = glGetUniformLocation(drawAreaLightProgram, "P");
	uniformNormalMatrix	= glGetUniformLocation(drawAreaLightProgram, "N");
	
	uniformLightIntensity	= glGetUniformLocation(drawAreaLightProgram, "intensity");
	
	V_RET_FOF(m_pAreaLightModel->Init(new CQuadMesh()));

	MATERIAL* mat = new MATERIAL();
	mat->diffuseColor = glm::vec4(flux / (PI * area), 1.0f);
	m_pAreaLightModel->SetMaterial(*mat);

	glm::mat4 scale = glm::scale(width/2.f, height/2.f, 1.0f);
	
	glm::mat4 position = glm::lookAt(centerPosition, centerPosition - frontDirection, upDirection);

	position = glm::inverse(position);
	m_pAreaLightModel->SetWorldTransform(position * scale);

	return true;
}

void AreaLight::Release()
{
	m_pAreaLightModel->Release();
}

void AreaLight::Draw(Camera* camera, CGLUniformBuffer* pUBTransform)
{
	glUseProgram(drawAreaLightProgram);

	glUniform3fv(uniformLightIntensity, 1, glm::value_ptr(flux/PI));
	
	glUniformMatrix4fv(uniformViewMatrix, 1, GL_FALSE, glm::value_ptr(camera->GetViewMatrix()));

	glUniformMatrix4fv(uniformProjectionMatrix, 1, GL_FALSE, glm::value_ptr(camera->GetProjectionMatrix()));
	
	m_pAreaLightModel->Draw(camera->GetViewMatrix(), camera->GetProjectionMatrix(), pUBTransform);

	glUseProgram(0);

}

Light* AreaLight::GetNewPrimaryLight(float& pdf)
{
		glm::mat4 transform = GetWorldTransform();
		glm::vec3 orientation = GetFrontDirection();
		
		glm::vec2 samplePos = glm::linearRand(glm::vec2(-1, -1), glm::vec2(1, 1));
		glm::vec4 positionTemp = transform * glm::vec4(samplePos.x, samplePos.y, 0.0f, 1.0f);		
		glm::vec3 position	= glm::vec3(positionTemp /= positionTemp.w);		
		
		pdf = 1.f/area;

		Light* newLight = new Light(
			glm::vec3(position), orientation, flux/pdf, 
			glm::vec3(0), glm::vec3(0), glm::vec3(0));
		
		return newLight;
}

glm::mat4 AreaLight::GetWorldTransform()
{ 
	return m_pAreaLightModel->GetWorldTransform(); 
}