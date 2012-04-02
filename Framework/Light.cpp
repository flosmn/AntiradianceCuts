#include "Light.h"

#include <glm/gtx/transform.hpp>

#include "Util.h"

Light::Light(glm::vec3 position, glm::vec3 orientation, glm::vec3 surfAlbedo,
	glm::vec3 flux, glm::vec3 antiflux, glm::vec3 antiPosition, glm::vec3 incLightDir)
{
	m_Position = position;
	m_Orientation = orientation;
	m_SurfaceAlbedo = surfAlbedo;
	m_Flux = flux;
	m_Antiflux = antiflux;
	m_AntiPosition = antiPosition;
	m_IncLightDir = incLightDir;
	
	m_DebugColor = glm::vec3(1.0f, 0.7f, 0.0f);

	m_ProjectionMatrix = glm::perspective(90.0f, 1.0f, 0.1f, 100.0f);

	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	if(glm::abs(glm::dot(up, orientation)) > 0.009) {
		up = glm::vec3(0.0f, 0.0f, 1.0f); 
	}

	m_ViewMatrix = glm::lookAt(position, position + orientation, up);
}
 
Light::~Light() 
{
}

void Light::Fill(LIGHT& light)
{
	light.m_Antiflux = glm::vec4(m_Antiflux, 1.f);
	light.m_AntiPosition = glm::vec4(m_AntiPosition, 1.f);
	light.m_DebugColor = glm::vec4(m_DebugColor, 1.f);
	light.m_IncLightDir = glm::vec4(m_IncLightDir, 0.f);
	light.m_Orientation = glm::vec4(m_Orientation, 0.f);
	light.m_Position = glm::vec4(m_Position, 1.f);
	light.m_ProjectionMatrix = m_ProjectionMatrix;
	light.m_Flux = glm::vec4(m_Flux, 1.f);
	light.m_SurfaceAlbedo = glm::vec4(m_SurfaceAlbedo, 1.f);
	light.m_ViewMatrix = m_ViewMatrix;
}