#include "Light.h"

#include <glm/gtx/transform.hpp>

#include "CUtils\Util.h"

Light::Light(glm::vec3 position, glm::vec3 orientation, glm::vec3 flux,
	glm::vec3 src_position, glm::vec3 src_orientation, glm::vec3 src_flux)
{
	m_Position = position;
	m_Orientation = orientation;
	m_Flux = flux;
	m_SrcPosition = src_position;
	m_SrcOrientation = src_orientation;
	m_SrcFlux = src_flux;
	
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
	light.Position = glm::vec4(m_Position, 1.f);
	light.Orientation = glm::vec4(m_Orientation, 1.f);
	light.Flux = glm::vec4(m_Flux, 1.f);
	light.SrcPosition = glm::vec4(m_SrcPosition, 1.f);
	light.SrcOrientation = glm::vec4(m_SrcOrientation, 1.f);
	light.SrcFlux = glm::vec4(m_SrcFlux, 1.f);
	light.DebugColor = glm::vec4(m_DebugColor, 1.f);
	light.ViewMatrix = m_ViewMatrix;
	light.ProjectionMatrix = m_ProjectionMatrix;
}