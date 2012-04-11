#ifndef LIGHT_H
#define LIGHT_H

#include "glm/glm.hpp"

#include "Structs.h"

class Light
{
public:
	Light(glm::vec3 position, glm::vec3 orientation, glm::vec3 flux, 
		  glm::vec3 src_position, glm::vec3 src_orientation, glm::vec3 src_flux);
	~Light();

	glm::mat4 GetViewMatrix() { return m_ViewMatrix; }
	glm::mat4 GetProjectionMatrix() { return m_ProjectionMatrix; }

	glm::vec3 GetPosition() { return m_Position; }
	glm::vec3 GetOrientation() { return m_Orientation; }
	glm::vec3 GetFlux() { return m_Flux; }
	glm::vec3 GetSrcPosition() { return m_SrcPosition; }
	glm::vec3 GetSrcOrientation() { return m_SrcOrientation; }
	glm::vec3 GetSrcFlux() { return m_SrcFlux; }	

	void SetDebugColor(glm::vec3 color) { m_DebugColor = color; }
	void SetFlux(glm::vec3 flux) { m_Flux = flux; }
	glm::vec3 GetDebugColor() { return m_DebugColor; }

	void Fill(LIGHT& light);

private:
	glm::vec3 m_Position;
	glm::vec3 m_Orientation;
	glm::vec3 m_Flux;
	glm::vec3 m_SrcPosition;
	glm::vec3 m_SrcOrientation;
	glm::vec3 m_SrcFlux;
	glm::vec3 m_DebugColor;
	glm::mat4 m_ViewMatrix;
	glm::mat4 m_ProjectionMatrix;
};

#endif