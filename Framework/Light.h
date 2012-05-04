#ifndef LIGHT_H
#define LIGHT_H

#include "glm/glm.hpp"

#include "Structs.h"

class Light
{
public:
	Light();
	Light(glm::vec3 position, glm::vec3 orientation, glm::vec3 contrib, 
		  glm::vec3 src_position, glm::vec3 src_orientation, glm::vec3 src_contrib, int bounce);
	~Light();

	glm::mat4 GetViewMatrix() { return m_ViewMatrix; }
	glm::mat4 GetProjectionMatrix() { return m_ProjectionMatrix; }

	glm::vec3 GetPosition() { return m_Position; }
	glm::vec3 GetOrientation() { return m_Orientation; }
	glm::vec3 GetContrib() { return m_Contrib; }
	glm::vec3 GetSrcPosition() { return m_SrcPosition; }
	glm::vec3 GetSrcOrientation() { return m_SrcOrientation; }
	glm::vec3 GetSrcContrib() { return m_SrcContrib; }	
		
	void SetContrib(glm::vec3 contrib) { m_Contrib = contrib; }
	int GetBounce() { return m_Bounce; } 
	
	void Fill(LIGHT& light);

private:
	glm::vec3 m_Position;
	glm::vec3 m_Orientation;
	glm::vec3 m_Contrib;
	glm::vec3 m_SrcPosition;
	glm::vec3 m_SrcOrientation;
	glm::vec3 m_SrcContrib;
	glm::vec3 m_DebugColor;
	glm::mat4 m_ViewMatrix;
	glm::mat4 m_ProjectionMatrix;
	int m_Bounce;
};

#endif