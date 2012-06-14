#ifndef _AVPL__H
#define _AVPL__H

#include "glm/glm.hpp"

#include "Structs.h"

class AVPL
{
public:
	AVPL();
	AVPL(glm::vec3 p, glm::vec3 n, glm::vec3 I, glm::vec3 A, glm::vec3 w_A, int bounce);
	~AVPL();

	glm::mat4 GetViewMatrix() { return m_ViewMatrix; }
	glm::mat4 GetProjectionMatrix() { return m_ProjectionMatrix; }

	glm::vec3 GetPosition() { return m_Position; }
	glm::vec3 GetOrientation() { return m_Orientation; }
	glm::vec3 GetIntensity(glm::vec3 w);
	glm::vec3 GetAntiintensity(glm::vec3 w, const float& N);
	glm::vec3 GetAntiintensity() { return m_Antiintensity; }
	glm::vec3 GetAntiradianceDirection() { return m_AntiradianceDirection; }	
	
	void SetIntensity(glm::vec3 i) { m_Intensity = i; }
	void SetAntiintensity(glm::vec3 a) { m_Antiintensity = a; }
	
	glm::vec3 SampleAntiradianceDirection(const float& N);

	int GetBounce() { return m_Bounce; } 
	
	void Fill(AVPL_STRUCT& avpl);
	void Fill(AVPL_BUFFER& avpl_buffer);

private:
	glm::vec3 m_Position;
	glm::vec3 m_Orientation;
	glm::vec3 m_Intensity;
	glm::vec3 m_Antiintensity;
	glm::vec3 m_AntiradianceDirection;
	glm::vec3 m_DebugColor;
	glm::mat4 m_ViewMatrix;
	glm::mat4 m_ProjectionMatrix;
	int m_Bounce;
};

#endif