#ifndef _C_CAMERA_H_
#define _C_CAMERA_H_

typedef unsigned int uint;

#include "glm/glm.hpp"

struct SphereCoord 
{
		float r;
		float phi;
		float theta;
};

class CCamera 
{
public:
	CCamera(int width, int height, float zNear, float zFar);

	glm::mat4 GetProjectionMatrix();
	glm::mat4 GetViewMatrix();
	glm::vec3 GetPosition();

	void Init(glm::vec3 position, glm::vec3 center, float speed);

	void RotLeft(float t);
	void RotRight(float t);
	void RotUp(float t);
	void RotDown(float t);
	void ZoomOut(float t);
	void ZoomIn(float t);
	void SetSpeed(float s);

	float GetAspectRatio() { return (float)m_Width/(float)m_Height; }
	uint GetHeight() { return m_Height; }
	uint GetWidth() { return m_Width; }
	
	void SetHeight(uint height) { m_Height = height; }
	void SetWidth(uint width) { m_Width = width; }

	void PrintConfig();

private:
	uint m_Width, m_Height;
	SphereCoord m_SphericalCoord;
	float m_Speed;
	
	glm::vec3 m_Center;
	glm::vec3 m_Position;
	glm::mat4 m_ProjectionMatrix;
	glm::mat4 m_ViewMatrix;
};

#endif // _C_CAMERA_H_