#ifndef _C_CAMERA_H_
#define _C_CAMERA_H_

typedef unsigned int uint;

#include "glm/glm.hpp"

#include "Ray.h"

#include <vector>

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

	glm::mat4 GetProjectionMatrix() const;
	glm::mat4 GetViewMatrix() const;
	glm::vec3 GetPosition() const;

	void Init(glm::vec3 position, glm::vec3 center, float speed);

	void RotLeft(float t);
	void RotRight(float t);
	void RotUp(float t);
	void RotDown(float t);
	void ZoomOut(float t);
	void ZoomIn(float t);
	void SetSpeed(float s);
	void MoveForward(float t);
	void MoveBackward(float t);

	float GetAspectRatio() { return (float)m_Width/(float)m_Height; }
	uint GetHeight() { return m_Height; }
	uint GetWidth() { return m_Width; }
	
	void SetHeight(uint height) { m_Height = height; }
	void SetWidth(uint width) { m_Width = width; }

	Ray GetEyeRay(uint p_x, uint p_y);
	void GetEyeRays(std::vector<Ray>& rays, std::vector<glm::vec2>& samples, uint numRays);
	
	void UpdateData();

	void PrintConfig();

private:
	uint m_Width, m_Height;
	SphereCoord m_SphericalCoord;
	float m_Speed;

	float m_FOV;
	glm::vec3 m_U, m_V, m_W, m_Up;
	float m_Scale;
	glm::vec3 m_IO;

	glm::vec3 m_Center;
	glm::vec3 m_Position;
	glm::mat4 m_ProjectionMatrix;
	glm::mat4 m_ViewMatrix;
};

#endif // _C_CAMERA_H_