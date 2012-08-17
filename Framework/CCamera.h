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

	glm::vec3 GetViewDirection();

	void Init(int config, glm::vec3 position, glm::vec3 center, glm::vec3 up, float speed);
	void UseCameraConfig(int config) { m_UseConfig = config; UpdateData(); PrintConfig(); }

	void RotLeft(float t);
	void RotRight(float t);
	void RotUp(float t);
	void RotDown(float t);
	void ZoomOut(float t);
	void ZoomIn(float t);
	void SetSpeed(float s);
	void MoveForward(float t);
	void MoveBackward(float t);
	void MoveLeft(float t);
	void MoveRight(float t);

	float GetAspectRatio() { return (float)m_Width/(float)m_Height; }
	uint GetHeight() { return m_Height; }
	uint GetWidth() { return m_Width; }
	
	void SetHeight(uint height) { m_Height = height; }
	void SetWidth(uint width) { m_Width = width; }

	Ray GetEyeRay();
	Ray GetEyeRay(float p_x, float p_y);
	void GetEyeRays(std::vector<Ray>& rays, std::vector<glm::vec2>& samples, uint numRays);
	float GetEyeRayPdf();

	float GetRho();

	void UpdateData();

	void PrintConfig();

private:
	uint m_Width, m_Height;
	float m_Speed;

	float m_FOV;
	float m_Aspect;
	glm::vec3 m_U[5], m_V[5], m_W[5], m_Up[5];
	float m_Scale;
	glm::vec3 m_IO[5];

	glm::vec3 m_Center[5];
	glm::vec3 m_Position[5];
	SphereCoord m_SphericalCoord[5];
	glm::mat4 m_ViewMatrix[5];
	glm::mat4 m_ProjectionMatrix;

	int m_UseConfig;
};

#endif // _C_CAMERA_H_