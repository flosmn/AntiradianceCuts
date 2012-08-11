#include "CCamera.h"

#include "Defines.h"

#include "Utils\Util.h"

#include <glm/gtx/transform.hpp>

#include <algorithm>
#include <iostream>

CCamera::CCamera(int w, int h, float zNear, float zFar) 
{
	m_Width = w;
	m_Height = h;
	
	m_FOV = PI / 4.f;
	m_Aspect = (float)m_Width / (float)m_Height;
	 
	m_ProjectionMatrix = glm::perspective(180.f/PI * m_FOV, m_Aspect, zNear, zFar);
	m_Speed = 2.0f;
		
	m_Scale = 2.0f / m_Height * tanf(0.5f * m_FOV);
	m_UseConfig = 0;
}

void CCamera::Init(int config, glm::vec3 position, glm::vec3 center, glm::vec3 up, float speed)
{
	m_Center[config] = center;
	m_Position[config] = position;
	m_Up[config] = up;

	m_SphericalCoord[config].r = 10.0f; 
	m_SphericalCoord[config].phi = 0.0f;
	m_SphericalCoord[config].theta = PI/2.0f;

	glm::vec3 delta = position - center;

	m_SphericalCoord[config].r = glm::length(delta);
	m_SphericalCoord[config].phi = atan2(delta.x, delta.z);
	m_SphericalCoord[config].theta = acos(delta.y / m_SphericalCoord[config].r);

	m_Speed = speed;
	UpdateData();
}

glm::mat4 CCamera::GetProjectionMatrix() const
{
	return m_ProjectionMatrix;
}

glm::mat4 CCamera::GetViewMatrix() const 
{
	return m_ViewMatrix[m_UseConfig];
}

inline glm::vec3 CCamera::GetPosition() const
{
	return m_Position[m_UseConfig];
}

void CCamera::ZoomOut(float t) 
{
	m_SphericalCoord[m_UseConfig].r += 50 *t * m_Speed;
	UpdateData();
}

void CCamera::ZoomIn(float t) 
{
	m_SphericalCoord[m_UseConfig].r -= 50 * t * m_Speed;
	m_SphericalCoord[m_UseConfig].r = std::max(m_SphericalCoord[m_UseConfig].r, 1.0f);
	UpdateData();
}

void CCamera::RotLeft(float t) {
	m_SphericalCoord[m_UseConfig].phi -= t * m_Speed;
	UpdateData();
}

void CCamera::RotRight(float t) {
	m_SphericalCoord[m_UseConfig].phi += t * m_Speed;
	UpdateData();
}

void CCamera::RotUp(float t) {
	m_SphericalCoord[m_UseConfig].theta -= t * m_Speed;
	UpdateData();
}

void CCamera::RotDown(float t) {
	m_SphericalCoord[m_UseConfig].theta += t * m_Speed;
	UpdateData();
}

void CCamera::MoveForward(float t)
{
	glm::vec3 w = glm::normalize(m_Center[m_UseConfig] - m_Position[m_UseConfig]);
	m_Center[m_UseConfig] += t * 25 * m_Speed * w;
	UpdateData();
}

void CCamera::MoveBackward(float t)
{
	glm::vec3 w = glm::normalize(m_Center[m_UseConfig] - m_Position[m_UseConfig]);
	m_Center[m_UseConfig] -= t * 25 * m_Speed * w;
	UpdateData();
}

void CCamera::MoveLeft(float t)
{
	m_Center[m_UseConfig] += t * 25 * m_Speed * glm::normalize(m_U[m_UseConfig]);
	UpdateData();
}

void CCamera::MoveRight(float t)
{
	m_Center[m_UseConfig] -= t * 25 * m_Speed * glm::normalize(m_U[m_UseConfig]);
	UpdateData();
}

void CCamera::UpdateData()
{
	glm::vec3 position = m_SphericalCoord[m_UseConfig].r * glm::vec3(
		sin(m_SphericalCoord[m_UseConfig].theta) * sin(m_SphericalCoord[m_UseConfig].phi),
		cos(m_SphericalCoord[m_UseConfig].theta),	
		sin(m_SphericalCoord[m_UseConfig].theta) * cos(m_SphericalCoord[m_UseConfig].phi));
	
	m_Position[m_UseConfig] = m_Center[m_UseConfig] + position;
	m_ViewMatrix[m_UseConfig] = glm::lookAt(position + m_Center[m_UseConfig], m_Center[m_UseConfig], m_Up[m_UseConfig]);

	m_W[m_UseConfig] = glm::normalize(glm::vec3(glm::inverse(m_ViewMatrix[m_UseConfig]) * glm::vec4(0.f, 0.f, 1.f, 0.f)));
    m_U[m_UseConfig] = glm::normalize(glm::vec3(glm::inverse(m_ViewMatrix[m_UseConfig]) * glm::vec4(-1.f, 0.f, 0.f, 0.f)));
	m_V[m_UseConfig] = glm::normalize(glm::vec3(glm::inverse(m_ViewMatrix[m_UseConfig]) * glm::vec4(0.f, 1.f, 0.f, 0.f)));
		
    m_U[m_UseConfig] *= m_Scale;
    m_V[m_UseConfig] *= m_Scale;

    m_IO[m_UseConfig] = -m_W[m_UseConfig] - 0.5f * m_U[m_UseConfig] - 0.5f * m_V[m_UseConfig];
}

void CCamera::PrintConfig()
{
	glm::vec3 pos = GetPosition();
	std::cout << "Position: (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
	std::cout << "Center: (" << m_Center[m_UseConfig].x << ", " << m_Center[m_UseConfig].y << ", " << m_Center[m_UseConfig].z << ")" << std::endl;
}

Ray CCamera::GetEyeRay(float p_x, float p_y)
{
	const float u = float(p_x) - float(m_Width)/2.f + 0.5f;
    const float v = float(p_y) - float(m_Height)/2.f + 0.5f;
    const glm::vec3 dir = m_IO[m_UseConfig] + u * m_U[m_UseConfig] + v * m_V[m_UseConfig];
    
	Ray r(m_Position[m_UseConfig], dir);
	return r;
}

void CCamera::GetEyeRays(std::vector<Ray>& rays, std::vector<glm::vec2>& samples, uint numRays)
{
	rays.clear();
	samples.clear();
	glm::vec2 range(m_Width, m_Height);
	GetStratifiedSamples2D(samples, range, numRays);
	for(uint i = 0; i < samples.size(); ++i)
	{
		rays.push_back(GetEyeRay(samples[i].x, samples[i].y));
	}
}

float CCamera::GetEyeRayPdf()
{
	const float a = (m_FOV * m_Aspect) * ( 1 - cos(m_FOV) );
	return 1.f / a;
};