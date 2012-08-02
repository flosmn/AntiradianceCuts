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
	const float aspect = (float)m_Width / (float)m_Height;
	 
	m_ProjectionMatrix = glm::perspective(180.f/PI * m_FOV, aspect, zNear, zFar);

	m_Center = glm::vec3(0.0f, 0.0f, 0.0f);
	m_SphericalCoord.r = 10.0f; 
	m_SphericalCoord.phi = 0.0f;
	m_SphericalCoord.theta = PI/2.0f;
	m_Speed = 2.0f;

	m_Up = glm::vec3(0.f, 1.f, 0.f);
	m_Scale = 2.0f / m_Height * tanf(0.5f * m_FOV);
}

void CCamera::Init(glm::vec3 position, glm::vec3 center, float speed)
{
	m_Center = center;
	m_Position = position;

	glm::vec3 delta = position - center;

	m_SphericalCoord.r = glm::length(delta);
	m_SphericalCoord.phi = atan2(delta.x, delta.z);
	m_SphericalCoord.theta = acos(delta.y / m_SphericalCoord.r);

	m_Speed = speed;
	UpdateData();
}

glm::mat4 CCamera::GetProjectionMatrix() const
{
	return m_ProjectionMatrix;
}

glm::mat4 CCamera::GetViewMatrix() const 
{
	return m_ViewMatrix;
}

inline glm::vec3 CCamera::GetPosition() const
{
	return m_Position;
}

void CCamera::ZoomOut(float t) 
{
	m_SphericalCoord.r += 50 *t * m_Speed;
	UpdateData();
}

void CCamera::ZoomIn(float t) 
{
	m_SphericalCoord.r -= 50 * t * m_Speed;
	m_SphericalCoord.r = std::max(m_SphericalCoord.r, 1.0f);
	UpdateData();
}

void CCamera::RotLeft(float t) {
	m_SphericalCoord.phi -= t * m_Speed;
	UpdateData();
}

void CCamera::RotRight(float t) {
	m_SphericalCoord.phi += t * m_Speed;
	UpdateData();
}

void CCamera::RotUp(float t) {
	m_SphericalCoord.theta -= t * m_Speed;
	UpdateData();
}

void CCamera::RotDown(float t) {
	m_SphericalCoord.theta += t * m_Speed;
	UpdateData();
}

void CCamera::MoveForward(float t)
{
	glm::vec3 w = glm::normalize(m_Center - m_Position);
	m_Center += t * 25 * m_Speed * w;
	UpdateData();
}

void CCamera::MoveBackward(float t)
{
	glm::vec3 w = glm::normalize(m_Center - m_Position);
	m_Center -= t * 25 * m_Speed * w;
	UpdateData();
}

void CCamera::UpdateData()
{
	glm::vec3 position = m_SphericalCoord.r * glm::vec3(
		sin(m_SphericalCoord.theta) * sin(m_SphericalCoord.phi),
		cos(m_SphericalCoord.theta),	
		sin(m_SphericalCoord.theta) * cos(m_SphericalCoord.phi));
	
	m_Position = position + m_Center;
	m_ViewMatrix = glm::lookAt(position + m_Center, m_Center, m_Up);

	m_W = glm::normalize(GetPosition() - m_Center);
    m_U = glm::normalize(glm::cross(-m_W, m_Up));
	m_V = glm::normalize(glm::cross(-m_W, m_U));
	m_V.y *= -1.f;
	    
    m_U *= m_Scale;
    m_V *= m_Scale;

    m_IO = -m_W - 0.5f * m_U - 0.5f * m_V;
}

void CCamera::PrintConfig()
{
	glm::vec3 pos = GetPosition();
	std::cout << "Position: (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
	std::cout << "Center: (" << m_Center.x << ", " << m_Center.y << ", " << m_Center.z << ")" << std::endl;
}

Ray CCamera::GetEyeRay(uint p_x, uint p_y)
{
	const float u = float(p_x) - float(m_Width)/2.f + 0.5f;
    const float v = float(p_y) - float(m_Height)/2.f + 0.5f;
    const glm::vec3 dir = m_IO + u * m_U + v * m_V;
    
	Ray r(GetPosition(), glm::normalize(dir));
	return r;
}

void CCamera::GetEyeRays(std::vector<Ray>& rays, std::vector<glm::vec2>& samples, uint numRays)
{
	glm::vec2 range(m_Width, m_Height);
	GetStratifiedSamples2D(samples, range, numRays);
	for(uint i = 0; i < samples.size(); ++i)
	{
		rays.push_back(GetEyeRay(samples[i].x, samples[i].y));
	}
}