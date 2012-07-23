#include "CCamera.h"

#include "Defines.h"

#include <glm/gtx/transform.hpp>

#include <algorithm>
#include <iostream>

CCamera::CCamera(int w, int h, float zNear, float zFar) 
{
	m_Width = w;
	m_Height = h;
 
	m_ProjectionMatrix = glm::perspective(45.0f, (float)m_Width / (float)m_Height, zNear, zFar);

	m_Center = glm::vec3(0.0f, 0.0f, 0.0f);
	m_SphericalCoord.r = 10.0f; 
	m_SphericalCoord.phi = 0.0f;
	m_SphericalCoord.theta = PI/2.0f;
	m_Speed = 2.0f;
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
}

glm::mat4 CCamera::GetProjectionMatrix() 
{
	return m_ProjectionMatrix;
}

glm::mat4 CCamera::GetViewMatrix() 
{
	glm::vec3 position = m_SphericalCoord.r * glm::vec3(
		sin(m_SphericalCoord.theta) * sin(m_SphericalCoord.phi),
		cos(m_SphericalCoord.theta),	
		sin(m_SphericalCoord.theta) * cos(m_SphericalCoord.phi));
	
	m_ViewMatrix = glm::lookAt(position + m_Center, m_Center, glm::vec3(0.0f, 1.0f, 0.0f));
	
	return m_ViewMatrix;
}

glm::vec3 CCamera::GetPosition() 
{
	return m_SphericalCoord.r * glm::vec3(
		sin(m_SphericalCoord.theta) * sin(m_SphericalCoord.phi),
		cos(m_SphericalCoord.theta),
		sin(m_SphericalCoord.theta) * cos(m_SphericalCoord.phi));
}

void CCamera::ZoomOut(float t) 
{
		m_SphericalCoord.r += 50 *t * m_Speed;
}

void CCamera::ZoomIn(float t) 
{
		m_SphericalCoord.r -= 50 * t * m_Speed;
		m_SphericalCoord.r = std::max(m_SphericalCoord.r, 1.0f);
}

void CCamera::RotLeft(float t) {
		m_SphericalCoord.phi -= t * m_Speed;
}

void CCamera::RotRight(float t) {
		m_SphericalCoord.phi += t * m_Speed;
}

void CCamera::RotUp(float t) {
		m_SphericalCoord.theta -= t * m_Speed;
}

void CCamera::RotDown(float t) {
		m_SphericalCoord.theta += t * m_Speed;
}

void CCamera::PrintConfig()
{
	glm::vec3 pos = GetPosition();
	std::cout << "Position: (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
	std::cout << "Center: (" << m_Center.x << ", " << m_Center.y << ", " << m_Center.z << ")" << std::endl;
}