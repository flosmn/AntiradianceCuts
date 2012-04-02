#include "Camera.h"

#include "Defines.h"

#include <glm/gtx/transform.hpp>

#include <algorithm>

Camera::Camera(int w, int h, float zNear, float zFar) 
{
	width = w;
	height = h;
 
	projectionMatrix = glm::perspective(45.0f, (float)width / (float)height, 
																			zNear, zFar);

	center = glm::vec3(0.0f, 0.0f, 0.0f);
	coord.r = 10.0f; coord.phi = 0.0f; coord.theta = PI/2.0f;
	speed = 2.0f;
}

void Camera::Init(glm::vec3 _position, glm::vec3 _center, float _speed)
{
	center = _center;

	glm::vec3 delta = _position - _center;

	coord.r = glm::length(delta);
	coord.phi = atan2(delta.x, delta.z);
	coord.theta = acos(delta.y / coord.r);

	speed = _speed;
}

glm::mat4 Camera::GetProjectionMatrix() 
{
	return projectionMatrix;
}

glm::mat4 Camera::GetViewMatrix() 
{
	glm::vec3 position = glm::vec3(coord.r * sin(coord.theta) * sin(coord.phi),
																 coord.r * cos(coord.theta),													 
																 coord.r * sin(coord.theta) * cos(coord.phi));
	
	viewMatrix = glm::lookAt(position + center, 
																				 center, 
																				 glm::vec3(0.0f, 1.0f, 0.0f));
	
	return viewMatrix;
}

glm::vec3 Camera::GetPosition() 
{
	return glm::vec3(coord.r * sin(coord.theta) * sin(coord.phi),
									 coord.r * cos(coord.theta),													 
									 coord.r * sin(coord.theta) * cos(coord.phi));
}

void Camera::ZoomOut(float t) 
{
		coord.r += t * speed;
}

void Camera::ZoomIn(float t) 
{
		coord.r -= t * speed;
		coord.r = std::max(coord.r, 1.0f);
}

void Camera::RotLeft(float t) {
		coord.phi -= t * speed;
}

void Camera::RotRight(float t) {
		coord.phi += t * speed;
}

void Camera::RotUp(float t) {
		coord.theta -= t * speed;
}

void Camera::RotDown(float t) {
		coord.theta += t * speed;
}