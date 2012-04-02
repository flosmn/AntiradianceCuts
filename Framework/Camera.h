#ifndef CAMERA_H
#define CAMERA_H

typedef unsigned int uint;

#include "glm/glm.hpp"

struct SphereCoord 
{
		float r;
		float phi;
		float theta;
};

class Camera 
{
public:
	Camera(int width, int height, float zNear, float zFar);

	glm::mat4 GetProjectionMatrix();
	glm::mat4 GetViewMatrix();
	glm::vec3 GetPosition();

	void Init(glm::vec3 _position, glm::vec3 _center, float _speed);

	void RotLeft(float t);
	void RotRight(float t);
	void RotUp(float t);
	void RotDown(float t);
	void ZoomOut(float t);
	void ZoomIn(float t);
	void SetSpeed(float s);

	float GetAspectRatio() { return (float)width/(float)height; }
	uint GetHeight() { return height; }
	uint GetWidth() { return width; }
	
	void SetHeight(uint _height) { height = _height; }
	void SetWidth(uint _width) { width = _width; }

private:
	uint width, height;
	SphereCoord coord;
	float speed;
	
	glm::vec3 center;
	glm::mat4 projectionMatrix;
	glm::mat4 viewMatrix;
};

#endif