#ifndef AREA_LIGHT_H
#define AREA_LIGHT_H

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "Defines.h"

class CModel;
class CCamera;
class Light;
class COGLUniformBuffer;

class AreaLight
{
public:
	AreaLight(float _width, float _height, glm::vec3 _centerPosition, 
		glm::vec3 _frontDirection, glm::vec3 _upDirection);
	~AreaLight();

	bool Init();
	void Release();

	void Draw(CCamera* camera, COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight);

	glm::vec3 GetCenterPosition() { return centerPosition; }
	glm::vec3 GetFrontDirection() { return frontDirection; }
	glm::vec2 GetDimensions() { return glm::vec2(width, height); }
	glm::mat4 GetWorldTransform();

	glm::vec3 SamplePos(float& pdf);
	glm::vec3 SampleDir(float& pdf, int order);
	
	glm::vec3 GetFlux() { return flux;}
	glm::vec3 GetIntensity() { return intensity; }
	glm::vec3 GetRadiance() { return radiance; }

	float GetArea() { return area; }

	void SetCenterPosition(glm::vec3 pos);
	void SetFrontDirection(glm::vec3 dir);

	void SetFlux(glm::vec3 _flux);
	void SetIntensity(glm::vec3 _intensity);
	void SetRadiance(glm::vec3 _radiance);

private:
	void UpdateWorldTransform();

	CModel* m_pAreaLightModel;
	float width;
	float height;
	float area;
	glm::vec3 centerPosition;
	glm::vec3 frontDirection;
	glm::vec3 upDirection;
	glm::vec3 intensity;
	glm::vec3 radiance;
	glm::vec3 flux;

	float* m_pPlaneHammersleyNumbers;
	int m_PlaneHammersleyIndex;
};

#endif