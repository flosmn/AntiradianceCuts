#ifndef AREA_LIGHT_H
#define AREA_LIGHT_H

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "Defines.h"

class CModel;
class Camera;
class Light;
class CGLUniformBuffer;

class AreaLight
{
public:
	AreaLight(float _width, float _height, glm::vec3 _centerPosition, 
		glm::vec3 _frontDirection, glm::vec3 _upDirection, glm::vec3 _flux);
	~AreaLight();

	bool Init();
	void Release();

	void Draw(Camera* camera, CGLUniformBuffer* pUBTransform);

	glm::vec3 GetCenterPosition() { return centerPosition; }
	glm::vec3 GetFrontDirection() { return frontDirection; }
	glm::vec2 GetDimensions() { return glm::vec2(width, height); }
	glm::mat4 GetWorldTransform();

	Light* GetNewPrimaryLight(float& pdf);
	glm::vec3 SamplePos(float& pdf);
	glm::vec3 SampleDir(float& pdf, int order);
	float GetArea() { return area; }
	glm::vec3 GetFlux() { return flux; }
	glm::vec3 GetRadiance() { return flux / (PI * area); }

private:
	GLuint drawAreaLightProgram;
	GLuint uniformModelMatrix;
	GLuint uniformViewMatrix;
	GLuint uniformProjectionMatrix;
	GLuint uniformNormalMatrix;
	GLuint uniformLightIntensity;
	
	CModel* m_pAreaLightModel;
	float width;
	float height;
	float area;
	glm::vec3 centerPosition;
	glm::vec3 frontDirection;
	glm::vec3 upDirection;
	glm::vec3 flux;

	float* m_pPlaneHammersleyNumbers;
	int m_PlaneHammersleyIndex;
};

#endif