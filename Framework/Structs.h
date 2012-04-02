#ifndef _STRUCTS_H_
#define _STRUCTS_H_

#include "glm/glm.hpp"

struct TRANSFORM
{
	glm::mat4 M;
	glm::mat4 V;
	glm::mat4 itM;
	glm::mat4 MVP;
};

struct MATERIAL
{
	MATERIAL() : diffuseColor(glm::vec4(0.f, 0.f, 0.f, 0.f)) {}

	glm::vec4 diffuseColor;
};

struct LIGHT
{
	glm::vec4 Position;
	glm::vec4 Orientation;
	glm::vec4 Flux;
	glm::vec4 SrcPosition;
	glm::vec4 SrcOrientation;
	glm::vec4 SrcFlux;
	glm::vec4 DebugColor;
	glm::mat4 ViewMatrix;
	glm::mat4 ProjectionMatrix;
};

struct CONFIG
{
	float uGeoTermLimit;
	float BlurSigma;
	float BlurK;
	int uUseAntiradiance;
	int uDrawAntiradiance;
	int nPaths;
};

struct CAMERA
{
	glm::vec3 positionWS;
	int width;
	int height;
};

#endif // _STRUCTS_H_