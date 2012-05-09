#ifndef _STRUCTS_H_
#define _STRUCTS_H_

typedef unsigned int uint;

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
	glm::vec4 Contrib;
	glm::vec4 SrcPosition;
	glm::vec4 SrcOrientation;
	glm::vec4 SrcContrib;
	glm::vec4 DebugColor;
	glm::mat4 ViewMatrix;
	glm::mat4 ProjectionMatrix;
	int Bounce;
};

struct LIGHT_BUFFER
{
	glm::vec4 Position;			// 0 - 3
	glm::vec4 Orientation;		// 4- 7
	glm::vec4 Contrib;			// 8 - 11
	glm::vec4 SrcPosition;		// 12 - 15
	glm::vec4 SrcOrientation;	// 16 - 19
	glm::vec4 SrcContrib;		// 20 - 23
	glm::vec4 padd0;
	glm::vec4 padd1;
};

struct CONFIG
{
	float GeoTermLimit;
	float BlurSigma;
	float BlurK;
	int UseAntiradiance;
	int DrawAntiradiance;
	int nPaths;
	int N;
	float Bias;
};

struct CAMERA
{
	glm::vec3 positionWS;
	int width;
	int height;
};

struct POINT_CLOUD_POINT
{
	glm::vec4 position;
	glm::vec4 color;
};

struct INFO
{
	int numLights;
	float debugColorR;
	float debugColorG;
	float debugColorB;
};

#endif // _STRUCTS_H_