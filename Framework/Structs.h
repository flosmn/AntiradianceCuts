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

struct MODEL
{
	glm::vec3 positionWS;
};

struct AVPL_STRUCT
{
	glm::vec4 I;	//Intensity;
	glm::vec4 A;	//Antiintensity;
	glm::vec4 pos;	// Position
	glm::vec4 norm;	//Orientation;
	glm::vec3 w_A;	//AntiintensityDirection;
	int Bounce;
	glm::vec4 DebugColor;
	glm::mat4 ViewMatrix;
	glm::mat4 ProjectionMatrix;
};

struct AVPL_BUFFER
{
	glm::vec4 I;	// Intensity;
	glm::vec4 A;	// Antiintensity;
	glm::vec4 pos;	// Position
	glm::vec4 norm;	// Orientation;
	glm::vec4 w_A;	// AntiintensityDirection;	
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
	int drawLightingOfLight;
	int filterAVPLAtlas;
	float padd;
};

struct AREA_LIGHT
{
	glm::vec4 radiance;
};

struct AVPL_POSITION
{
	glm::vec4 positionWS;
};

struct ATLAS_INFO
{
	int dim_atlas;
	int dim_tile;
};

struct TEST_STRUCT
{
	glm::vec4 color1;
	glm::vec4 color2;
};

#endif // _STRUCTS_H_