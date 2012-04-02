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
	glm::vec4 m_Position;
	glm::vec4 m_Orientation;
	glm::vec4 m_SurfaceAlbedo;
	glm::vec4 m_Flux;
	glm::vec4 m_Antiflux;
	glm::vec4 m_AntiPosition;
	glm::vec4 m_IncLightDir;
	glm::vec4 m_DebugColor;
	glm::mat4 m_ViewMatrix;
	glm::mat4 m_ProjectionMatrix;
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