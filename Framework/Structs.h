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

struct MODEL
{
	glm::vec3 positionWS;
};

struct AVPL_STRUCT
{
	glm::mat4 ViewMatrix;
	glm::mat4 ProjectionMatrix;
	glm::vec4 L;		// radiance
	glm::vec4 A;		// antiradiance;
	glm::vec4 pos;		// Position
	glm::vec4 norm;		// orientation;
	glm::vec4 w;		// direction of antiradiance, incident light direction;
	glm::vec4 DebugColor;
	float angleFactor;	
	float bounce;
	float materialIndex;
	float padd0;
};

struct AVPL_BUFFER
{
	glm::vec4 L;	// radiance;
	glm::vec4 A;	// antiradiance;
	glm::vec4 pos;	// position
	glm::vec4 norm;	// orientation;
	glm::vec4 w;	// direction of antiradiance, incident light direction;
	float angleFactor;	// PI/AngleFactor = Half opening angleFactor of AR cone
	float materialIndex;
	float padd0;
	float padd1;
};

struct CLUSTER_BUFFER
{
	float id;					// 0
	glm::vec3 mean;
		
	float avplIndex;			// 4
	glm::vec3 intensity;
	
	float depth;				// 8
	glm::vec3 normal;
	
	float size;				// 12
	float left_id;			// 13
	float right_id;			// 14
	float padd;				// 15

	glm::vec3 pMin;			// 16
	float padd0;

	glm::vec3 pMax;			// 20
	float padd1;
};

struct CONFIG
{
	float GeoTermLimitRadiance;
	float GeoTermLimitAntiradiance;
	float AntiradFilterK;
	float AntiradFilterGaussFactor;
	int ClampGeoTerm;
	int AntiradFilterMode;	
	int nPaths;
	int padd;
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
	int numClusters;
	int UseIBL;
	int filterAVPLAtlas;
	
	int lightTreeCutDepth;
	float clusterRefinementThreshold;
	int padd1;
	int padd2;
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

struct CLUSTERING
{
	int leftChildId;
	int rightChildId;
	int isLeaf;
	int isAlreadyCalculated;
};

#endif // _STRUCTS_H_