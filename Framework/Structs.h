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

struct AVPL_TRANSFORM
{
	glm::mat4 ViewMatrix;
	glm::mat4 ProjectionMatrix;
};

struct AVPL_DATA
{
	glm::vec4 L;			// radiance
	glm::vec4 A;			// antiradiance;
	
	glm::vec3 pos;			// position
	float materialIndex;	// Index in the material buffer
	
	glm::vec3 norm;			// orientation;
	float angleFactor;	
	
	glm::vec3 w;			// direction of antiradiance, incident light direction;
	float bounnce;			// order of indiection

	glm::vec4 DebugColor;
	
	glm::vec3 pMin;
	float padd0;

	glm::vec3 pMax;
	float padd1;

	float left_id;
	float right_id;
	float size;
	float depth;	// depth in the cluster tree
};

struct AVPL_BUFFER
{
	glm::vec4 L;	// radiance;
	glm::vec4 A;	// antiradiance;
	glm::vec4 pos;	// position
	glm::vec4 norm;	// orientation;
	glm::vec4 w;	// direction of antiradiance, incident light direction;
	float angleFactor;	// M_PI/AngleFactor = Half opening angleFactor of AR cone
	float materialIndex;

	float padd0;
	float padd1;
};

struct CLUSTER_BUFFER
{
	glm::vec3 mean;
	float depth;
	
	glm::vec3 intensity;
	float avplIndex;
	
	glm::vec3 normal;
	float id;	
	
	float left_id;
	float right_id;
	float size;
	float padd;

	glm::vec3 pMin;
	float padd0;

	glm::vec3 pMax;
	float padd1;

	glm::vec3 incomingDirection;
	float materialIndex;
};

struct CONFIG
{
	float GeoTermLimitRadiance;
	float GeoTermLimitAntiradiance;
	float AntiradFilterK;
	float AntiradFilterGaussFactor;
	int ClampGeoTerm;
	int AntiradFilterMode;	
	int padd;
	int padd1;
};

struct CAMERA
{
	glm::vec3 positionWS;
	int width;
	int height;
};

struct INFO
{
	int numLights;
	int numClusters;
	int UseIBL;
	int filterAVPLAtlas;
	
	int lightTreeCutDepth;
	float clusterRefinementMaxRadiance;
	float clusterRefinementWeight;
	float clusterRefinementThreshold;
};

struct ATLAS_INFO
{
	int dim_atlas;
	int dim_tile;
};

struct CLUSTERING
{
	int leftChildId;
	int rightChildId;
	int isLeaf;
	int isAlreadyCalculated;
};

#endif // _STRUCTS_H_
