#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.3183f
#define PI 3.14159f
#define ONE_OVER_THREE 0.333333f

#define EPSILON 0.001f
#define SIZE_CLUSTER 7
#define SIZE_MATERIAL 4
#define MAX_CLUSTER_SIZE 5

const bool useDiskAreaLight = false;
const bool useAtlas = false;
const bool useAntiradiance = false;

uniform atlas_info
{
	int dim_atlas;
	int dim_tile;
} uAtlasInfo;

uniform info_block
{
	int numLights;
	int numClusters;
	int UseIBL;
	int filterAVPLAtlas;
	int lightTreeCutDepth;
	float clusterRefinementMaxRadiance;
	float clusterRefinementWeight;
	float clusterRefinementThreshold;
} uInfo;

uniform config
{
	float GeoTermLimitRadiance;
	float GeoTermLimitAntiradiance;
	float AntiradFilterK;
	float AntiradFilterGaussFactor;
	int ClampGeoTerm;
	int AntiradFilterMode;	
	int padd;
	int padd1;
} uConfig;

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

layout(location = 0) out vec4 outputDiff;
layout(location = 1) out vec4 outputRadiance;
layout(location = 2) out vec4 outputAntiradiance;
layout(location = 3) out vec4 outputCutSizes;

layout(binding=0) uniform sampler2D samplerPositionWS;
layout(binding=1) uniform sampler2D samplerNormalWS;
layout(binding=2) uniform samplerBuffer samplerLightBuffer;
layout(binding=3) uniform samplerBuffer samplerMaterialBuffer;
layout(binding=4) uniform samplerBuffer samplerClusterBuffer;
layout(binding=5) uniform sampler2D samplerLightAtlas;
layout(binding=6) uniform sampler2D samplerLightClusterAtlas;

void ProcessCluster(in int clusterId, in vec3 mean, in vec3 vPositionWS, in vec3 vNormalWS, 
	out vec4 radiance, out vec4 antiradiance,
	in vec4 diffuse, in vec4 specular, in float exponent);

void AccessAvplAtlas(in vec3 direction, inout vec4 radiance, inout vec4 antiradiance, int avpl_id, bool leaf);

float map(float x, float x0, float x1, float y0, float y1);
float avg(in vec3 c);
float avg(in vec4 c);

vec4 f_r(in vec3 w_i, in vec3 w_o, in vec3 n, in vec4 diffuse, 
	in vec4 specular, in float exponent);
vec4 f_r(in vec3 from, in vec3 over, in vec3 to, in vec3 n, in vec4 diffuse,
	in vec4 specular, in float exponent);

bool Refine(in int cluster_id, in vec3 cluster_center, in vec3 position, in vec3 normal, in vec4 radiance, in vec4 antiradiance);

float G(vec3 p1, vec3 n1, vec3 p2, vec3 n3);
float G_CLAMP(vec3 p1, vec3 n1, vec3 p2, vec3 n3);

vec2 GetTexCoordForDir(vec3 dir);

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;
		
	vec3 vPositionWS = texture2D(samplerPositionWS, coord).xyz;
	vec3 vNormalWS = normalize(texture2D(samplerNormalWS, coord).xyz);

	int materialIndex = int(texture2D(samplerNormalWS, coord).w);
	const vec4 cDiffuse =	texelFetch(samplerMaterialBuffer, materialIndex * SIZE_MATERIAL + 1);
	const vec4 cSpecular =	texelFetch(samplerMaterialBuffer, materialIndex * SIZE_MATERIAL + 2);
	const float exponent =	texelFetch(samplerMaterialBuffer, materialIndex * SIZE_MATERIAL + 3).r;

	outputDiff = vec4(0.f);
	outputRadiance = vec4(0.f);
	outputAntiradiance = vec4(0.f);
	
	int columns = uAtlasInfo.dim_atlas / uAtlasInfo.dim_tile;

	int cut_size = 0;
	int indicesStack[32];
	int stackPointer = 0;
	
	indicesStack[stackPointer++] = (uInfo.numClusters-1); // push root id onto stack

	int breakLoop = 0;
	int currentId;
	while(stackPointer != 0)
	{
		currentId = indicesStack[--stackPointer];
		
		const vec4 temp = texelFetch(samplerClusterBuffer, currentId * SIZE_CLUSTER + 0);
		const vec3 mean = temp.xyz;
		int depth = int(temp.w);

		const vec4 ids = texelFetch(samplerClusterBuffer, currentId * SIZE_CLUSTER + 3);
		int leftId = int(ids.x);
		int rightId = int(ids.y);
		int size = int(ids.z);
		bool bigEnough = (size >= MAX_CLUSTER_SIZE);

		vec4 radiance = vec4(0.f);
		vec4 antiradiance = vec4(0.f);

		ProcessCluster(currentId, mean, vPositionWS, vNormalWS, radiance, antiradiance, cDiffuse, cSpecular, exponent);

		const bool leaf = (rightId == -1 && leftId == -1);
		bool refine = leaf ? false : (!useDiskAreaLight || bigEnough) && Refine(currentId, mean, vPositionWS, vNormalWS, radiance, antiradiance);
		if(refine)
		{
			if(leftId != -1)
				indicesStack[stackPointer++] = leftId;	// issue traversal of left subtree	
			if(rightId != -1)
				indicesStack[stackPointer++] = rightId;	// issue traversal of right subtree
		}
		else
		{
			vec4 diff = vec4(0.f);
			
			if(useAntiradiance)
				outputDiff += radiance - antiradiance;
			else
				outputDiff += radiance;
			outputRadiance += radiance;
			outputAntiradiance += vec4(0.f); //antiradiance;

			cut_size += 1;
		}
		
		breakLoop++;
	}

	if(breakLoop >= 100000)
		outputDiff += vec4(0.f, 0.f, 1000.f, 1.f);

	const float frac_temp01 = float(cut_size) / float(uInfo.numLights);
	outputCutSizes = vec4(frac_temp01, 0.f, 0.f, 1.f);
	
	outputDiff.w = 1.f;
	outputRadiance.w = 1.f;
	outputAntiradiance.w = 1.f;
}

void ProcessCluster(in int clusterId, in vec3 avpl_pos, in vec3 pos, in vec3 norm, 
	out vec4 radiance, out vec4 antiradiance, in vec4 diffuse, in vec4 specular, in float exponent)
{	
	const vec4 ids = texelFetch(samplerClusterBuffer, clusterId * SIZE_CLUSTER + 3);
	int leftId = int(ids.x);
	int rightId = int(ids.y);
	int size = int(ids.z);
	
	const vec4 avplInfo = texelFetch(samplerClusterBuffer, clusterId * SIZE_CLUSTER + 6);
	const vec3 incomingDirectionAVPL = normalize(avplInfo.xyz);
	int materialIndex = int(avplInfo.w);

	const vec4 cDiffuseAVPL = texelFetch(samplerMaterialBuffer, materialIndex * SIZE_MATERIAL + 1);
	const vec4 cSpecularAVPL = texelFetch(samplerMaterialBuffer, materialIndex * SIZE_MATERIAL + 2);
	const float exponentAVPL = texelFetch(samplerMaterialBuffer, materialIndex * SIZE_MATERIAL + 3).r;
	
	const vec3 avpl_norm = normalize(texelFetch(samplerClusterBuffer, clusterId * SIZE_CLUSTER + 2).xyz);
	const vec3 avpl_to_pos = normalize(pos - avpl_pos);

	vec4 brdf_avpl = f_r(-incomingDirectionAVPL, avpl_to_pos, avpl_norm, cDiffuseAVPL, cSpecularAVPL, exponentAVPL);
	vec4 brdf_pos = f_r(avpl_pos, pos, uCamera.vPositionWS, norm, diffuse, specular, exponent);

	const bool leaf = (rightId == -1 && leftId == -1);
	int atlasIndex = leaf ? clusterId : clusterId - uInfo.numLights;
					
	vec4 rad = vec4(0.f);
	vec4 antirad = vec4(0.f);

	if(!useAtlas)
		rad = vec4(texelFetch(samplerClusterBuffer, clusterId * SIZE_CLUSTER + 1).xyz, 1.f);
	else
		AccessAvplAtlas(avpl_to_pos, rad, antirad, atlasIndex, leaf);
	
	if(dot(norm, -avpl_to_pos) <= 0.f + EPSILON)
		antirad = vec4(0.f);

	if(dot(avpl_norm, avpl_to_pos) >= 0.f - EPSILON)
		antirad = vec4(0.f);
	
	const float dist = length(pos - avpl_pos);
	const float cos_theta_pos_to_avpl = clamp(dot(norm, -avpl_to_pos), 0.f, 1.f);
	const float cos_theta_avpl_to_pos = clamp(dot(avpl_norm, avpl_to_pos), 0.f, 1.f);

	if (!useDiskAreaLight || (size >= MAX_CLUSTER_SIZE))
	{
		float G = (cos_theta_pos_to_avpl) / (dist * dist);
		
		if(!useAtlas)
			G *= cos_theta_avpl_to_pos;

		const float G_rad = uConfig.ClampGeoTerm == 1 ? clamp(G, 0, uConfig.GeoTermLimitRadiance) : G;
		const float G_antirad = uConfig.ClampGeoTerm == 1 ? clamp(G, 0, uConfig.GeoTermLimitAntiradiance) : G;
		
		if(useAtlas)
			antiradiance = brdf_pos * G_antirad * antirad;
		else
			antiradiance = vec4(0.f);

		radiance = brdf_pos * G_rad * rad;

		if(!useAtlas)
			radiance *= brdf_avpl;
	}
	else // treat cluster as area light (disk)
	{
		const vec3 bbMin = texelFetch(samplerClusterBuffer, clusterId * SIZE_CLUSTER + 4).xyz;
		const vec3 bbMax = texelFetch(samplerClusterBuffer, clusterId * SIZE_CLUSTER + 5).xyz;
		const float radius = 0.5f * length(bbMin - bbMax);
		const float area = PI * radius * radius;

		if(!useAtlas)
		{
			antiradiance = vec4(0.f);
			radiance = PI * brdf_pos * (cos_theta_avpl_to_pos * cos_theta_pos_to_avpl) / (area + PI * dist * dist) * brdf_avpl * rad;		
		}
		else
		{
			antiradiance = PI * brdf_pos * (cos_theta_avpl_to_pos * cos_theta_pos_to_avpl) / (area + PI * dist * dist) * antirad;
			radiance = PI * brdf_pos * (cos_theta_pos_to_avpl) / (area + PI * dist * dist) * rad;
		}
	}
}

bool Refine(in int cluster_id, in vec3 cluster_center, in vec3 position, in vec3 normal, in vec4 radiance, in vec4 antiradiance)
{
#if 0

	vec3 pMin = texelFetch(samplerClusterBuffer, cluster_id * SIZE_CLUSTER + 4).xyz;
	vec3 pMax = texelFetch(samplerClusterBuffer, cluster_id * SIZE_CLUSTER + 5).xyz;
	
	const vec3 clusterToPoint = normalize(position - cluster_center);
	const float dist = length(position - cluster_center);
	const float one_over_dist_squared = 1.f / (dist * dist);

	const float A_x = (pMax.y - pMin.y) * (pMax.z - pMin.z);
	const float A_y = (pMax.x - pMin.x) * (pMax.z - pMin.z);
	const float A_z = (pMax.x - pMin.x) * (pMax.y - pMin.y);

	const float proj_A_x = A_x * dot(clusterToPoint, vec3(1.f, 0.f, 0.f));
	const float proj_A_y = A_y * dot(clusterToPoint, vec3(0.f, 1.f, 0.f));
	const float proj_A_z = A_z * dot(clusterToPoint, vec3(0.f, 0.f, 1.f));

	/*const*/ float solid_angle = map(clamp((A_x + A_y + A_z) * one_over_dist_squared, 0.f, PI), 0.f, PI, 0.f, 1.f);
	const vec4 diff = abs(radiance - antiradiance);
	/*const*/ float rad = map(clamp(avg(diff), 0.f, uInfo.clusterRefinementMaxRadiance), 0.f, uInfo.clusterRefinementMaxRadiance, 0.f, 1.f);
	const float alpha = uInfo.clusterRefinementWeight;

	const float weight = solid_angle; //alpha * solid_angle + (1.f-alpha) * rad;

#endif

	const vec3 bbMin = texelFetch(samplerClusterBuffer, cluster_id * SIZE_CLUSTER + 4).xyz;
	const vec3 bbMax = texelFetch(samplerClusterBuffer, cluster_id * SIZE_CLUSTER + 5).xyz;
	const float radius = 0.5f * length(bbMin - bbMax);
	const float area = PI * radius * radius;

	const float dist = length(position - cluster_center);
	const vec3 avpl_to_pos = normalize(position - cluster_center);

	const vec3 avpl_norm = normalize(texelFetch(samplerClusterBuffer, cluster_id * SIZE_CLUSTER + 2).xyz);
	const float cos_theta_pos_to_avpl = clamp(dot(normal, -avpl_to_pos), 0.f, 1.f);
	const float cos_theta_avpl_to_pos = 1.f; //clamp(dot(avpl_norm, avpl_to_pos), 0.f, 1.f);

	const float weight = /*(1.f - dist / (sqrt(radius * radius + dist * dist)));*/
		area / (area + PI * dist * dist);

	if(weight >= uInfo.clusterRefinementThreshold)
	{
		return true;
	}
	return false;
}

void AccessAvplAtlas(in vec3 direction, inout vec4 radiance, inout vec4 antiradiance, int avpl_id, bool leaf)
{
	vec2 texel_local;
	if(uInfo.filterAVPLAtlas > 0)
	{
		texel_local = (uAtlasInfo.dim_tile-2) * GetTexCoordForDir(direction) + vec2(1.f, 1.f);
	}
	else
	{
		texel_local = uAtlasInfo.dim_tile * GetTexCoordForDir(direction);
	}
	
	int columns = uAtlasInfo.dim_atlas / uAtlasInfo.dim_tile;
	const int row = avpl_id / columns;
	const int column = avpl_id % columns;
	const vec2 globalOffset = vec2(column * uAtlasInfo.dim_tile, row * uAtlasInfo.dim_tile);
	const vec2 texel_global = texel_local + globalOffset;
	
	vec4 sam;
	if(leaf)
		sam = texture2D(samplerLightAtlas, (1.f/float(uAtlasInfo.dim_atlas)) * texel_global);
	else
		sam = texture2D(samplerLightClusterAtlas, (1.f/float(uAtlasInfo.dim_atlas)) * texel_global);
	
	radiance = max(vec4(0.f), sam);
	antiradiance = -min(vec4(0.f), sam);
}

float G_CLAMP(in vec3 p1, in vec3 n1, in vec3 p2, in vec3 n2)
{
	float g = G(p1, n1, p2, n2);
	return uConfig.ClampGeoTerm == 1 ? clamp(g, 0, uConfig.GeoTermLimitRadiance) : g;
}

float G(in vec3 p1, in vec3 n1, in vec3 p2, in vec3 n2)
{
	vec3 n_1 = normalize(n1);
	vec3 n_2 = normalize(n2);
	vec3 w = normalize(p2 - p1);

	float cos_theta_1 = clamp(dot(n_1, w), 0, 1);
	float cos_theta_2 = clamp(dot(n_2, -w), 0, 1);

	float dist = length(p2 - p1);
	
	return (cos_theta_1 * cos_theta_2) / (dist * dist);
}

vec4 f_r(in vec3 from, in vec3 over, in vec3 to, in vec3 n, in vec4 diffuse, in vec4 specular, in float exponent)
{
	const vec3 w_i = normalize(from - over);
	const vec3 w_o = normalize(to - over);
	return f_r(w_i, w_o, n, diffuse, specular, exponent);
}

vec4 f_r(in vec3 w_i, in vec3 w_o, in vec3 n, in vec4 diffuse, in vec4 specular, in float exponent)
{
	const vec4 d = ONE_OVER_PI * diffuse;
	const float cos_theta = max(0.f, dot(reflect(-w_i, n), w_o));
	const vec4 s = max(0.5f * ONE_OVER_PI * (exponent+2.f) * pow(cos_theta, exponent) * specular, 0.f);
	return d; //vec4(d.x + s.x, d.y + s.y, d.z + s.z, 1.f);
}

vec2 GetTexCoordForDir(vec3 dir){
	// Project from sphere onto octahedron
	dir /= dot(vec3(1.0f), abs(dir));
	
	// If on lower hemisphere...
	if (dir.y < 0.0f) {
		// ...unfold
		float x = (1.0f - abs(dir.z)) * sign(dir.x);
		float z = (1.0f - abs(dir.x)) * sign(dir.z);
		dir.x = x;
		dir.z = z;
	}

	// [-1,1]^2 to [0,1]^2
	dir.xz = dir.xz * 0.5f + 0.5f;
	
	return dir.xz;
}

float map(float x, float x0, float x1, float y0, float y1) {
	return y1 - (x1 - x) * (y1 - y0) / (x1 - x0);
}

float avg(in vec3 c)
{
	return ONE_OVER_THREE * (c.x + c.y + c.z);
}

float avg(in vec4 c)
{
	return ONE_OVER_THREE * (c.x + c.y + c.z);
}