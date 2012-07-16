#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.3183
#define PI 3.14159

vec2 GetTexCoordForDir(vec3 dir) {
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

uniform atlas_info
{
	int dim_atlas;
	int dim_tile;
} uAtlasInfo;

uniform info_block
{
	int numLights;
	int numClusters;
	int drawLightingOfLight;
	int filterAVPLAtlas;
	
	int lightTreeCutDepth;
	float clusterRefinementThreshold;
	int padd1;
	int padd2;
} uInfo;

uniform config
{
	float GeoTermLimit;
	float BlurSigma;
	float BlurK;
	int UseAntiradiance;
	int DrawAntiradiance;
	int nPaths;
	int N;
	float Bias;
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
layout(binding=3) uniform samplerBuffer samplerClusterBuffer;
layout(binding=4) uniform sampler2D samplerLightAtlas;
layout(binding=5) uniform sampler2D samplerLightClusterAtlas;

void ProcessLight(in int i, in vec3 vPositionWS, in vec3 vNormalWS, out vec4 radiance, out vec4 antiradiance);
void ProcessCluster(in int clusterId, in vec3 vPositionWS, in vec3 vNormalWS, out vec4 radiance, out vec4 antiradiance);

float G(vec3 p1, vec3 n1, vec3 p2, vec3 n3);
float G_CLAMP(vec3 p1, vec3 n1, vec3 p2, vec3 n3);

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;
	
	const float N = float(uConfig.N);
	const float PI_OVER_N = PI / N;
	
	vec3 vPositionWS = texture2D(samplerPositionWS, coord).xyz;
	vec3 vNormalWS = normalize(texture2D(samplerNormalWS, coord).xyz);

	outputDiff = vec4(0.f);
	outputRadiance = vec4(0.f);
	outputAntiradiance = vec4(0.f);
	
	int size_cluster = 4 * 4;
	int cut_size = 0;
	int indicesStack[64];
	int stackPointer = 0;
	
	indicesStack[stackPointer++] = (uInfo.numClusters-1); // push root id onto stack

	int breakLoop = 0;
	int currentId;
	while(stackPointer != 0 && breakLoop < 100000)
	{
		currentId = indicesStack[--stackPointer];
		
		// process current node
		int depth = int(texelFetch(samplerClusterBuffer, currentId * size_cluster + 8).r);
		if(uInfo.lightTreeCutDepth == -1)
		{
			// use every leaf
			int rightId = int(texelFetch(samplerClusterBuffer, currentId * size_cluster + 14).r);
			int leftId = int(texelFetch(samplerClusterBuffer, currentId * size_cluster + 13).r);
			
			const bool leaf = (rightId == -1 && leftId == -1);
			if(leaf)
			{		
				vec4 radiance = vec4(0.f);
				vec4 antiradiance = vec4(0.f);
				vec4 diff = vec4(0.f);
			
				ProcessCluster(currentId, vPositionWS, vNormalWS, radiance, antiradiance);		
			
				outputDiff += radiance - antiradiance;
				outputRadiance += radiance;
				outputAntiradiance += antiradiance;	

				cut_size += 1;
			}
			else
			{			
				// decide whether to refine or use clustering
				int clusterSize = int(texelFetch(samplerClusterBuffer, currentId * size_cluster + 12).r);
				bool refine = false;
				if(clusterSize >= uInfo.numLights / 10)
				{
					//refine
					if(rightId != -1)
						indicesStack[stackPointer++] = rightId;	// issue traversal of right subtree
					if(leftId != -1)
						indicesStack[stackPointer++] = leftId;	// issue traversal of left subtree	
				}
				else
				{
					vec4 radianceCluster = vec4(0.f);
					vec4 antiradianceCluster = vec4(0.f);
					vec4 diffCluster = vec4(0.f);
					vec4 radianceRightChild = vec4(0.f);
					vec4 antiradianceRightChild = vec4(0.f);
					vec4 diffRightChild = vec4(0.f);
					vec4 radianceLeftChild = vec4(0.f);
					vec4 antiradianceLeftChild = vec4(0.f);
					vec4 diffLeftChild = vec4(0.f);

					ProcessCluster(currentId, vPositionWS, vNormalWS, radianceCluster, antiradianceCluster);
					ProcessCluster(rightId, vPositionWS, vNormalWS, radianceRightChild, antiradianceRightChild);
					ProcessCluster(leftId, vPositionWS, vNormalWS, radianceLeftChild, antiradianceLeftChild);
					diffCluster = radianceCluster - antiradianceCluster;
					diffRightChild = radianceRightChild - antiradianceRightChild;
					diffLeftChild = radianceLeftChild - antiradianceLeftChild;

					vec3 intensity = vec3(0.f);
					intensity.x = texelFetch(samplerClusterBuffer, currentId * size_cluster + 5).r;
					intensity.y = texelFetch(samplerClusterBuffer, currentId * size_cluster + 6).r;
					intensity.z = texelFetch(samplerClusterBuffer, currentId * size_cluster + 7).r;

					const vec4 cChilds  = diffRightChild + diffLeftChild;
					const vec4 cCluster = diffCluster;

					if(length(cChilds - cCluster) >= uInfo.clusterRefinementThreshold)
					{
						//refine
						if(rightId != -1)
							indicesStack[stackPointer++] = rightId;	// issue traversal of right subtree
						if(leftId != -1)
							indicesStack[stackPointer++] = leftId;	// issue traversal of left subtree	
					}
					else
					{
						outputDiff += diffCluster;
						outputRadiance += radianceCluster;
						outputAntiradiance += antiradianceCluster;	

						cut_size += 1;
					}
				}			
			}
		}
		else
		{
			int rightId = int(texelFetch(samplerClusterBuffer, currentId * size_cluster + 14).r);
			int leftId = int(texelFetch(samplerClusterBuffer, currentId * size_cluster + 13).r);
			const bool leaf = rightId == -1 && leftId == -1;

			if(depth == uInfo.lightTreeCutDepth || leaf)
			{			
				vec4 radiance = vec4(0.f);
				vec4 antiradiance = vec4(0.f);
				vec4 diff = vec4(0.f);
								
				ProcessCluster(currentId, vPositionWS, vNormalWS, radiance, antiradiance);		
			
				outputDiff += radiance - antiradiance;
				outputRadiance += radiance;
				outputAntiradiance += antiradiance;	

				cut_size += 1;
			}
			else
			{							
				if(rightId != -1)
					indicesStack[stackPointer++] = rightId;	// issue traversal of right subtree
				if(leftId != -1)
					indicesStack[stackPointer++] = leftId;	// issue traversal of left subtree		
			}
		}
		
		breakLoop++;		
	}

	if(breakLoop >= 100000)
		outputDiff += vec4(0.f, 0.f, 1000.f, 1.f);
		
	const float frac_temp01 = float(cut_size) / float(uInfo.numLights);
	outputCutSizes = vec4(frac_temp01, frac_temp01, frac_temp01, 0.f);
	outputDiff.w = 1.f;
	outputRadiance.w = 1.f;
	outputAntiradiance.w = 1.f;
}

void ProcessCluster(in int clusterId, in vec3 vPositionWS, in vec3 vNormalWS, out vec4 radiance, out vec4 antiradiance)
{
	vec4 rad = vec4(0.f);
	vec4 antirad = vec4(0.f);
	
	int size_cluster = 4 * 4;

	int rightId = int(texelFetch(samplerClusterBuffer, clusterId * size_cluster + 14).r);
	int leftId = int(texelFetch(samplerClusterBuffer, clusterId * size_cluster + 13).r);
	
	const bool leaf = (rightId == -1 && leftId == -1);
	int atlasIndex = leaf ? clusterId : clusterId - uInfo.numLights;
		
	const vec3 p = vec3(
		texelFetch(samplerClusterBuffer, clusterId * size_cluster + 1).r,
		texelFetch(samplerClusterBuffer, clusterId * size_cluster + 2).r,
		texelFetch(samplerClusterBuffer, clusterId * size_cluster + 3).r);
		
	const vec3 direction = normalize(vPositionWS - p);
	
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
	const int row = atlasIndex / columns;
	const int column = atlasIndex % columns;
	const vec2 globalOffset = vec2(column * uAtlasInfo.dim_tile, row * uAtlasInfo.dim_tile);
	const vec2 texel_global = texel_local + globalOffset;
	
	vec4 sam;
	if(leaf)
		sam = texture2D(samplerLightAtlas, (1.f/float(uAtlasInfo.dim_atlas)) * texel_global);
	else
		sam = texture2D(samplerLightClusterAtlas, (1.f/float(uAtlasInfo.dim_atlas)) * texel_global);
	
	rad = max(vec4(0.f), sam);
	antirad = -min(vec4(0.f), sam);
	
	const float dist = length(vPositionWS - p);
	const float cos_theta = clamp(dot(vNormalWS, -direction), 0, 1);
	rad = clamp(cos_theta / (dist * dist), 0, uConfig.GeoTermLimit) * rad;
	
	antirad = (cos_theta) / (dist * dist) * antirad;

	antiradiance = antirad;
	radiance = rad;
}

void ProcessLight(in int i, in vec3 vPositionWS, in vec3 vNormalWS, out vec4 radiance, out vec4 antiradiance)
{
	vec4 rad = vec4(0.f);
	vec4 antirad = vec4(0.f);

	int size = 5 * 4; // sizeof(AVPL_BUFFER)
	const vec3 p = vec3(
		texelFetch(samplerLightBuffer, i * size + 8).r,
		texelFetch(samplerLightBuffer, i * size + 9).r,
		texelFetch(samplerLightBuffer, i * size + 10).r);

	const vec3 direction = normalize(vPositionWS - p);
	
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
	const int row = i / columns;
	const int column = i % columns;
	const vec2 globalOffset = vec2(column * uAtlasInfo.dim_tile, row * uAtlasInfo.dim_tile);
	const vec2 texel_global = texel_local + globalOffset;
	
	vec4 sam = texture2D(samplerLightAtlas, (1.f/float(uAtlasInfo.dim_atlas)) * texel_global);
	
	rad = max(vec4(0.f), sam);
	antirad = -min(vec4(0.f), sam);
	
	const float dist = length(vPositionWS - p);
	const float cos_theta = clamp(dot(vNormalWS, -direction), 0, 1);
	rad = clamp(cos_theta / (dist * dist), 0, uConfig.GeoTermLimit) * rad;
	
	antirad = (cos_theta) / (dist * dist) * antirad;

	antiradiance = antirad;
	radiance = rad;
}

float G_CLAMP(in vec3 p1, in vec3 n1, in vec3 p2, in vec3 n2)
{
	return clamp(G(p1, n1, p2, n2), 0, uConfig.GeoTermLimit);
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