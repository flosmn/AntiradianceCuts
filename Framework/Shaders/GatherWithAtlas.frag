#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.3183
#define PI 3.14159

#define SIZE_LIGHT 6
#define SIZE_MATERIAL 4
#define EPSILON 0.001f

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
	int UseIBL;
	int filterAVPLAtlas;
	int lightTreeCutDepth;
	float clusterRefinementThreshold;
	float padd1;
	float padd2;
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

layout(binding=0) uniform sampler2D samplerPositionWS;
layout(binding=1) uniform sampler2D samplerNormalWS;
layout(binding=2) uniform samplerBuffer samplerLightBuffer;
layout(binding=3) uniform samplerBuffer samplerMaterialBuffer;
layout(binding=4) uniform sampler2D samplerLightAtlas;

void ProcessLight(in int i, in vec3 vPositionWS, in vec3 vNormalWS, out vec4 radiance, out vec4 antiradiance, in vec4 diffuse, in vec4 specular, in float exponent);

vec4 f_r(in vec3 w_i, in vec3 w_o, in vec3 n, in vec4 diffuse, in vec4 specular, in float exponent);
vec4 f_r(in vec3 from, in vec3 over, in vec3 to, in vec3 n, in vec4 diffuse, in vec4 specular, in float exponent);

float G(vec3 p1, vec3 n1, vec3 p2, vec3 n3);
float G_CLAMP(vec3 p1, vec3 n1, vec3 p2, vec3 n3);

void AccessAvplAtlas(in vec3 direction, inout vec4 radiance, inout vec4 antiradiance, int avpl_id);

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
	
	for(int i = 0; i < uInfo.numLights; ++i)
	{		
		vec4 radiance = vec4(0.f);
		vec4 antiradiance = vec4(0.f);
		vec4 diff = vec4(0.f);
		
		ProcessLight(i, vPositionWS, vNormalWS, radiance, antiradiance,
			cDiffuse, cSpecular, exponent);
		
		outputDiff += radiance - antiradiance;
		outputRadiance += radiance;
		outputAntiradiance += antiradiance;
	}

	outputDiff.w = 1.f;
	outputRadiance.w = 1.f;
	outputAntiradiance.w = 1.f;
}

void ProcessLight(in int avpl_id, in vec3 pos, in vec3 norm, out vec4 radiance, out vec4 antiradiance,
	in vec4 diffuse, in vec4 specular, in float exponent)
{
	const vec3 avpl_pos =	vec3(texelFetch(samplerLightBuffer, avpl_id * SIZE_LIGHT + 2));
	const vec3 avpl_norm =	vec3(	texelFetch(samplerLightBuffer, avpl_id * SIZE_LIGHT + 3));
	const vec3 avpl_to_pos = normalize(pos - avpl_pos);
	
	vec4 rad = vec4(0.f);
	vec4 antirad = vec4(0.f);
	AccessAvplAtlas(avpl_to_pos, rad, antirad, avpl_id);
		
	if(dot(norm, -avpl_to_pos) <= 0.f + EPSILON)
		antirad = vec4(0.f);

	if(dot(avpl_norm, avpl_to_pos) >= 0.f - EPSILON)
		antirad = vec4(0.f);

	const float dist = length(pos - avpl_pos);
	const float cos_theta_pos_to_avpl = clamp(dot(norm, -avpl_to_pos), 0.f, 1.f);
	
	const float G = cos_theta_pos_to_avpl / (dist * dist);
	const float G_rad = uConfig.ClampGeoTerm == 1 ? clamp(G, 0, uConfig.GeoTermLimitRadiance) : G;
	const float G_antirad = uConfig.ClampGeoTerm == 1 ? clamp(G, 0, uConfig.GeoTermLimitAntiradiance) : G;
	
	vec4 BRDF = f_r(avpl_pos, pos, uCamera.vPositionWS, norm, diffuse, specular, exponent);
	
	antiradiance = BRDF * G_antirad * antirad;
	radiance = BRDF * G_rad * rad;
}

void AccessAvplAtlas(in vec3 direction, inout vec4 radiance, inout vec4 antiradiance, int avpl_id)
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
	
	vec4 sam = texture2D(samplerLightAtlas, (1.f/float(uAtlasInfo.dim_atlas)) * texel_global);
	
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
	return vec4(d.x + s.x, d.y + s.y, d.z + s.z, 1.f);
}
