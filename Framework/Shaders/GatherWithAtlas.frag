#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.3183
#define PI 3.14159

vec2 GetTexCoordForDirection(vec3 d)
{
	// projection onto octahedron
	d /= dot( vec3(1.f), abs(d) );
	
	// out-folding of the downward faces
	if ( d.y < 0.0f )
	{
		float x = (1-abs(d.z)) * sign(d.x);
		float z = (1-abs(d.x)) * sign(d.z);
		d.x = x;
		d.z = z;
	}
	// mapping to [0;1]ˆ2 texture space
	d.xz = d.xz * 0.5 + 0.5;
	
	return d.xz;
}

uniform atlas_info
{
	int dim_atlas;
	int dim_tile;
} uAtlasInfo;

uniform info_block
{
	int numLights;
	int drawLightingOfLight;
	bool filterAVPLAtlas;
	float debugColorB;
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

layout(binding=0) uniform sampler2D samplerPositionWS;
layout(binding=1) uniform sampler2D samplerNormalWS;
layout(binding=2) uniform samplerBuffer samplerLightPositionBuffer;
layout(binding=3) uniform sampler2D samplerLightAtlas;

float G(vec3 p1, vec3 n1, vec3 p2, vec3 n3);
float G_CLAMP(vec3 p1, vec3 n1, vec3 p2, vec3 n3);

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;
	
	const float N = float(uConfig.N);
	const float PI_OVER_N = PI / N;
	const float K = (PI * (1 - cos(PI_OVER_N))) / (PI - N * sin(PI_OVER_N));
	
	vec3 vPositionWS = texture2D(samplerPositionWS, coord).xyz;
	vec3 vNormalWS = normalize(texture2D(samplerNormalWS, coord).xyz);

	outputDiff = vec4(0.f);
	outputRadiance = vec4(0.f);
	outputAntiradiance = vec4(0.f);

	int columns = uAtlasInfo.dim_atlas / uAtlasInfo.dim_tile;

	if(uInfo.drawLightingOfLight == -1)
	{
		int size = 4;
		for(int i = 0; i < uInfo.numLights; ++i)
		{		
			vec4 radiance = vec4(0.f);
			vec4 antiradiance = vec4(0.f);
			vec4 diff = vec4(0.f);
			
			const vec3 p = vec3(
				texelFetch(samplerLightPositionBuffer, i * size + 0).r,
				texelFetch(samplerLightPositionBuffer, i * size + 1).r,
				texelFetch(samplerLightPositionBuffer, i * size + 2).r);

			const vec3 direction = normalize(vPositionWS - p);

			vec2 texel_local;
			if(uInfo.filterAVPLAtlas)
			{
				texel_local = (uAtlasInfo.dim_tile-2) * GetTexCoordForDirection(direction) + vec2(1.f, 1.f) + 0.5f / uAtlasInfo.dim_atlas;
			}
			else
			{
				texel_local = (uAtlasInfo.dim_tile) * GetTexCoordForDirection(direction) + 0.5f / uAtlasInfo.dim_atlas;
			}

			const int row = i / columns;
			const int column = i % columns;
			const vec2 globalOffset = vec2(column * uAtlasInfo.dim_tile, row * uAtlasInfo.dim_tile);
			const vec2 texel_global = texel_local + globalOffset;
			
			vec4 sam = texture2D(samplerLightAtlas, (1.f/float(uAtlasInfo.dim_atlas)) * texel_global);
			
			radiance = max(vec4(0.f), sam);
			antiradiance = -min(vec4(0.f), sam);
			
			const float dist = length(vPositionWS - p);
			const float cos_theta = clamp(dot(vNormalWS, -direction), 0, 1);
			radiance = clamp(cos_theta / (dist * dist), 0, uConfig.GeoTermLimit) * radiance;
			antiradiance = (cos_theta) / (dist * dist) * antiradiance;
			
			outputDiff += radiance - antiradiance;
			outputRadiance += radiance;
			outputAntiradiance += antiradiance;
		}
	}

	else if(uInfo.drawLightingOfLight < uInfo.numLights)
	{
		vec4 radiance = vec4(0.f);
		vec4 antiradiance = vec4(0.f);
		vec4 diff = vec4(0.f);
		
		const vec3 p = vec3(
			texelFetch(samplerLightPositionBuffer, uInfo.drawLightingOfLight * 4 + 0).r,
			texelFetch(samplerLightPositionBuffer, uInfo.drawLightingOfLight * 4 + 1).r,
			texelFetch(samplerLightPositionBuffer, uInfo.drawLightingOfLight * 4 + 2).r);

		const vec3 direction = normalize(vPositionWS - p);

		vec2 texel_local;
		if(uInfo.filterAVPLAtlas)
		{
			texel_local = (uAtlasInfo.dim_tile-2) * GetTexCoordForDirection(direction) + vec2(1.f, 1.f) + 0.5f / uAtlasInfo.dim_atlas;
		}
		else
		{
			texel_local = (uAtlasInfo.dim_tile) * GetTexCoordForDirection(direction) + 0.5f / uAtlasInfo.dim_atlas;
		}

		const int row = uInfo.drawLightingOfLight / columns;
		const int column = uInfo.drawLightingOfLight % columns;
		const vec2 globalOffset = vec2(column * uAtlasInfo.dim_tile, row * uAtlasInfo.dim_tile);
		const vec2 texel_global = texel_local + globalOffset;
		
		vec4 sam = texture2D(samplerLightAtlas, (1.f/float(uAtlasInfo.dim_atlas)) * texel_global);
		
		radiance = max(vec4(0.f), sam);
		antiradiance = -min(vec4(0.f), sam);
			
		const float dist = length(vPositionWS - p);
		const float cos_theta = clamp(dot(vNormalWS, -direction), 0, 1);
		radiance = clamp(cos_theta / (dist * dist), 0, uConfig.GeoTermLimit) * radiance;
		antiradiance = (cos_theta) / (dist * dist) * antiradiance;
		
		outputDiff += radiance - antiradiance;
		outputRadiance += radiance;
		outputAntiradiance += antiradiance;
	}
	else
	{
	}

	outputDiff.w = 1.f;
	outputRadiance.w = 1.f;
	outputAntiradiance.w = 1.f;
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