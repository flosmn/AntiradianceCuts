#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.3183
#define PI 3.14159

uniform info_block
{
	int numLights;
	int numClusters;
	int UseIBL;
	int filterAVPLAtlas;
	
	int lightTreeCutDepth;
	int padd0;
	int padd1;
	int padd2;
} uInfo;

uniform config
{
	float GeoTermLimitRadiance;
	float GeoTermLimitAntiradiance;
	float AntiradFilterK;
	float AntiradFilterGaussFactor;
	int ClampGeoTerm;
	int AntiradFilterMode;	
	int nPaths;
	int padd;
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
layout(binding=3) uniform samplerCube samplerCubeMap;

float G(vec3 p1, vec3 n1, vec3 p2, vec3 n3);
float G_CLAMP(vec3 p1, vec3 n1, vec3 p2, vec3 n3);

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;
		
	vec3 vPositionWS = texture2D(samplerPositionWS, coord).xyz;
	vec3 vNormalWS = normalize(texture2D(samplerNormalWS, coord).xyz);

	outputDiff = vec4(0.f);
	outputRadiance = vec4(0.f);
	outputAntiradiance = vec4(0.f);

	int size = 5; // sizeof(AVPL_BUFFER)
	for(int i = 0; i < uInfo.numLights; ++i)
	{		
		vec4 radiance = vec4(0.f);
		vec4 antiradiance = vec4(0.f);
		vec4 diff = vec4(0.f);
		
		const vec4 I = texelFetch(samplerLightBuffer, i * size + 0);
		vec4 A = texelFetch(samplerLightBuffer, i * size + 1);
		const vec3 p = vec3(texelFetch(samplerLightBuffer, i * size + 2));
		const vec3 n = vec3(texelFetch(samplerLightBuffer, i * size + 3));
		
		const vec4 temp = texelFetch(samplerLightBuffer, i * size + 4);
		const vec3 w_A = temp.xyz;		
		const float coneFactor = temp.w;
				
		// calc radiance
		if(length(I) > 0.f)
		{
			float G = G_CLAMP(vPositionWS, vNormalWS, p, n);
			vec4 Irradiance = I * G;	
			radiance = Irradiance;
		}
		
		// calc antiradiance
		if(length(A) > 0.f)
		{
			const vec3 vLightToPos = vPositionWS - p;
			const float d_xz = length(vLightToPos);
			vec3 w = normalize(vLightToPos);
									
			if(dot(w, w_A) > 0.f && dot(vNormalWS, -w) > 0.f)
			{				
				const float theta = acos(clamp(dot(w, w_A), 0, 1));
				if(theta < PI / coneFactor)
				{
					const float cos_theta_xz = clamp(dot(vNormalWS, -w), 0, 1);
					
					if(uConfig.AntiradFilterMode == 1)
					{
						A = - 2 * A * uConfig.AntiradFilterK * (coneFactor / PI * theta - 1);
					}
					if(uConfig.AntiradFilterMode == 2)
					{
						const float M = uConfig.AntiradFilterGaussFactor;
						const float s = PI / (M*coneFactor);
						A = uConfig.AntiradFilterK * ( exp(-(theta*theta)/(s*s)) - exp(-(M*M)) ) * A;
					}

					float G = cos_theta_xz / (d_xz * d_xz);
					G = uConfig.ClampGeoTerm == 1 ? clamp(G, 0, uConfig.GeoTermLimitAntiradiance) : G;
					vec4 A_in = G * A;
										
					antiradiance = A_in;
				}
			}
		}
		
		diff = radiance; // - antiradiance;

		outputDiff += diff;
		outputRadiance += radiance;
		outputAntiradiance += antiradiance;
	}
			
	outputDiff.w = 1.f;
	outputRadiance.w = 1.f;
	outputAntiradiance.w = 1.f;
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