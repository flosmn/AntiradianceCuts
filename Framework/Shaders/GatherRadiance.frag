#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.3183
#define PI 3.14159

uniform info_block
{
	int numLights;
	float debugColorR;
	float debugColorG;
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

out vec4 outputColor;

layout(binding=0) uniform sampler2D samplerPositionWS;
layout(binding=1) uniform sampler2D samplerNormalWS;
layout(binding=2) uniform samplerBuffer samplerLightBuffer;

float G(vec3 p1, vec3 n1, vec3 p2, vec3 n3);
float G_CLAMP(vec3 p1, vec3 n1, vec3 p2, vec3 n3);

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;
	
	vec3 vPositionWS = texture2D(samplerPositionWS, coord).xyz;
	vec3 vNormalWS = normalize(texture2D(samplerNormalWS, coord).xyz);

	int size = 4 * 8;
	outputColor = vec4(0);
	for(int i = 0; i < uInfo.numLights; ++i)
	{		
		const vec3 vLightPosWS = vec3(	texelFetch(samplerLightBuffer, i * size + 0).r,
										texelFetch(samplerLightBuffer, i * size + 1).r,
										texelFetch(samplerLightBuffer, i * size + 2).r);

		const vec3 vLightOrientationWS = vec3(	texelFetch(samplerLightBuffer, i * size + 4).r,
												texelFetch(samplerLightBuffer, i * size + 5).r,
												texelFetch(samplerLightBuffer, i * size + 6).r);

		const vec4 vLightContrib	= vec4(	texelFetch(samplerLightBuffer, i * size + 8).r,
											texelFetch(samplerLightBuffer, i * size + 9).r,
											texelFetch(samplerLightBuffer, i * size + 10).r,
											texelFetch(samplerLightBuffer, i * size + 11).r);

		// calc radiance
		vec4 I = vLightContrib;
		float G = G_CLAMP(vPositionWS, vNormalWS, vLightPosWS, vLightOrientationWS);
		vec4 Irradiance = I * G;	
	
		outputColor += Irradiance;
	}
	outputColor.w = 1.0f;
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