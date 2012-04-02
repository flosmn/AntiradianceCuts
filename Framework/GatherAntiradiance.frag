#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.3183
#define PI 3.14159

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

uniform config
{
	float GeoTermLimit;
	float BlurSigma;
	float BlurK;
	int UseAntiradiance;
	int DrawAntiradiance;
	int nPaths;
} uConfig;

uniform light
{
	vec4 vPositionWS;
	vec4 vOrientationWS;
	vec4 SurfaceAlbedo;
	vec4 Flux;
	vec4 Antiflux;
	vec4 vAntiPosWS;
	vec4 vIncLightDirWS;
	vec4 DebugColor;
	mat4 mView;
	mat4 mProj;
} uLight;

out vec4 outputColor;

layout(binding=0) uniform sampler2D samplerPositionWS;
layout(binding=1) uniform sampler2D samplerNormalWS;

float G(vec3 p1, vec3 n1, vec3 p2, vec3 n3);
float G_CLAMP(vec3 p1, vec3 n1, vec3 p2, vec3 n3);

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;

	vec3 vPositionWS = texture2D(samplerPositionWS, coord).xyz;
	vec3 vNormalWS = normalize(texture2D(samplerNormalWS, coord).xyz);
	
	vec3 vLightDir = normalize(uLight.vAntiPosWS.xyz - vPositionWS);
	vec3 vSrcLightDir = normalize(uLight.vPositionWS.xyz - vPositionWS);
	vec3 vAntiRadDir = normalize(uLight.vIncLightDirWS.xyz);
	
	float theta = acos(clamp(dot(vAntiRadDir, - vLightDir), 0, 1));
	const float sigma = uConfig.BlurSigma;
	float blur = uConfig.BlurK / (sqrt(2*PI)*sigma) * exp(-(theta*theta)/(2*sigma*sigma));
	
	const float dist = length(vPositionWS - uLight.vAntiPosWS.xyz);
	float G = 1 / (dist * dist) * clamp(dot(vSrcLightDir, vNormalWS), 0, 1);
	vec4 A_in = uLight.Antiflux;

	vec4 AntiIrradiance = 1 / (32 * PI) * A_in * blur * G;

	//AntiIrradiance = max(vec4(0.f, 0.f, 0.f, 0.f), AntiIrradiance);

	outputColor = AntiIrradiance;
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