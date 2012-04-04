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
	vec4 Position;
	vec4 Orientation;
	vec4 Flux;
	vec4 SrcPosition;
	vec4 SrcOrientation;
	vec4 SrcFlux;
	vec4 DebugColor;
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
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
	
	vec3 vLightDir = normalize(vPositionWS - uLight.Position.xyz);				// direction from light to point
	vec3 vAntiRadDir = normalize(uLight.Position.xyz - uLight.SrcPosition.xyz); // direction of antiradiance
	
	float theta = acos(clamp(dot(vAntiRadDir, vLightDir), 0, 1));
	const float sigma = uConfig.BlurSigma;
	float blur = uConfig.BlurK / (sqrt(2*PI)*sigma) * exp(-(theta*theta)/(2*sigma*sigma));
	
	vec4 A_in = blur * uLight.SrcFlux;	// blurred antiradiance

	//float G = G_CLAMP(vPositionWS, vNormalWS, uLight.Position.xyz, -uLight.Orientation.xyz);
	float G = G_CLAMP(vPositionWS, vNormalWS, uLight.SrcPosition.xyz, uLight.SrcOrientation.xyz);

	vec4 AntiIrradiance = A_in * G;
	
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