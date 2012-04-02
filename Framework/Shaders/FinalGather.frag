#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.3183
#define PI 3.14159

uniform config
{
	float GeoTermLimit;
	float BlurSigma;
	float BlurK;
	int UseAntiradiance;
	int DrawAntiradiance;
	int nPaths;
} uConfig;

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

out vec4 outputColor;

layout(binding=0) uniform sampler2D samplerRadiance;
layout(binding=1) uniform sampler2D samplerAntiRadiance;
layout(binding=2) uniform sampler2D samplerMaterial;

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;

	vec4 R = texture2D(samplerRadiance, coord);
	vec4 A = texture2D(samplerAntiRadiance, coord);
	vec4 cAlbedo = texture2D(samplerMaterial, coord);
	
	vec4 temp = R - A;
	temp.r = max(temp.r, 0.f);
	temp.g = max(temp.g, 0.f);
	temp.b = max(temp.b, 0.f);
	
	outputColor = cAlbedo * temp;
	outputColor.w = 1.0f;
}

