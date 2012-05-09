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

out vec4 outputColor;

layout(binding=0) uniform sampler2D samplerIrradiance;
layout(binding=1) uniform sampler2D samplerMaterial;

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;

	vec4 cIrradiance = texture2D(samplerIrradiance, coord);
	vec4 cAlbedo = texture2D(samplerMaterial, coord);
		
	outputColor = cIrradiance;
	outputColor.w = 1.0f;
}

