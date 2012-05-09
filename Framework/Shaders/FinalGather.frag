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
		
	outputColor = R-A;
	outputColor.w = 1.0f;
}

