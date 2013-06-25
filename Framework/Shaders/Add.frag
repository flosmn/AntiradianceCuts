#version 420

layout(std140) uniform;

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

out vec4 outputColor;

layout(binding=0) uniform sampler2D sampler1;
layout(binding=1) uniform sampler2D sampler2;

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;

	vec4 c1 = texture2D(sampler1, coord);
	vec4 c2 = texture2D(sampler2, coord);
		
	outputColor = c1 + c2;
	outputColor.w = 1.0f;
}

