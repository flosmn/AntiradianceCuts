#version 420

layout(std140) uniform;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;

smooth out vec3 positionVS;

void main()
{
	vec4 pos = vec4(position.xyz, 1.f);
	gl_Position = uTransform.MVP * pos;
	positionVS = vec3(uTransform.V * uTransform.M * pos);
}