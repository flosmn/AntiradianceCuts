#version 420

layout(std140) uniform;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;

smooth out vec4 normalWS;
smooth out vec4 positionWS;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;

void main()
{
	gl_Position = uTransform.MVP * position;
	
	positionWS = uTransform.M * position;
	normalWS = uTransform.itM * vec4(normal, 1.0f);
}