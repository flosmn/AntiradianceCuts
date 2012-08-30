#version 420

layout(std140) uniform;

layout(location = 0) in vec4 position;
layout(location = 1) in vec3 normal;

smooth out vec4 normalWS;
smooth out vec4 positionWS;
flat out float materialIndex;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;

void main()
{
	vec4 pos = vec4(position.xyz, 1.f);
	materialIndex = position.w;

	gl_Position = uTransform.MVP * pos;
		
	positionWS = uTransform.M * pos;
	normalWS = uTransform.itM * vec4(normal, 1.0f);
}