#version 420

layout(std140) uniform;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in int material;

smooth out vec3 normalWS;
smooth out vec3 positionWS;
flat out int materialIndex;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;

void main()
{
	vec4 pos = vec4(position, 1.f);
	gl_Position = uTransform.MVP * pos;
		
	positionWS = vec3(uTransform.M * pos);
	normalWS = vec3(uTransform.itM * vec4(normal, 1.0f));
	materialIndex = material;
}
