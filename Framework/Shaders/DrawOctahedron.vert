#version 420

layout(std140) uniform;

layout(location = 0) in vec4 position;

smooth out vec3 positionWS;

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

	vec4 temp = (uTransform.M * position);
	temp = temp / temp.w;
	positionWS = temp.xyz;
}