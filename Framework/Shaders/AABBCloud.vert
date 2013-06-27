#version 420

layout(std140) uniform;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_instMin;
layout(location = 2) in vec3 in_instMax;

void main()
{
	const vec3 min = in_instMin;
	const vec3 max = in_instMax;

	vec3 pos = in_position * vec3(0.998f) + vec3(0.001f);
	pos = pos * (max - min) + min;
	gl_Position = uTransform.MVP * vec4(pos, 1.f);
}
