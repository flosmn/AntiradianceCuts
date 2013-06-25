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

void main()
{
	gl_Position = uTransform.MVP * vec4(0.1f * in_position, 1.f);
}
