#version 420

layout(std140) uniform;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_color;

smooth out vec4 color;

void main()
{
	gl_Position = uTransform.MVP * in_position;
	color = in_color;
}