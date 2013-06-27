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
layout(location = 1) in vec3 in_instancePosition;
layout(location = 2) in vec3 in_instanceColor;

out vec3 instancePosition;
out vec3 instanceColor;

void main()
{
	gl_Position = uTransform.MVP * vec4(in_instancePosition + 5.f * in_position, 1.f);
	instancePosition = in_instancePosition;
	instanceColor = in_instanceColor;
}
