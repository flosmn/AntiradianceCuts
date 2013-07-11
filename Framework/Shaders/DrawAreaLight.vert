#version 420

layout(std140) uniform;

layout(location = 0) in vec3 position;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;

void main()
{
	gl_Position = uTransform.MVP * vec4(position, 1.f);
}
