#version 420

layout(std140) uniform;

layout(location = 0) in vec4 position;

void main()
{
	gl_Position = position;
}