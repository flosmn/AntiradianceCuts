#version 420

layout(std140) uniform;

uniform mat4 V;

layout(location = 0) in vec4 position;

void main()
{
		gl_Position = position;
}