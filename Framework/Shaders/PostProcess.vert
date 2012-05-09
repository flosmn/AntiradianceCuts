#version 420

layout(std140) uniform;

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec3 in_texCoord;

smooth out vec2 texCoord;

void main()
{
	gl_Position = in_position;
	texCoord = in_texCoord.xy;
}