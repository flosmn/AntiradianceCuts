#version 420

uniform vec3 intensity;

out vec4 outputColor;

void main()
{
	outputColor = vec4(intensity, 1.0f);
}