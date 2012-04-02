#version 420

uniform vec3 lightColor;

out vec4 outputColor;

void main()
{
	outputColor = vec4(lightColor, 1.0f);
}