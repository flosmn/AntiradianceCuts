#version 420

uniform arealight
{
	vec4 radiance;
} uAreaLight;

out vec4 outputColor;

void main()
{
	outputColor = uAreaLight.radiance;
}