#version 420

out vec4 outputColor;

uniform vec3 radiance;

void main()
{
	outputColor = vec4(radiance, 1.f);
}
