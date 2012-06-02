#version 420

out vec4 outputColor;

smooth in vec4 color;

void main()
{
	outputColor = 0.01 * color;
	outputColor.w = 1.f;
}
