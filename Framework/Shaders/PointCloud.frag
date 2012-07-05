#version 420

out vec4 outputColor;

smooth in vec4 color;

void main()
{
	outputColor = color;
	outputColor.w = 1.f;
}
