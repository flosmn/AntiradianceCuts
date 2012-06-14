#version 420

out vec4 outputColor;

smooth in vec4 color;

void main()
{
	outputColor = vec4(1.f, 1.f, 1.f, 1.f);
	outputColor.w = 1.f;
}
