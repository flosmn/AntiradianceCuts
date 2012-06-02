#version 420

layout(std140) uniform;

smooth in vec2 texCoord;

out vec4 outputColor;

uniform sampler2D textureSampler;

uniform postprocess
{
	float one_over_gamma;
	float exposure;
} uPostProcess;

void main()
{
	vec4 color = texture2D(textureSampler, texCoord);
	color = 1 - exp(-uPostProcess.exposure * color);
	color.r = pow(color.r, uPostProcess.one_over_gamma);
	color.g = pow(color.g, uPostProcess.one_over_gamma);
	color.b = pow(color.b, uPostProcess.one_over_gamma);
	
	outputColor = color;
}
