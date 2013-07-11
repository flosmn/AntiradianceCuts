#version 420

layout(std140) uniform;

smooth in vec2 texCoord;

out vec4 outputColor;

uniform sampler2D textureSampler;

uniform float gamma;
uniform float exposure;

void main()
{
	vec4 color = texture2D(textureSampler, texCoord);
	//color = 1 - exp(-uPostProcess.exposure * color);
	color.r = pow(color.r, 1.f / gamma);
	color.g = pow(color.g, 1.f / gamma);
	color.b = pow(color.b, 1.f / gamma);
	
	outputColor = color;
}
