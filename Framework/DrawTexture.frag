#version 420

layout(std140) uniform;

smooth in vec2 texCoord;

out vec4 outputColor;

uniform sampler2D textureSampler;

void main()
{
	outputColor = texture2D(textureSampler, texCoord);
}