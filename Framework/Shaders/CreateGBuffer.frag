#version 420

layout(std140) uniform;

smooth in vec4 normalWS;
smooth in vec4 positionWS;
flat in float materialIndex;

layout(location = 0) out vec4 outputPositionWS;
layout(location = 1) out vec4 outputNormalWS;

void main()
{
	outputPositionWS = positionWS;
	outputNormalWS = vec4(normalize(normalWS.xyz), materialIndex);
}