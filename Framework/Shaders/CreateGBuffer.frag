#version 420

layout(std140) uniform;

uniform material
{
	vec4 diffuseMaterial;
} uMaterial;

smooth in vec4 normalWS;
smooth in vec4 positionWS;

layout(location = 0) out vec4 outputPositionWS;
layout(location = 1) out vec4 outputNormalWS;
layout(location = 2) out vec4 outputMaterial;

void main()
{
	outputPositionWS = positionWS;
	outputNormalWS = vec4(normalize(normalWS.xyz), 1.f);
	outputMaterial = uMaterial.diffuseMaterial;
}