#version 420

layout(std140) uniform;

in vec3 normalWS;
in vec3 positionWS;
flat in int materialIndex;

layout(location = 0) out vec4 outputPositionWS;
layout(location = 1) out vec4 outputNormalWS;

void main()
{
	outputPositionWS = vec4(positionWS, 1.f);
	outputNormalWS = vec4(normalize(normalWS), float(materialIndex));
}
