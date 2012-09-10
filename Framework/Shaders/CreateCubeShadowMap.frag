#version 420

layout(std140) uniform;

smooth in vec3 positionVS;

layout(location = 0) out vec4 outputColor0;

void main()
{	
	const float depth = length(positionVS) / 500.f;
	outputColor0 = vec4(depth, depth, depth, 1.f);
}