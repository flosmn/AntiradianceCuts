#version 420

layout(std140) uniform;

layout(location = 0) in vec4 position;

smooth out vec3 positionVS;
smooth out float depthVS;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;


void main()
{
	float zNear = 0.1f;
	float zFar =  100.0f;
	float zBias = 0.0f;

	vec4 positionTemp = uTransform.V * uTransform.M * position;
	positionTemp = positionTemp / positionTemp.w;
	
	float len = length(positionTemp.xyz);
	positionTemp = positionTemp/len;
	
	depthVS = positionTemp.z;
	
	positionTemp.z = positionTemp.z - 1.0f;
	positionTemp.x = positionTemp.x / positionTemp.z;
	positionTemp.y = positionTemp.y / positionTemp.z;
	positionTemp.z = (len - zNear)/(zFar-zNear) + zBias;
	positionTemp.w = 1.0f;
	gl_Position = positionTemp;
}
