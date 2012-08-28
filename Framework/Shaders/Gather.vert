#version 420

layout(std140) uniform;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;

layout(location = 0) in vec4 position;

out vec3 direction;

void main()
{
	direction = transpose(mat3(uTransform.V)) * normalize(vec3(position));
	gl_Position = position;
}