#version 420

layout(std140) uniform;

uniform transform
{
	mat4 M;
	mat4 V;
	mat4 itM;
	mat4 MVP;
} uTransform;

layout(location = 0) in vec3 position;

out vec3 direction;

void main()
{
	direction = transpose(mat3(uTransform.V)) * normalize(position);
	gl_Position = vec4(position, 1.f);
}
