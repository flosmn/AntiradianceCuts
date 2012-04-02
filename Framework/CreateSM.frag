#version 420

out vec4 out_color;

smooth in float depthVS;

void main()
{
	if(depthVS >= 0) discard;
}