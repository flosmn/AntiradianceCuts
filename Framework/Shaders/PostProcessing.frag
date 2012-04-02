#version 420

layout(std140) uniform;

out vec4 outputColor;

uniform sampler2D sampler;

uniform float numberOfLightPaths;
uniform float exposure;
uniform float gamma;
uniform float width;
uniform float height;

float IsLit(in vec4 positionWS);

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= width;
	coord.y /= height;

	vec4 color = texture2D(sampler, coord);
	outputColor = 1000.f * color / numberOfLightPaths;

	outputColor.w = 1.0f;
}
