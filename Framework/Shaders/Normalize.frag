#version 420

layout(std140) uniform;

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

uniform float normFactor;

out vec4 outputColor;

layout(binding=0) uniform sampler2D accumDiff;
layout(binding=1) uniform sampler2D accumRadiance;
layout(binding=2) uniform sampler2D accumAntiradiance;

layout(location = 0) out vec4 normDiff;
layout(location = 1) out vec4 normRadiance;
layout(location = 2) out vec4 normAntiradiance;

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;

	vec4 aDiff = texture2D(accumDiff, coord);
	vec4 aRad = texture2D(accumRadiance, coord);
	vec4 aAntirad = texture2D(accumAntiradiance, coord);
	
	normDiff = aDiff * normFactor;
	normDiff.w = 1.0f;

	normRadiance = aRad * normFactor;
	normRadiance.w = 1.0f;

	normAntiradiance = aAntirad * normFactor;
	normAntiradiance.w = 1.0f;
}
