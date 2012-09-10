#version 420

layout(std140) uniform;

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

in vec3 direction;

layout(location = 0) out vec4 outputColor;

layout(binding=0) uniform sampler2D samplerIrradiance;

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;
	
	vec4 cIrradiance = texture2D(samplerIrradiance, coord);
	cIrradiance = max(cIrradiance, vec4(0.f, 0.f, 0.f, 0.f));
		
	outputColor = cIrradiance;
	outputColor.w = 1.0f;
}