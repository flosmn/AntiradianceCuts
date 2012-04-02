#version 420

layout(std140) uniform;

uniform config
{
	float GeoTermLimit;
	float BlurSigma;
	float BlurK;
	int UseAntiradiance;
	int DrawAntiradiance;
	int nPaths;
} uConfig;

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

out vec4 outputColor;

layout(binding=0) uniform sampler2D accumRadiance;

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;

	vec4 accumRad = texture2D(accumRadiance, coord);
			
	float p = 1.f / float(uConfig.nPaths);
	outputColor = accumRad * p;
	outputColor.w = 1.0f;
}