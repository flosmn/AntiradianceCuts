#version 420

layout(std140) uniform;

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

layout(location = 0) out vec4 outputColor;

layout(binding=0) uniform sampler2D samplerResult;
layout(binding=1) uniform sampler2D samplerReference;

float Luminance(in vec4 c)
{
	return	0.2126f * max(c.r, 0.f) + 
			0.7152f * max(c.g, 0.f) + 
			0.0722f * max(c.b, 0.f);
}

float Error(in vec4 c1, in vec4 c2)
{
	vec3 c = vec3(abs(c1 - c2));
	return 1.f/3.f * (c.x + c.y + c.z);
	//return Luminance(abs(c1 - c2));
}

vec3 hue_colormap(const float v, const float range_min, const float range_max)
{
	const float H = clamp((range_max - v)/(range_max - range_min), 0.f, 1.f) * 4.f;
	const float X = clamp(1.f - abs(mod(H, 2.f) - 1.f), 0.f, 1.f); 

	if(H < 1.f)
		return vec3(1.f, X, 0.f);
	else if(H < 2.f)
		return vec3(X, 1.f, 0.f);
	else if(H < 3.f)
		return vec3(0.f, 1.f, X);
	else
		return vec3(0.f, X, 1.f);
}

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;
	
	vec4 cResult = texture2D(samplerResult, coord);
	cResult = max(cResult, vec4(0.f, 0.f, 0.f, 0.f));
	
	vec4 cReference = texture2D(samplerReference, coord);
		
	if(length(cReference.xyz) == 0.f && length(cResult.xyz) == 0.f)
		outputColor = vec4(0.f, 0.f, 0.f, 1.f);
	else
		outputColor = vec4(hue_colormap(Error(cResult, cReference), 0.f, 0.1f), 1.f);
}