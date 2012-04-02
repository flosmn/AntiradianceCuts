#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.3183
#define PI 3.14159

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

uniform config
{
	float GeoTermLimit;
	float BlurSigma;
	float BlurK;
	int UseAntiradiance;
	int DrawAntiradiance;
	int nPaths;
} uConfig;

uniform light
{
	vec4 vPositionWS;
	vec4 vOrientationWS;
	vec4 SurfaceAlbedo;
	vec4 Flux;
	vec4 Antiflux;
	vec4 vAntiPosWS;
	vec4 vIncLightDirWS;
	vec4 DebugColor;
	mat4 mView;
	mat4 mProj;
} uLight;

out vec4 outputColor;

layout(binding=0) uniform sampler2D samplerShadowMap;
layout(binding=1) uniform sampler2D samplerPositionWS;
layout(binding=2) uniform sampler2D samplerNormalWS;
layout(binding=3) uniform sampler2D samplerMaterial;

float IsLit(in vec3 positionWS);
float G(vec3 p1, vec3 n1, vec3 p2, vec3 n3);
float G_CLAMP(vec3 p1, vec3 n1, vec3 p2, vec3 n3);

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;

	vec3 vPositionWS = texture2D(samplerPositionWS, coord).xyz;
	vec3 vNormalWS = normalize(texture2D(samplerNormalWS, coord).xyz);
	vec4 cAlbedo = texture2D(samplerMaterial, coord);

	float V = 1.f;

	if (uConfig.UseAntiradiance == 0) {
		// shadow map shadows
		V = IsLit(vPositionWS);
	}

	// calc radiance	
	vec4 L_in = uLight.Flux * 0.25f * ONE_OVER_PI;
	float G = G_CLAMP(vPositionWS, vNormalWS, uLight.vPositionWS.xyz, uLight.vOrientationWS.xyz);
	vec4 BRDF = cAlbedo * ONE_OVER_PI;
	vec4 Irradiance = L_in * V * G;	
	vec4 AntiIrradiance = vec4(0.f);
	
	if (uConfig.UseAntiradiance == 1) {
		// calc Antiradiance
		vec3 vLightDir = normalize(uLight.vPositionWS.xyz - vPositionWS);
		vec3 vAntiRadDir = normalize(uLight.vIncLightDirWS.xyz);
				
		float cos_theta = clamp(dot(vAntiRadDir, - vLightDir), 0, 1);
		float blur = (uConfig.CosBlurFactor + 1) * 0.5 * ONE_OVER_PI * pow(cos_theta, uConfig.CosBlurFactor);
		
		float G_A = dot(vNormalWS, vLightDir) * (1.f / length(vPositionWS - uLight.vAntiPosWS.xyz)); 
		vec4 A_in = uLight.Antiflux;

		AntiIrradiance = A_in * blur * G_A;
	}
	
	outputColor = BRDF * (Irradiance - AntiIrradiance);
	outputColor.w = 1.0f;

	if (uConfig.DrawAntiradiance == 1) {
		outputColor = AntiIrradiance;
		outputColor.w = 1.0f;
	}
}

float G_CLAMP(in vec3 p1, in vec3 n1, in vec3 p2, in vec3 n2)
{
	return clamp(G(p1, n1, p2, n2), 0, uConfig.GeoTermLimit);
}

float G(in vec3 p1, in vec3 n1, in vec3 p2, in vec3 n2)
{
	vec3 n_1 = normalize(n1);
	vec3 n_2 = normalize(n2);
	vec3 w = normalize(p2 - p1);

	float cos_theta_1 = clamp(dot(n_1, w), 0, 1);
	float cos_theta_2 = clamp(dot(n_2, -w), 0, 1);

	float dist = length(p2 - p1);
	
	return (cos_theta_1 * cos_theta_2) / (dist * dist);
}

float IsLit(in vec3 position)
{
	float lit = 0.0f;
	
	float zNear = 0.1f;
	float zFar = 100.0f;
	float zBias = 0.0f;
	
	vec4 positionLS = uLight.mView * vec4(position, 1.f);
	positionLS = positionLS / positionLS.w;
	
	float depthLS = positionLS.z;
		
	if(depthLS <= 0)
	{
		// paraboloid projection
		float len = length(positionLS.xyz);
		positionLS = positionLS/len;
		positionLS.z = positionLS.z - 1.0f;
		positionLS.x = positionLS.x / positionLS.z;
		positionLS.y = positionLS.y / positionLS.z;
		positionLS.z = (len - zNear)/(zFar-zNear) + zBias;
		positionLS.w = 1.0f;
	
		vec3 texCoord = positionLS.xyz * 0.5f + 0.5f;
		float depthSM = texture2D(samplerShadowMap, texCoord.xy).r;
		lit = (depthSM < texCoord.z) ? 0.0f : 1.0f;
	}
	else
	{
		lit = 0.0f;
	}
	return lit;
}