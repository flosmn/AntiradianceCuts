#version 420

layout(std140) uniform;

#define ONE_OVER_PI 0.31831
#define M_PI 3.14159

uniform camera
{
	vec3 vPositionWS;
	int width;
	int height;
} uCamera;

uniform config
{
	float GeoTermLimitRadiance;
	float GeoTermLimitAntiradiance;
	float AntiradFilterK;
	float AntiradFilterGaussFactor;
	int ClampGeoTerm;
	int AntiradFilterMode;	
	int nPaths;
	int padd;
} uConfig;

uniform light
{
	mat4 ViewMatrix;
	mat4 ProjectionMatrix;
	vec4 L;	//Intensity;
	vec4 A;	//Antiintensity;
	vec4 pos;	// Position
	vec4 norm;	//Orientation;
	vec4 w_A;	//AntiintensityDirection;
	vec4 debugColor;
	float AngleFactor;
	float Bounce;
	float materialIndex;
	float padd0;
} uLight;

out vec4 outputColor;

layout(binding=0) uniform sampler2D samplerShadowMap;
layout(binding=1) uniform sampler2D samplerPositionWS;
layout(binding=2) uniform sampler2D samplerNormalWS;
layout(binding=3) uniform samplerBuffer samplerMaterialBuffer;

float IsLit(in vec3 positionWS);
float G(vec3 p1, vec3 n1, vec3 p2, vec3 n3);
float G_CLAMP(vec3 p1, vec3 n1, vec3 p2, vec3 n3);

vec4 f_r(in vec3 w_i, in vec3 w_o, in vec3 n, in vec4 diffuse, in vec4 specular, in float exponent);
vec4 f_r(in vec3 from, in vec3 over, in vec3 to, in vec3 n, in vec4 diffuse, in vec4 specular, in float exponent);

vec4 f_r(in vec3 from, in vec3 over, in vec3 to, in vec3 n, in vec4 diffuse, in vec4 specular, in float exponent)
{
	const vec3 w_i = normalize(from - over);
	const vec3 w_o = normalize(to - over);
	return f_r(w_i, w_o, n, diffuse, specular, exponent);
}

vec4 f_r(in vec3 w_i, in vec3 w_o, in vec3 n, in vec4 diffuse, in vec4 specular, in float exponent)
{
	const vec4 d = ONE_OVER_PI * diffuse;
	const float cos_theta = max(0.f, dot(reflect(-w_i, n), w_o));
	const vec4 s = max(0.5f * ONE_OVER_PI * (exponent+2.f) * pow(cos_theta, exponent) * specular, 0.f);
	return vec4(d.x + s.x, d.y + s.y, d.z + s.z, 1.f);
}

void main()
{
	vec2 coord = gl_FragCoord.xy;
	coord.x /= uCamera.width;
	coord.y /= uCamera.height;
	
	vec3 vPositionWS = texture2D(samplerPositionWS, coord).xyz;
	vec3 vNormalWS = normalize(texture2D(samplerNormalWS, coord).xyz);
	
	float V = IsLit(vPositionWS);
	
	int sizeMaterial = 4;
	int materialIndex = int(texture2D(samplerNormalWS, coord).w);
	const vec4 cDiffuse =	texelFetch(samplerMaterialBuffer, materialIndex * sizeMaterial + 1);
	const vec4 cSpecular =	texelFetch(samplerMaterialBuffer, materialIndex * sizeMaterial + 2);
	const float exponent =	texelFetch(samplerMaterialBuffer, materialIndex * sizeMaterial + 3).r;

	const int lightMaterialIndex = int(uLight.materialIndex);
	const vec4 cEmissiveLight	=	texelFetch(samplerMaterialBuffer, lightMaterialIndex * sizeMaterial + 0);
	const vec4 cDiffuseLight	=	texelFetch(samplerMaterialBuffer, lightMaterialIndex * sizeMaterial + 1);
	const vec4 cSpecularLight	=	texelFetch(samplerMaterialBuffer, lightMaterialIndex * sizeMaterial + 2);
	const float exponentLight	=	texelFetch(samplerMaterialBuffer, lightMaterialIndex * sizeMaterial + 3).r;
	
	const vec3 direction = normalize(vPositionWS - vec3(uLight.pos));
			
	vec4 BRDF_light = f_r(-vec3(uLight.w_A), direction, vec3(uLight.norm), cDiffuseLight, cSpecularLight, exponentLight);
	// check for light source AVPL
	if(length(vec3(uLight.w_A)) == 0.f)
		BRDF_light = vec4(1.f);
		 
	vec4 BRDF = f_r(vec3(uLight.pos), vPositionWS, uCamera.vPositionWS, vNormalWS, cDiffuse, cSpecular, exponent);

	// calc radiance
	vec4 L = uLight.L;
	float G = G_CLAMP(vPositionWS, vNormalWS, uLight.pos.xyz, uLight.norm.xyz);
	vec4 Irradiance = V * L * BRDF_light * G * BRDF;
	
	outputColor = Irradiance;
	outputColor.w = 1.0f;
}

float G_CLAMP(in vec3 p1, in vec3 n1, in vec3 p2, in vec3 n2)
{
	float g = G(p1, n1, p2, n2);
	return uConfig.ClampGeoTerm == 1 ? clamp(g, 0, uConfig.GeoTermLimitRadiance) : g;
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
	
	float zNear = 0.01f;
	float zFar =  10000.0f;
	float zBias = 0.0f;
	
	vec4 positionLS = uLight.ViewMatrix * vec4(position, 1.f);
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