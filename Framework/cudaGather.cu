#include "cudaGather.h"
#include "CudaResources/cudaUtil.hpp"

#include "Utils/stream.h"

using namespace cuda;

inline __device__ float G(float3 const& p1, float3 const& n1, float3 const& p2, float3 const& n2)
{
	float3 n_1 = normalize(n1);
	float3 n_2 = normalize(n2);
	float3 w = normalize(p2 - p1);

	float cos_theta_1 = clamp(dot(n_1, w), 0.f, 1.f);
	float cos_theta_2 = clamp(dot(n_2, -w), 0.f, 1.f);

	float dist = length(p2 - p1);
	
	return (cos_theta_1 * cos_theta_2) / (dist * dist);
}

inline __device__ float4 f_r(float3 const& w_i, float3 const& w_o, float3 const& n, MATERIAL const& mat)
{
	const float4 d = ONE_OVER_PI * make_float4(mat.diffuse);
	const float cos_theta = max(0.f, dot(reflect(-w_i, n), w_o));
	const float4 s = fmaxf(0.5f * ONE_OVER_PI * (mat.exponent+2.f)
		* pow(cos_theta, mat.exponent) * make_float4(mat.specular), 0.f);
	return d; //vec4(d.x + s.x, d.y + s.y, d.z + s.z, 1.f);
}
inline __device__ float4 f_r(float3 const& from, float3 const& over, float3 const& to, float3 const& n, MATERIAL const& mat)
{
	const float3 w_i = normalize(from - over);
	const float3 w_o = normalize(to - over);
	return f_r(w_i, w_o, n, mat);
}

__global__ void kernel(
		cudaSurfaceObject_t outResult,
		cudaSurfaceObject_t outRadiance,
		cudaSurfaceObject_t outAntiradiance,
		cudaSurfaceObject_t inPositions,
		cudaSurfaceObject_t inNormals,
		NEW_AVPL* inAvpls,
		int numAvpls,
		MATERIAL* inMaterials,
		float3 cameraPosition,
		int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
		
	float4 data;
	surf2Dread(&data, inPositions, x * sizeof(float4), y);
	float3 position = make_float3(data);
	surf2Dread(&data, inNormals, x * sizeof(float4), y);
	float3 normal= make_float3(data);
	int materialIndex = int(data.w);

	MATERIAL const& mat = inMaterials[materialIndex];
	
	float4 outRes		= make_float4(0.f);
	float4 outRad		= make_float4(0.f);
	float4 outAntirad	= make_float4(0.f);
		
	for(int i = 0; i < numAvpls; ++i)
	{		
		NEW_AVPL& avpl = inAvpls[i];
		MATERIAL& mat_avpl = inMaterials[avpl.materialIndex];

		float4 rad = make_float4(0.0f, 0.f, 0.f, 0.f);
		float4 antirad = make_float4(0.f);
	
		float3 direction = normalize(position - make_float3(avpl.pos));
				
		float4 brdf_light = f_r(make_float3(-avpl.w), direction, make_float3(avpl.norm), mat_avpl);

		// check for light source AVPL
		if(length(make_float3(avpl.w)) == 0.f)
			brdf_light = make_float4(1.f);
			 
		float4 brdf = f_r(make_float3(avpl.pos), position, cameraPosition, normal, mat);

		rad = make_float4(avpl.L) * brdf_light * G(position, normal, make_float3(avpl.pos), make_float3(avpl.norm)) * brdf;
		//antirad = brdf * GetAntiradiance(A, p, vPositionWS, n, vNormalWS, w, coneFactor);
		
		outRes += rad - antirad;
		outRad += rad;
		outAntirad += antirad;
	}
	
	outRes.w = 1.f;
	outRad.w = 1.f;
	outAntirad.w = 1.f;

	surf2Dwrite(outRes, outResult, x * sizeof(float4), y);
	//surf2Dwrite(outRad, outRadiance, x * sizeof(float4), y);
	//surf2Dwrite(outAntirad, outAntiradiance, x * sizeof(float4), y);
}

CudaGather::CudaGather(int width, int height, 
	GLuint glPositonTexture, GLuint glNormalTexture,
	GLuint glResultOutputTexture,
	GLuint glRadianceOutputTexture,
	GLuint glAntiradianceOutputTexture,
	std::vector<MATERIAL> const& materials)
	: m_width(width), m_height(height)
{
	m_positionResource.reset(new CudaGraphicsResource(glPositonTexture, GL_TEXTURE_2D, 
		cudaGraphicsRegisterFlagsNone));
	m_normalResource.reset(new CudaGraphicsResource(glNormalTexture, GL_TEXTURE_2D, 
		cudaGraphicsRegisterFlagsNone));

	m_antiradianceOutputResource.reset(new CudaGraphicsResource(glAntiradianceOutputTexture,
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	m_radianceOutputResource.reset(new CudaGraphicsResource(glRadianceOutputTexture,
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	m_resultOutputResource.reset(new CudaGraphicsResource(glResultOutputTexture,
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

	m_materials.reset(new CudaBuffer<MATERIAL>(materials));
}

CudaGather::~CudaGather()
{
}

void CudaGather::run(std::vector<AVPL> const& avpls, glm::vec3 const& cameraPosition)
{
	CudaGraphicsResourceMappedArray positionsMapped(m_positionResource.get());
	CudaGraphicsResourceMappedArray normalsMapped(m_normalResource.get());
	CudaGraphicsResourceMappedArray radianceOutMapped(m_radianceOutputResource.get());
	CudaGraphicsResourceMappedArray antiradianceOutMapped(m_antiradianceOutputResource.get());
	CudaGraphicsResourceMappedArray resultOutMapped(m_resultOutputResource.get());

	std::vector<NEW_AVPL> new_avpls(avpls.size());
	for (int i = 0; i < avpls.size(); ++i)
	{
		NEW_AVPL newavpl;
		newavpl.L = glm::vec4(avpls[i].m_Radiance, 1.f);
		newavpl.A = glm::vec4(avpls[i].m_Antiradiance, 1.f);
		newavpl.pos = avpls[i].m_Position;
		newavpl.materialIndex = avpls[i].m_MaterialIndex;
		newavpl.norm = avpls[i].m_Orientation;
		newavpl.angleFactor = avpls[i].m_ConeAngle;
		newavpl.w = avpls[i].m_Direction;
		newavpl.bounce = avpls[i].m_Bounce;
		new_avpls[i] = newavpl;
	}

	m_avpls.reset(new CudaBuffer<NEW_AVPL>(new_avpls));

	// Invoke kernel
	dim3 dimBlock(32, 32);
	dim3 dimGrid((m_width  + dimBlock.x - 1) / dimBlock.x,
		(m_height + dimBlock.y - 1) / dimBlock.y);
	kernel<<<dimGrid, dimBlock>>>(
			resultOutMapped.getCudaSurfaceObject(), 
			radianceOutMapped.getCudaSurfaceObject(), 
			antiradianceOutMapped.getCudaSurfaceObject(), 
			positionsMapped.getCudaSurfaceObject(), 
			normalsMapped.getCudaSurfaceObject(), 
			m_avpls->getDevicePtr(), avpls.size(),
			m_materials->getDevicePtr(),
			make_float3(cameraPosition),
			m_width, m_height);

	//std::cout << "avpl.L: " << avpls[0].m_Radiance << std::endl;
}

