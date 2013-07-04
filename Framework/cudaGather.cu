#include "cudaGather.h"

#include "CudaResources/cudaUtil.hpp"
#include "CudaResources/cudaTimer.hpp"

#include "data_gpu.h"

#include "Utils/stream.h"

#define THREADS_X 16
#define THREADS_Y 16
#define MAX_NUM_MATERIALS 20

using namespace cuda;

struct MAT
{
	__device__ __host__ MAT(float3 const& d, float3 const& s, float e)
		: diffuse(d), specular(s), exponent(e)
	{ }

	float3 diffuse;
	float3 specular;
	float exponent;
};

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

inline __device__ float3 f_r(float3 const& w_i, float3 const& w_o, float3 const& n, MAT const& mat)
{
	const float3 d = ONE_OVER_PI * mat.diffuse;
	const float cos_theta = max(0.f, dot(reflect(-w_i, n), w_o));
	const float3 s = 0.5f * ONE_OVER_PI * (mat.exponent+2.f)
		* pow(cos_theta, mat.exponent) * mat.specular;
	return d; //vec4(d.x + s.x, d.y + s.y, d.z + s.z, 1.f);
}

inline __device__ float3 f_r(float3 const& from, float3 const& over, float3 const& to, float3 const& n, MAT const& mat)
{
	const float3 w_i = normalize(from - over);
	const float3 w_o = normalize(to - over);
	return f_r(w_i, w_o, n, mat);
}

inline __device__ float3 getRadiance(float3 const& pos, 
		float3 const& norm, MAT const& mat,
		float3 const& avpl_pos, float3 const& avpl_norm, 
		float3 const& avpl_w, float3 const& avpl_L, 
		MAT const& avpl_mat, float3 const& camPos)
{
	const float3 direction = normalize(pos - avpl_pos);
			
	float3 brdf_light = f_r(-avpl_w, direction, avpl_norm, avpl_mat);

	// check for light source AVPL
	if(length(avpl_w) == 0.f)
		brdf_light = make_float3(1.f);
		 
	const float3 brdf = f_r(avpl_pos, pos, camPos, norm, mat);

	return avpl_L * brdf_light * G(pos, norm, avpl_pos, avpl_norm) * brdf;
}

inline __device__ float3 getRadiance(float3 const& pos, float3 const& norm, MAT const& mat, MAT const& avpl_mat,
		int idx, AvplBvhNodeDataParam* dataParam, BvhParam* bvhParam, float3 const& camPos)
{
	const float3 avpl_pos = dataParam->position[bvhParam->ids[idx]];
	const float3 avpl_norm = dataParam->normal[bvhParam->ids[idx]];
	const float3 avpl_incRad = dataParam->incRadiance[bvhParam->ids[idx]];
	const float3 avpl_incDir = dataParam->incDirection[bvhParam->ids[idx]];
	return getRadiance(pos, norm, mat, avpl_pos, avpl_norm, avpl_incDir, avpl_incRad, avpl_mat, camPos);
}


// TODO: shared memory
__global__ void kernel(
		cudaSurfaceObject_t outResult,
		cudaSurfaceObject_t outRadiance,
		cudaSurfaceObject_t outAntiradiance,
		cudaSurfaceObject_t inPositions,
		cudaSurfaceObject_t inNormals,
		AvplsGpuParam* avpls,
		MaterialsGpuParam* materials,
		float3 camPos,
		int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
		
	const bool calcPixelValue = (x < width && y < height);
	
	float3 pos = make_float3(1.f);
	float3 norm = make_float3(1.f);
	int materialIndex = 0;

	if (calcPixelValue) { // fetch gbuffer
		float4 data;
		surf2Dread(&data, inPositions, x * sizeof(float4), y);
		pos = make_float3(data);
		surf2Dread(&data, inNormals, x * sizeof(float4), y);
		norm= make_float3(data);
		materialIndex = int(data.w);
	}

	float3 outRad		= make_float3(0.f);
	float3 outAntirad	= make_float3(0.f);
	
	const int numAvpls = avpls->numAvpls;
	const int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	const int chunkSize = THREADS_X * THREADS_Y;
	const int numChunks = max(numAvpls / chunkSize, 0);
	
	__shared__ float3 avpl_position[chunkSize];
	__shared__ float3 avpl_normal[chunkSize];
	__shared__ float3 avpl_incRadiance[chunkSize];
	__shared__ float3 avpl_incDirection[chunkSize];
	__shared__ int avpl_materialIndex[chunkSize];
	
	__shared__ float3 material_diffuse[MAX_NUM_MATERIALS];
	__shared__ float3 material_specular[MAX_NUM_MATERIALS];
	__shared__ float material_exponent[MAX_NUM_MATERIALS];
	if (threadId < materials->numMaterials && threadId < MAX_NUM_MATERIALS) {
		material_diffuse[threadId] = materials->diffuse[threadId];
		material_specular[threadId] = materials->specular[threadId];
		material_exponent[threadId] = materials->exponent[threadId];
	}
	syncthreads();
	
	MAT mat(material_diffuse[materialIndex],
			material_specular[materialIndex],
			material_exponent[materialIndex]);

	for (int chunk = 0; chunk < numChunks; ++chunk) 
	{
		// load chunk into shared memory
		const int index = chunkSize * chunk + threadId;
		avpl_position[threadId] = avpls->position[index];
		avpl_normal[threadId] = avpls->normal[index];
		avpl_incRadiance[threadId] = avpls->incRadiance[index];
		avpl_incDirection[threadId] = avpls->incDirection[index];
		avpl_materialIndex[threadId] = avpls->materialIndex[index];
		syncthreads();
		
		if (!calcPixelValue) 
			continue;
	
		// process avpls
		for(int i = 0; i < chunkSize; ++i)
		{		
			int matIndex = avpl_materialIndex[i];
			MAT avpl_mat(material_diffuse[matIndex],
						 material_specular[matIndex],
						 material_exponent[matIndex]);
			outRad += getRadiance(pos, norm, mat, 
					avpl_position[i], avpl_normal[i], 
					avpl_incDirection[i], avpl_incRadiance[i],
					avpl_mat, camPos);
		}
	}
	
	// remainig avpls
	const int index = chunkSize * numChunks + threadId;
	if (index < numAvpls) {
		avpl_position[threadId] = avpls->position[index];
		avpl_normal[threadId] = avpls->normal[index];
		avpl_incRadiance[threadId] = avpls->incRadiance[index];
		avpl_incDirection[threadId] = avpls->incDirection[index];
		avpl_materialIndex[threadId] = avpls->materialIndex[index];
	}
	syncthreads();
	
	const int remaining = numAvpls - numChunks * chunkSize;
	for (int i = 0; i < remaining; ++i) 
	{
		MAT avpl_mat(material_diffuse[avpl_materialIndex[i]],
					 material_specular[avpl_materialIndex[i]],
					 material_exponent[avpl_materialIndex[i]]);
		outRad += getRadiance(pos, norm, mat, 
				avpl_position[i], avpl_normal[i], 
				avpl_incDirection[i], avpl_incRadiance[i],
				mat, camPos);
	}
	
	if (!calcPixelValue) 
		return;

	float4 out = make_float4(outRad, 1.f);
	surf2Dwrite(out, outResult, x * sizeof(float4), y);
}


// TODO: shared memory
__global__ void kernel_bvh(
		cudaSurfaceObject_t outResult,
		cudaSurfaceObject_t outRadiance,
		cudaSurfaceObject_t outAntiradiance,
		cudaSurfaceObject_t inPositions,
		cudaSurfaceObject_t inNormals,
		BvhParam* bvhParam,
		AvplBvhNodeDataParam* dataParam,
		MaterialsGpuParam* materials,
		float3 camPos,
		int bvhLevel,
		int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
		
	float4 data;
	surf2Dread(&data, inPositions, x * sizeof(float4), y);
	const float3 pos= make_float3(data);
	surf2Dread(&data, inNormals, x * sizeof(float4), y);
	const float3 norm= make_float3(data);
	const int materialIndex = int(data.w);

	const int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	__shared__ float3 material_diffuse[MAX_NUM_MATERIALS];
	__shared__ float3 material_specular[MAX_NUM_MATERIALS];
	__shared__ float material_exponent[MAX_NUM_MATERIALS];
	if (threadId < materials->numMaterials && threadId < MAX_NUM_MATERIALS) {
		material_diffuse[threadId] = materials->diffuse[threadId];
		material_specular[threadId] = materials->specular[threadId];
		material_exponent[threadId] = materials->exponent[threadId];
	}
	syncthreads();
	
	MAT mat(material_diffuse[materialIndex],
			material_specular[materialIndex],
			material_exponent[materialIndex]);
	
	float3 outRad = make_float3(0.f);
	
	int stack[64];
	int stack_depth[64];
	stack[0] = 0;
	stack_depth[0] = 0;
	int stack_ptr = 1;

	while (stack_ptr > 0)
	{
		stack_ptr--;
		const int depth = stack_depth[stack_ptr];	
		const int nodeIndex = stack[stack_ptr];
		const BvhNode bvhNode = bvhParam->nodes[nodeIndex];

		const bool useNode = false;
		if (useNode) { // 
				int idx = nodeIndex	+ bvhParam->numLeafs;
				const int matIdx = dataParam->materialIndex[idx];
				MAT avpl_mat(material_diffuse[matIdx], material_specular[matIdx], material_exponent[matIdx]);
				outRad += getRadiance(pos, norm, mat, avpl_mat, idx, dataParam, bvhParam, camPos);
		} else { // refine
			if (bvhNode.left > 0) { // leaf node
				int idx = bvhNode.left;
				const int matIdx = dataParam->materialIndex[idx];
				MAT avpl_mat(material_diffuse[matIdx], material_specular[matIdx], material_exponent[matIdx]);
				outRad += getRadiance(pos, norm, mat, avpl_mat, idx, dataParam, bvhParam, camPos);
			} else { // inner node
				stack[stack_ptr] = bvhNode.left;
				stack_depth[stack_ptr] = depth + 1;
				stack_ptr++;
			}
			if (bvhNode.right > 0) { // leaf node
				int idx = bvhNode.right;
				const int matIdx = dataParam->materialIndex[idx];
				MAT avpl_mat(material_diffuse[matIdx], material_specular[matIdx], material_exponent[matIdx]);
				outRad += getRadiance(pos, norm, mat, avpl_mat, idx, dataParam, bvhParam, camPos);
			} else {
				stack[stack_ptr] = bvhNode.right;
				stack_depth[stack_ptr] = depth + 1;
				stack_ptr++;
			}
		}
	}
	
	float4 out = make_float4(outRad, 1.f);
	surf2Dwrite(out, outResult, x * sizeof(float4), y);
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
	m_materialsGpu.reset(new MaterialsGpu(materials));
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
		newavpl.L = make_float4(glm::vec4(avpls[i].m_Radiance, 1.f));
		newavpl.A = make_float4(glm::vec4(avpls[i].m_Antiradiance, 1.f));
		newavpl.pos = make_float3(avpls[i].m_Position);
		newavpl.materialIndex = avpls[i].m_MaterialIndex;
		newavpl.norm = make_float3(avpls[i].m_Orientation);
		newavpl.angleFactor = avpls[i].m_ConeAngle;
		newavpl.w = make_float3(avpls[i].m_Direction);
		newavpl.bounce = avpls[i].m_Bounce;
		new_avpls[i] = newavpl;
	}

	AvplsGpu avplsGpu(avpls);

	m_avpls.reset(new CudaBuffer<NEW_AVPL>(new_avpls));

	CudaTimer timer;
	timer.start();
	// Invoke kernel
	dim3 dimBlock(THREADS_X, THREADS_Y);
	dim3 dimGrid((m_width  + dimBlock.x - 1) / dimBlock.x,
		(m_height + dimBlock.y - 1) / dimBlock.y);

	kernel<<<dimGrid, dimBlock>>>(
			resultOutMapped.getCudaSurfaceObject(), 
			radianceOutMapped.getCudaSurfaceObject(), 
			antiradianceOutMapped.getCudaSurfaceObject(), 
			positionsMapped.getCudaSurfaceObject(), 
			normalsMapped.getCudaSurfaceObject(), 
			avplsGpu.param->getDevicePtr(),
			m_materialsGpu->param->getDevicePtr(),
			make_float3(cameraPosition),
			m_width, m_height);
	
	timer.stop();
	std::cout << "kernel execution time: " << timer.getTime() << std::endl;
}

void CudaGather::run_bvh(AvplBvh* avplBvh, glm::vec3 const& cameraPosition, int bvhLevel)
{
	CudaGraphicsResourceMappedArray positionsMapped(m_positionResource.get());
	CudaGraphicsResourceMappedArray normalsMapped(m_normalResource.get());
	CudaGraphicsResourceMappedArray radianceOutMapped(m_radianceOutputResource.get());
	CudaGraphicsResourceMappedArray antiradianceOutMapped(m_antiradianceOutputResource.get());
	CudaGraphicsResourceMappedArray resultOutMapped(m_resultOutputResource.get());

	CudaTimer timer;
	timer.start();
	// Invoke kernel
	dim3 dimBlock(THREADS_X, THREADS_Y);
	dim3 dimGrid((m_width  + dimBlock.x - 1) / dimBlock.x,
		(m_height + dimBlock.y - 1) / dimBlock.y);
	kernel_bvh<<<dimGrid, dimBlock>>>(
			resultOutMapped.getCudaSurfaceObject(), 
			radianceOutMapped.getCudaSurfaceObject(), 
			antiradianceOutMapped.getCudaSurfaceObject(), 
			positionsMapped.getCudaSurfaceObject(), 
			normalsMapped.getCudaSurfaceObject(), 
			avplBvh->getBvhParam(),
			avplBvh->getAvplBvhNodeDataParam(),
			m_materialsGpu->param->getDevicePtr(),
			make_float3(cameraPosition),
			bvhLevel, m_width, m_height);

	timer.stop();
	//std::cout << "kernel execution time: " << timer.getTime() << std::endl;
}

