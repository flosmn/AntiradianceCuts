#include "cudaGather.h"

#include "CudaResources/cudaTimer.hpp"

#include "data_gpu.h"

#include "Utils/stream.h"

#include "CudaResources/cudaUtil.hpp"

#define THREADS_X 16
#define THREADS_Y 16
#define MAX_NUM_MATERIALS 20
#define STACK_SIZE 1024

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
	const float3 d = CUDA_ONE_OVER_PI * mat.diffuse;
	const float cos_theta = max(0.f, dot(reflect(-w_i, n), w_o));
	const float3 s = 0.5f * CUDA_ONE_OVER_PI * (mat.exponent+2.f)
		* pow(cos_theta, mat.exponent) * mat.specular;
	return d; //vec4(d.x + s.x, d.y + s.y, d.z + s.z, 1.f);
}

inline __device__ float3 f_r(float3 const& from, float3 const& over, float3 const& to,
		float3 const& n, MAT const& mat)
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
		int idx, AvplBvhNodeDataParam* dataParam, float3 const& camPos)
{
	const float3 avpl_pos = dataParam->position[idx];
	const float3 avpl_norm = dataParam->normal[idx];
	const float3 avpl_incRad = dataParam->incRadiance[idx];
	const float3 avpl_incDir = dataParam->incDirection[idx];
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
		float refThresh, 
		bool genDebugInfo,
		uint2 debugPixel,
		int* usedAvpls,
		int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	bool generateDebugInfo = (x == debugPixel.x) && (y == debugPixel.y) && genDebugInfo;
		
	float4 data;
	surf2Dread(&data, inPositions, x * sizeof(float4), y);
	const float3 pos= make_float3(data);
	surf2Dread(&data, inNormals, x * sizeof(float4), y);
	const float3 norm= make_float3(data);
	const int materialIndex = int(data.w);

	//if (length(pos) == 0 && length(norm) == 0) {
	//	float4 out = make_float4(0.f, 100000.f, 0.f, 1.f);
	//	surf2Dwrite(out, outResult, x * sizeof(float4), y);
	//	return;
	//}

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

	/*	
	for (int i = 0; i < bvhParam->numLeafs + bvhParam->numNodes; ++i) {
		int matIndex = dataParam->materialIndex[i];
		MAT avpl_mat(material_diffuse[matIndex],
					 material_specular[matIndex],
					 material_exponent[matIndex]);
		outRad += getRadiance(pos, norm, mat, 
				dataParam->position[i], dataParam->normal[i], 
				dataParam->incDirection[i], dataParam->incRadiance[i],
				avpl_mat, camPos);
	}
	*/
		
	int stack[STACK_SIZE];
	int stack_depth[STACK_SIZE];
	stack[0] = -1;
	stack_depth[0] = 0;
	int stack_ptr = 1;
	bool error = false;

	while (stack_ptr > 0) 
	{
		stack_ptr--;
		const int nodeIndex = stack[stack_ptr];
		const int nodeDepth = stack_depth[stack_ptr];

		if (nodeIndex >= 0) { // leaf
			const int matIdx = dataParam->materialIndex[nodeIndex];
			MAT avpl_mat(material_diffuse[matIdx], material_specular[matIdx], material_exponent[matIdx]);
			outRad += getRadiance(pos, norm, mat, avpl_mat, nodeIndex, dataParam, camPos);
			if (generateDebugInfo) {
				usedAvpls[nodeIndex] = 1;
			}
		} else { // inner node
			const int idx = bvhParam->numLeafs-(nodeIndex+1);
			const BvhNode bvhNode = bvhParam->nodes[-(nodeIndex+1)];
			
			if (-(nodeIndex+1) < 0 || -(nodeIndex+1) >= bvhParam->numNodes) {
				error = true;
				break;
			}
			const bool clusterVisible = (dot(norm, normalize(bvhNode.bbMax - pos)) > 0) ||
										(dot(norm, normalize(bvhNode.bbMin - pos)) > 0);
			if (!clusterVisible)
				continue;
			
			const float3 clusterToPoint = normalize(dataParam->position[idx] - pos);
			const float radius = 0.5f * length(bvhNode.bbMax - bvhNode.bbMin);
			const float dist = length(dataParam->position[idx] - pos);
			const float solidAngle = 2.f * CUDA_PI * (1.f - dist / (sqrt(radius * radius + dist * dist)));
			const bool useNode = (solidAngle < refThresh);

			if (useNode) {
				const int matIdx = dataParam->materialIndex[idx];
				MAT avpl_mat(material_diffuse[matIdx], material_specular[matIdx], material_exponent[matIdx]);
				outRad += getRadiance(pos, norm, mat, avpl_mat, idx, dataParam, camPos);
				if (generateDebugInfo) {
					usedAvpls[idx] = 1;
				}
			} else {
				stack[stack_ptr] = bvhNode.left;
				stack_depth[stack_ptr] = nodeDepth + 1;
				stack_ptr++;
				stack[stack_ptr] = bvhNode.right;
				stack_depth[stack_ptr] = nodeDepth + 1;
				stack_ptr++;

				if (stack_ptr >= STACK_SIZE) {
					error = true;
					break;
				}
			}
		}
	}
	
	float4 out = make_float4(outRad, 1.f);
	if (error) {
		out = make_float4(100000.f, 0.f, 100000.f, 1.f);
	}
	surf2Dwrite(out, outResult, x * sizeof(float4), y);
}


CudaGather::CudaGather(int width, int height, 
	GLuint glPositonTexture, GLuint glNormalTexture,
	GLuint glResultOutputTexture,
	GLuint glRadianceOutputTexture,
	GLuint glAntiradianceOutputTexture,
	std::vector<MATERIAL> const& materials,
	COGLUniformBuffer* ubTransform)
	: m_width(width), m_height(height), m_ubTransform(ubTransform)
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

void CudaGather::run_bvh(AvplBvh* avplBvh, glm::vec3 const& cameraPosition, int bvhLevel, float refThresh,
		glm::uvec2 const& debugPixel, bool generateDebugInfo)
{
	CudaGraphicsResourceMappedArray positionsMapped(m_positionResource.get());
	CudaGraphicsResourceMappedArray normalsMapped(m_normalResource.get());
	CudaGraphicsResourceMappedArray radianceOutMapped(m_radianceOutputResource.get());
	CudaGraphicsResourceMappedArray antiradianceOutMapped(m_antiradianceOutputResource.get());
	CudaGraphicsResourceMappedArray resultOutMapped(m_resultOutputResource.get());
	
	thrust::device_vector<int> usedAvplsGpu(avplBvh->getBvhData()->numLeafs + avplBvh->getBvhData()->numNodes);
	thrust::host_vector<int> usedAvplsCpu(avplBvh->getBvhData()->numLeafs + avplBvh->getBvhData()->numNodes);
	thrust::fill(usedAvplsGpu.begin(), usedAvplsGpu.end(), 0);
	
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
			bvhLevel, 
			refThresh, 
			generateDebugInfo,
			make_uint2(debugPixel),
			thrust::raw_pointer_cast(&usedAvplsGpu[0]),
			m_width, m_height);

	timer.stop();
	std::cout << "kernel execution time: " << timer.getTime() << std::endl;
	
	if (generateDebugInfo) {
		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> colors;
		std::vector<glm::vec3> bbMins;
		std::vector<glm::vec3> bbMaxs;
		int numLeafs = avplBvh->getBvhData()->numLeafs;
		thrust::copy(usedAvplsGpu.begin(), usedAvplsGpu.end(), usedAvplsCpu.begin());
		for (int i = 0; i < usedAvplsCpu.size(); ++i) {
			if (usedAvplsCpu[i] == 1) {
				glm::vec3 pos(make_vec3(avplBvh->getAvplBvhNodeData()->position[i]));
				positions.push_back(pos);
				if (i < numLeafs) {
					bbMins.push_back(pos);
					bbMaxs.push_back(pos);
					colors.push_back(glm::vec3(1.f, 0.f, 0.f));
				} else {
					colors.push_back(glm::vec3(0.f, 1.f, 0.f));
					int idx = i - numLeafs;
					glm::vec3 bbmin(make_vec3(avplBvh->getNode(idx).bbMin));
					glm::vec3 bbmax(make_vec3(avplBvh->getNode(idx).bbMax));
					bbMins.push_back(bbmin);
					bbMaxs.push_back(bbmax);
				}
			}
		}

		m_pointCloud.reset(new PointCloud(positions, colors, m_ubTransform));
		m_aabbCloud.reset(new AABBCloud(bbMins, bbMaxs, m_ubTransform));
	}
}

