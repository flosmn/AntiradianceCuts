#include "cudaGather.h"

#include "data_gpu.h"
#include "bvh.h"

#include "SceneProbe.h"

#include "CudaResources/cudaTimer.hpp"
#include "CudaResources/cudaUtil.hpp"

#include "Utils/stream.h"

#include "CConfigManager.h"
#include "CCamera.h"

#define THREADS_X 16
#define THREADS_Y 16
#define MAX_NUM_MATERIALS 20
#define STACK_SIZE 32
#define NUM_THREADS_GATHER_CLUSTER_LIGHTS 32

using namespace cuda;

template<typename T>
void printDeviceVector(std::string const& text, thrust::device_vector<T> const& vector)
{
	std::cout << text << " ";
	for (int i = 0; i < vector.size(); i++) {
		std::cout << vector[i] << ", ";
	}
	std::cout << std::endl;
}

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
	return d + s;
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

	// check for light source Avpl
	if(length(avpl_w) == 0.f)
		brdf_light = make_float3(1.f);
		 
	const float3 brdf = f_r(avpl_pos, pos, camPos, norm, mat);

	return avpl_L * brdf_light * G(pos, norm, avpl_pos, avpl_norm) * brdf;
}

inline __device__ bool intersectLinePlane(float3 const& dir_line, float3 const& point_line,
		float3 const& normal_plane, float3 const& point_plane, float &t)
{
	const float denom = dot(dir_line, normal_plane);
	
	if (denom == 0.f) 
		return false;

	t = dot(point_plane - point_line, normal_plane) / denom;
	return true;
}

inline __device__ float3 getAntiradiance(float3 const& pos, 
		float3 const& norm, MAT const& mat,
		float3 const& avpl_pos, float3 const& avpl_norm, 
		float3 const& avpl_w, float3 const& avpl_A, 
		MAT const& avpl_mat, float3 const& camPos,
		float radius)
{
	float3 antirad = make_float3(0.f);

	float d = 0.f;
	if (!intersectLinePlane(avpl_w, avpl_pos, norm, pos, d) || d <= 0.f) {
		return antirad;
	}
	const float3 hitPoint = avpl_pos + d * normalize(avpl_w);
	
	if (length(hitPoint - pos) >= radius) {
		return antirad;
	}
	
	if (dot(normalize(hitPoint - avpl_pos), avpl_w) <= 0.f) {
		return antirad;
	}
	
	if (dot(norm, avpl_w) >= 0.f) {
		return antirad;
	}
	
	const float3 direction = normalize(pos - avpl_pos);
	if (dot(direction, avpl_norm) >= 0.f) {
		return antirad;
	}
	
	const float cos_theta = max(0.f, dot(norm, -direction));
	const float3 brdf = f_r(avpl_pos, pos, camPos, norm, mat);
	return avpl_A * /*cos_theta **/ brdf / (CUDA_PI * radius * radius);
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

__global__ void kernel_radiance(
		cudaSurfaceObject_t outRadiance,
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
				avpl_mat, camPos);
	}
	
	if (!calcPixelValue) 
		return;

	float4 out = make_float4(outRad, 1.f);
	surf2Dwrite(out, outRadiance, x * sizeof(float4), y);
}

__global__ void kernel_radiance_simple(
		cudaSurfaceObject_t outRadiance,
		cudaSurfaceObject_t inPositions,
		cudaSurfaceObject_t inNormals,
		AvplsGpuParam* avpls,
		MaterialsGpuParam* materials,
		float3 camPos,
		int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= width || y >= height) {
		return;
	}
	
	float4 data;
	surf2Dread(&data, inPositions, x * sizeof(float4), y);
	const float3 pos = make_float3(data);
	surf2Dread(&data, inNormals, x * sizeof(float4), y);
	const float3 norm = make_float3(data);
	const int materialIndex = int(data.w);
	
	float3 outRad		= make_float3(0.f);
	
	const int numAvpls = avpls->numAvpls;
	
	MAT mat(materials->diffuse[materialIndex],
			materials->specular[materialIndex],
			materials->exponent[materialIndex]);

	for (int i = 0; i < numAvpls; ++i) 
	{
		const int avpl_mat_index = avpls->materialIndex[i];
		MAT avpl_mat(materials->diffuse[avpl_mat_index],
					 materials->specular[avpl_mat_index],
					 materials->exponent[avpl_mat_index]);
		outRad += getRadiance(pos, norm, mat, 
				avpls->position[i], avpls->normal[i], 
				avpls->incDirection[i], avpls->incRadiance[i],
				avpl_mat, camPos);
	}
	
	float4 out = make_float4(outRad, 1.f);
	surf2Dwrite(out, outRadiance, x * sizeof(float4), y);
}

__global__ void kernel_bvh(
		cudaSurfaceObject_t outRadiance,
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
	surf2Dwrite(out, outRadiance, x * sizeof(float4), y);
}

__global__ void kernel_antiradiance(
		cudaSurfaceObject_t outAntiradiance,
		cudaSurfaceObject_t inPositions,
		cudaSurfaceObject_t inNormals,
		AvplsGpuParam* avpls,
		MaterialsGpuParam* materials,
		float3 camPos,
		float photonRadius,
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

	float3 outAntirad	= make_float3(0.f);
	
	const int numAvpls = avpls->numAvpls;
	const int threadId = threadIdx.x + threadIdx.y * blockDim.x;
	const int chunkSize = THREADS_X * THREADS_Y;
	const int numChunks = max(numAvpls / chunkSize, 0);
	
	__shared__ float3 avpl_position[chunkSize];
	__shared__ float3 avpl_normal[chunkSize];
	__shared__ float3 avpl_antiradiance[chunkSize];
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
		avpl_antiradiance[threadId] = avpls->antiradiance[index];
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
			outAntirad += getAntiradiance(pos, norm, mat, 
					avpl_position[i], avpl_normal[i], 
					avpl_incDirection[i], avpl_antiradiance[i],
					avpl_mat, camPos, photonRadius);
		}
	}
	
	// remainig avpls
	const int index = chunkSize * numChunks + threadId;
	if (index < numAvpls) {
		avpl_position[threadId] = avpls->position[index];
		avpl_normal[threadId] = avpls->normal[index];
		avpl_antiradiance[threadId] = avpls->antiradiance[index];
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
		outAntirad += getAntiradiance(pos, norm, mat, 
				avpl_position[i], avpl_normal[i], 
				avpl_incDirection[i], avpl_antiradiance[i],
				avpl_mat, camPos, photonRadius);
	}
	
	if (!calcPixelValue) 
		return;

	float4 out = make_float4(outAntirad, 1.f);
	surf2Dwrite(out, outAntiradiance, x * sizeof(float4), y);
}

__global__ void kernel_antiradiance_simple(
		cudaSurfaceObject_t outAntiradiance,
		cudaSurfaceObject_t inPositions,
		cudaSurfaceObject_t inNormals,
		AvplsGpuParam* avpls,
		MaterialsGpuParam* materials,
		float3 camPos,
		float photonRadius,
		int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
		
	if (x >= width || y >= height) {
		return;
	}
	
	float4 data;
	surf2Dread(&data, inPositions, x * sizeof(float4), y);
	const float3 pos = make_float3(data);
	surf2Dread(&data, inNormals, x * sizeof(float4), y);
	const float3 norm = make_float3(data);
	const int materialIndex = int(data.w);
	
	float3 outAntirad = make_float3(0.f);
	
	MAT mat(materials->diffuse[materialIndex],
			materials->specular[materialIndex],
			materials->exponent[materialIndex]);

	for (int i = 0; i < avpls->numAvpls; ++i) 
	{
		const int avpl_mat_index = avpls->materialIndex[i];
		MAT avpl_mat(materials->diffuse[avpl_mat_index],
					 materials->specular[avpl_mat_index],
					 materials->exponent[avpl_mat_index]);
		outAntirad += getAntiradiance(pos, norm, mat, 
				avpls->position[i], avpls->normal[i], 
				avpls->incDirection[i], avpls->antiradiance[i],
				avpl_mat, camPos, photonRadius);
	}
	
	float4 out = make_float4(outAntirad, 1.f);
	surf2Dwrite(out, outAntiradiance, x * sizeof(float4), y);
}

__global__ void kernel_antiradiance_clusterred(
		dim3 dimensions,
		cudaSurfaceObject_t outAntiradiance,
		cudaSurfaceObject_t inPositions,
		cudaSurfaceObject_t inNormals,
		AvplsGpuParam* avpls,
		MaterialsGpuParam* materials,
		float3 camPos,
		float photonRadius,
		int* numLightsPerCluster,
		int* lightsPerCluster,
		int* clusterIds,
		int maxNumLightsPerCluster,
		int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
		
	if (x >= width || y >= height) {
		return;
	}
	
	float4 data;
	surf2Dread(&data, inPositions, x * sizeof(float4), y);
	const float3 pos = make_float3(data);
	surf2Dread(&data, inNormals, x * sizeof(float4), y);
	const float3 norm = make_float3(data);
	const int materialIndex = int(data.w);

	MAT mat(materials->diffuse[materialIndex], 
			materials->specular[materialIndex],
			materials->exponent[materialIndex]);

	float3 outAntirad = make_float3(0.f);
	
	const int clusterId = clusterIds[y * dimensions.x + x];
	const int numLights = numLightsPerCluster[clusterId];
	const int clusterOffset = clusterId * maxNumLightsPerCluster;

	for (int i = 0; i < numLights; ++i) 
	{
		int lightIdx = lightsPerCluster[clusterOffset + i];
		int matIndex = avpls->materialIndex[lightIdx];
		MAT avpl_mat(materials->diffuse[matIndex],
					 materials->specular[matIndex],
					 materials->exponent[matIndex]);
		outAntirad += getAntiradiance(pos, norm, mat, 
				avpls->position[lightIdx], avpls->normal[lightIdx], 
				avpls->incDirection[lightIdx], avpls->antiradiance[lightIdx],
				avpl_mat, camPos, photonRadius);
	}

	float4 out = make_float4(outAntirad, 1.f);
	surf2Dwrite(out, outAntiradiance, x * sizeof(float4), y);
}



// http://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
__device__ bool intersectBB(float3 const& bbMin, float3 const& bbMax, float3 const& o, float3 const& d) {
	const float dirfracx = 1.0f / d.x;
	const float dirfracy = 1.0f / d.y;
	const float dirfracz = 1.0f / d.z;
	
	const float t1 = (bbMin.x - o.x) * dirfracx;
	const float t2 = (bbMax.x - o.x) * dirfracx;
	const float t3 = (bbMin.y - o.y) * dirfracy;
	const float t4 = (bbMax.y - o.y) * dirfracy;
	const float t5 = (bbMin.z - o.z) * dirfracz;
	const float t6 = (bbMax.z - o.z) * dirfracz;
	
	const float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	const float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));
	
	if (tmax < 0 || tmin > tmax) {
		return false;
	}
	
	return true;
}

__global__ void kernel_gather_cluster_lights(
		AvplsGpuParam* avpls,
		int* numLightsPerCluster,
		int* lightsPerCluster,
		float photonRadius,
		int maxNumLightsPerCluster,
		float3* clusterBBMin,
		float3* clusterBBMax,
		int numClusters)	
{
	const int clusterId = blockIdx.x * blockDim.x + threadIdx.x;
	const bool validClusterId = clusterId < numClusters;
	const float3 bbMin = clusterBBMin[clusterId] - 2.f * make_float3(photonRadius);
	const float3 bbMax = clusterBBMax[clusterId] + 2.f * make_float3(photonRadius);
	const int clusterOffset = maxNumLightsPerCluster * clusterId; 
	int numClusterLights = 0;
	
	const int numLights = avpls->numAvpls;
	const int threadId = threadIdx.x;
	const int chunkSize = NUM_THREADS_GATHER_CLUSTER_LIGHTS;
	const int numChunks = numLights / chunkSize;
	
	__shared__ float3 avpl_position[chunkSize];
	__shared__ float3 avpl_incDirection[chunkSize];
	
	for (int chunk = 0; chunk < numChunks; ++chunk) 
	{
		// load chunk into shared memory
		const int index = chunkSize * chunk + threadId;
		avpl_position[threadId] = avpls->position[index];
		avpl_incDirection[threadId] = avpls->incDirection[index];
		syncthreads();
	
		if (!validClusterId)
			continue;

		for(int i = 0; i < chunkSize; ++i)
		{
			// check for hit with cluster
			if (intersectBB(bbMin, bbMax, avpl_position[i], avpl_incDirection[i])) {
				const int idx = chunkSize * chunk + i;
				lightsPerCluster[clusterOffset + numClusterLights] = idx;
				numClusterLights++;
			}
		}
	}
	
	syncthreads();
	
	if (!validClusterId)
		return;

	// remainig avpls
	for (int i = numChunks * chunkSize; i < numLights; ++i) 
	{
		if (intersectBB(bbMin, bbMax, avpls->position[i], avpls->incDirection[i])) {
			lightsPerCluster[clusterOffset + numClusterLights] = i;
			numClusterLights++;
		}
	}

	numLightsPerCluster[clusterId] = numClusterLights;
}

__global__ void kernel_gather_cluster_lights_simple(
		AvplsGpuParam* avpls,
		int* numLightsPerCluster,
		int* lightsPerCluster,
		float photonRadius,
		int maxNumLightsPerCluster,
		float3* clusterBBMin,
		float3* clusterBBMax,
		int numClusters)	
{
	const int clusterId = blockIdx.x * blockDim.x + threadIdx.x;

	if (clusterId >= numClusters) {
		return;
	}

	const float3 bbMin = clusterBBMin[clusterId] - 2.f * make_float3(photonRadius);
	const float3 bbMax = clusterBBMax[clusterId] + 2.f * make_float3(photonRadius);
	const int clusterOffset = maxNumLightsPerCluster * clusterId; 
	int numClusterLights = 0;
	
	for (int i = 0; i < avpls->numAvpls; ++i) 
	{
		if (intersectBB(bbMin, bbMax, avpls->position[i], avpls->incDirection[i])) {
			lightsPerCluster[clusterOffset + numClusterLights] = i;
			numClusterLights++;
		}
	}
	
	numLightsPerCluster[clusterId] = numClusterLights;
}

__global__ void kernel_combine(
		cudaSurfaceObject_t outResult,
		cudaSurfaceObject_t inRadiance,
		cudaSurfaceObject_t inAntiradiance,
		int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	float4 radiance;
	surf2Dread(&radiance, inRadiance, x * sizeof(float4), y);
	float4 antiradiance;
	surf2Dread(&antiradiance, inAntiradiance, x * sizeof(float4), y);

	//float4 out = max(make_float4(0.f), radiance - antiradiance);
	float4 out = radiance - antiradiance;
	surf2Dwrite(out, outResult, x * sizeof(float4), y);
}

CudaGather::CudaGather(CCamera* camera, 
	GLuint glPositonTexture, GLuint glNormalTexture,
	GLuint glResultOutputTexture,
	GLuint glRadianceOutputTexture,
	GLuint glAntiradianceOutputTexture,
	std::vector<MATERIAL> const& materials,
	COGLUniformBuffer* ubTransform,
	CConfigManager* confManager)
	: m_camera(camera), m_width(camera->GetWidth()), m_height(camera->GetHeight()), 
	  m_ubTransform(ubTransform), m_confManager(confManager)
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

	m_materialsGpu.reset(new MaterialsGpu(materials));
}

CudaGather::~CudaGather()
{
}

void CudaGather::run(std::vector<Avpl> const& avpls, glm::vec3 const& cameraPosition, 
	SceneProbe* sceneProbe, float sceneExtent, bool profile)
{
	CudaGraphicsResourceMappedArray positionsMapped(m_positionResource.get());
	CudaGraphicsResourceMappedArray normalsMapped(m_normalResource.get());
	CudaGraphicsResourceMappedArray radianceOutMapped(m_radianceOutputResource.get());
	CudaGraphicsResourceMappedArray antiradianceOutMapped(m_antiradianceOutputResource.get());
	CudaGraphicsResourceMappedArray resultOutMapped(m_resultOutputResource.get());
	
	CudaTimer timer;
	timer.start();	
	AvplsGpu avplsGpu(avpls);
	timer.stop();
	if(profile) std::cout << "transfer avpl data to gpu took: " << timer.getTime() << "ms" << std::endl;

	////////////////////////////////////////////////////////////////////////
	// gather radiance
	////////////////////////////////////////////////////////////////////////
	if (m_confManager->GetConfVars()->useLightCuts) {
		bool generateDebugInfo = m_confManager->GetConfVars()->UseDebugMode && sceneProbe;
		cuda::CudaTimer buildBvh;
		buildBvh.start();
		std::unique_ptr<AvplBvh> avplBvh = std::unique_ptr<AvplBvh>(new AvplBvh(avpls, false));
		buildBvh.stop();
		
		cuda::CudaTimer gather;
		gather.start();
		glm::uvec2 pixel(0, 0);
		if (sceneProbe) pixel = sceneProbe->getPixel();
		
		thrust::device_vector<int> usedAvplsGpu(avplBvh->getBvhData()->numLeafs + avplBvh->getBvhData()->numNodes);
		thrust::host_vector<int> usedAvplsCpu(avplBvh->getBvhData()->numLeafs + avplBvh->getBvhData()->numNodes);
		thrust::fill(usedAvplsGpu.begin(), usedAvplsGpu.end(), 0);
		
		// Invoke kernel
		dim3 dimBlock(THREADS_X, THREADS_Y);
		dim3 dimGrid((m_width  + dimBlock.x - 1) / dimBlock.x,
			(m_height + dimBlock.y - 1) / dimBlock.y);
		kernel_bvh<<<dimGrid, dimBlock>>>(
				radianceOutMapped.getCudaSurfaceObject(), 
				positionsMapped.getCudaSurfaceObject(), 
				normalsMapped.getCudaSurfaceObject(), 
				avplBvh->getBvhParam(),
				avplBvh->getAvplBvhNodeDataParam(),
				m_materialsGpu->param->getDevicePtr(),
				make_float3(cameraPosition),
				m_confManager->GetConfVars()->bvhLevel, 
				m_confManager->GetConfVars()->ClusterRefinementThreshold,
				generateDebugInfo,
				make_uint2(pixel),
				thrust::raw_pointer_cast(&usedAvplsGpu[0]),
				m_width, m_height);

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
		
			m_pointCloud.reset(new PointCloud(positions, colors, m_ubTransform, 
				m_confManager->GetConfVars()->lightRadiusScale * sceneExtent / 100.f));
			m_aabbCloud.reset(new AABBCloud(bbMins, bbMaxs, m_ubTransform));
		}

		gather.stop();
		if(profile) std::cout << "gather build bvh took: " << buildBvh.getTime() << "ms" << std::endl;
		if(profile) std::cout << "gather radiance with lightcuts took: " << gather.getTime() << "ms" << std::endl;
	}
	else {
		CudaTimer gather;
		gather.start();
		// Invoke kernel
		dim3 dimBlock(THREADS_X, THREADS_Y);
		dim3 dimGrid((m_width  + dimBlock.x - 1) / dimBlock.x,
			(m_height + dimBlock.y - 1) / dimBlock.y);

		kernel_radiance_simple<<<dimGrid, dimBlock>>>(
				radianceOutMapped.getCudaSurfaceObject(), 
				positionsMapped.getCudaSurfaceObject(), 
				normalsMapped.getCudaSurfaceObject(), 
				avplsGpu.param->getDevicePtr(),
				m_materialsGpu->param->getDevicePtr(),
				make_float3(cameraPosition),
				m_width, m_height);
		
		gather.stop();
		if(profile) std::cout << "gather radiance without lightcuts took: " << gather.getTime() << "ms" << std::endl;
	}

	////////////////////////////////////////////////////////////////////////
	// gather antiradiance
	////////////////////////////////////////////////////////////////////////
	{
		const float photonRadius = m_confManager->GetConfVars()->photonRadiusScale * sceneExtent / 100.f;
		if (m_confManager->GetConfVars()->useClusteredDeferred) {
			const int maxNumLightsPerCluster = avpls.size();
			thrust::device_vector<int> numLightsPerCluster(m_visiblePointsBvh->m_numClusters, 0);
			thrust::device_vector<int> lightsPerCluster(maxNumLightsPerCluster * m_visiblePointsBvh->m_numClusters, 0);
			{	// gather lights per cluster
				CudaTimer timer;
				timer.start();
				
				if (m_confManager->GetConfVars()->gatherClusterLightsSimple) {
					dim3 dimBlock(128);
					dim3 dimGrid((m_visiblePointsBvh->m_numClusters + dimBlock.x - 1) / dimBlock.x);
					kernel_gather_cluster_lights_simple<<<dimGrid, dimBlock>>>(
						avplsGpu.param->getDevicePtr(),
						thrust::raw_pointer_cast(&numLightsPerCluster[0]),
						thrust::raw_pointer_cast(&lightsPerCluster[0]),
						photonRadius,
						maxNumLightsPerCluster,
						thrust::raw_pointer_cast(&(m_visiblePointsBvh->m_clusterBBMin[0])),
						thrust::raw_pointer_cast(&(m_visiblePointsBvh->m_clusterBBMax[0])),
						m_visiblePointsBvh->m_numClusters);
				} else {
					dim3 dimBlock(NUM_THREADS_GATHER_CLUSTER_LIGHTS);
					dim3 dimGrid((m_visiblePointsBvh->m_numClusters + dimBlock.x - 1) / dimBlock.x);
					kernel_gather_cluster_lights<<<dimGrid, dimBlock>>>(
						avplsGpu.param->getDevicePtr(),
						thrust::raw_pointer_cast(&numLightsPerCluster[0]),
						thrust::raw_pointer_cast(&lightsPerCluster[0]),
						photonRadius,
						maxNumLightsPerCluster,
						thrust::raw_pointer_cast(&(m_visiblePointsBvh->m_clusterBBMin[0])),
						thrust::raw_pointer_cast(&(m_visiblePointsBvh->m_clusterBBMax[0])),
						m_visiblePointsBvh->m_numClusters);
				}
				
				timer.stop();
				if(profile) std::cout << "gather lights per cluster: " << timer.getTime() << "ms" << std::endl;
			}
			
			// gather antiradiance
			CudaTimer gather;
			gather.start();
			dim3 dimBlock(THREADS_X, THREADS_Y);
			dim3 dimGrid((m_width  + dimBlock.x - 1) / dimBlock.x,
				(m_height + dimBlock.y - 1) / dimBlock.y);
			dim3 dimensions(m_width, m_height);
			kernel_antiradiance_clusterred<<<dimGrid, dimBlock>>>(
				dimensions,
				antiradianceOutMapped.getCudaSurfaceObject(), 
				positionsMapped.getCudaSurfaceObject(), 
				normalsMapped.getCudaSurfaceObject(), 
				avplsGpu.param->getDevicePtr(),
				m_materialsGpu->param->getDevicePtr(),
				make_float3(cameraPosition),
				photonRadius,
				thrust::raw_pointer_cast(&numLightsPerCluster[0]),
				thrust::raw_pointer_cast(&lightsPerCluster[0]),
				thrust::raw_pointer_cast(&(m_visiblePointsBvh->m_clusterIdBuffer[0])),
				maxNumLightsPerCluster,
				m_width, m_height);
			gather.stop();
			if(profile) std::cout << "gather antiradiance took: " << gather.getTime() << "ms" << std::endl;
		}
		else{
			CudaTimer gather;
			gather.start();
			dim3 dimBlock(THREADS_X, THREADS_Y);
			dim3 dimGrid((m_width  + dimBlock.x - 1) / dimBlock.x,
				(m_height + dimBlock.y - 1) / dimBlock.y);

			kernel_antiradiance<<<dimGrid, dimBlock>>>(
				antiradianceOutMapped.getCudaSurfaceObject(), 
				positionsMapped.getCudaSurfaceObject(), 
				normalsMapped.getCudaSurfaceObject(), 
				avplsGpu.param->getDevicePtr(),
				m_materialsGpu->param->getDevicePtr(),
				make_float3(cameraPosition),
				m_confManager->GetConfVars()->photonRadiusScale * sceneExtent / 100.f,
				m_width, m_height);
			gather.stop();
			if(profile) std::cout << "gather antiradiance took: " << gather.getTime() << "ms" << std::endl;
		}
	}

	////////////////////////////////////////////////////////////////////////
	// combine radiance and antiradiance
	////////////////////////////////////////////////////////////////////////
	{
		CudaTimer timer;
		timer.start();
		dim3 dimBlock(32, 32);
		dim3 dimGrid((m_width  + dimBlock.x - 1) / dimBlock.x,
			(m_height + dimBlock.y - 1) / dimBlock.y);

		kernel_combine<<<dimGrid, dimBlock>>>(
			resultOutMapped.getCudaSurfaceObject(),
			radianceOutMapped.getCudaSurfaceObject(),
			antiradianceOutMapped.getCudaSurfaceObject(), 
			m_width, m_height);
		timer.stop();
		if(profile) std::cout << "combine radiance and antiradiance took: " << timer.getTime() << "ms" << std::endl;
	}
}

void CudaGather::rebuildVisiblePointsBvh()
{
	CudaGraphicsResourceMappedArray positionsMapped(m_positionResource.get());
	CudaGraphicsResourceMappedArray normalsMapped(m_normalResource.get());

	m_visiblePointsBvh.reset(new VisiblePointsBvh(
		positionsMapped.getCudaSurfaceObject(),
		normalsMapped.getCudaSurfaceObject(),
		m_width, m_height, false,
		make_float3(m_camera->GetPosition()), m_camera->getZNear(), m_camera->getTheta()));
}
