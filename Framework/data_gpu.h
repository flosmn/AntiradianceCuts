#ifndef DATAGPU_H_
#define DATAGPU_H_

#include "AVPL.h"
#include "Material.h"

#include "CudaResources/cudaBuffer.hpp"

#include <thrust/device_vector.h>

#include <memory>
#include <vector>

struct AvplsGpuParam
{
	float3* position;
	float3* normal;
	float3* incDirection;
	float3* incRadiance;
	float3* antiradiance;
	int* materialIndex;
	int* bounce;
	int numAvpls;
};

struct AvplsGpu
{
	AvplsGpu(std::vector<AVPL> const& avpls);

	std::unique_ptr<cuda::CudaBuffer<AvplsGpuParam>> param;

	thrust::device_vector<float3> position;
	thrust::device_vector<float3> normal;
	thrust::device_vector<float3> incDirection;	// incoming direction
	thrust::device_vector<float3> incRadiance;	// incident radiance
	thrust::device_vector<float3> antiradiance;	// incident radiance
	thrust::device_vector<int> materialIndex;
	thrust::device_vector<int> bounce;
	int numAvpls;
};

struct MaterialsGpuParam
{
	float3* emissive;
	float3* diffuse;
	float3* specular;
	float* exponent;
	int numMaterials;
};

struct MaterialsGpu
{
	MaterialsGpu(std::vector<MATERIAL> const& materials);

	std::unique_ptr<cuda::CudaBuffer<MaterialsGpuParam>> param;

	thrust::device_vector<float3> emissive;
	thrust::device_vector<float3> diffuse;
	thrust::device_vector<float3> specular;
	thrust::device_vector<float> exponent;
	int numMaterials;
};

#endif // DATAGPU_H_
