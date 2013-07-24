#include "data_gpu.h"

#include <thrust/host_vector.h>

AvplsGpu::AvplsGpu(std::vector<Avpl> const& avpls)
{
	numAvpls = avpls.size();
	thrust::host_vector<float3> position_h(numAvpls);
	thrust::host_vector<float3> normal_h(numAvpls);
	thrust::host_vector<float3> incDirection_h(numAvpls);	// incoming direction
	thrust::host_vector<float3> incRadiance_h(numAvpls);	// incident radiance
	thrust::host_vector<float3> antiradiance_h(numAvpls);	// incident radiance
	thrust::host_vector<int> materialIndex_h(numAvpls);
	thrust::host_vector<int> bounce_h(numAvpls);

	for (int i = 0; i < avpls.size(); ++i) {
		position_h[i] = make_float3(avpls[i].getPosition());
		normal_h[i] = make_float3(avpls[i].getNormal());
		incDirection_h[i] = make_float3(avpls[i].getIncidentDirection());
		incRadiance_h[i] = make_float3(avpls[i].getIncidentRadiance());
		antiradiance_h[i] = make_float3(avpls[i].getAntiradiance());
		materialIndex_h[i] = avpls[i].getMaterialIndex();
		bounce_h[i] = avpls[i].getBounce();
	}

	position.resize(numAvpls);
	normal.resize(numAvpls);
	incDirection.resize(numAvpls);
	incRadiance.resize(numAvpls);
	antiradiance.resize(numAvpls);
	materialIndex.resize(numAvpls);
	bounce.resize(numAvpls);

	thrust::copy(position_h.begin(), position_h.end(), position.begin());
	thrust::copy(normal_h.begin(), normal_h.end(), normal.begin());
	thrust::copy(incDirection_h.begin(), incDirection_h.end(), incDirection.begin());
	thrust::copy(incRadiance_h.begin(), incRadiance_h.end(), incRadiance.begin());
	thrust::copy(antiradiance_h.begin(), antiradiance_h.end(), antiradiance.begin());
	thrust::copy(materialIndex_h.begin(), materialIndex_h.end(), materialIndex.begin());
	thrust::copy(bounce_h.begin(), bounce_h.end(), bounce.begin());

	AvplsGpuParam p;
	p.position = thrust::raw_pointer_cast(&position[0]);
	p.normal = thrust::raw_pointer_cast(&normal[0]);
	p.incDirection = thrust::raw_pointer_cast(&incDirection[0]);
	p.incRadiance = thrust::raw_pointer_cast(&incRadiance[0]);
	p.antiradiance = thrust::raw_pointer_cast(&antiradiance[0]);
	p.materialIndex = thrust::raw_pointer_cast(&materialIndex[0]);
	p.bounce = thrust::raw_pointer_cast(&bounce[0]);
	p.numAvpls = numAvpls;
	param.reset(new cuda::CudaBuffer<AvplsGpuParam>(&p));
}

MaterialsGpu::MaterialsGpu(std::vector<MATERIAL> const& materials)
{
	numMaterials = materials.size();
	thrust::host_vector<float3> emissive_h(numMaterials);
	thrust::host_vector<float3> diffuse_h(numMaterials);
	thrust::host_vector<float3> specular_h(numMaterials);	// incoming direction
	thrust::host_vector<float> exponent_h(numMaterials);

	for (int i = 0; i < materials.size(); ++i) {
		emissive_h[i] = make_float3(glm::vec3(materials[i].emissive));
		diffuse_h[i] = make_float3(glm::vec3(materials[i].diffuse));
		specular_h[i] = make_float3(glm::vec3(materials[i].specular));
		exponent_h[i] = materials[i].exponent;
	}

	emissive.resize(numMaterials);
	diffuse.resize(numMaterials);
	specular.resize(numMaterials);
	exponent.resize(numMaterials);

	thrust::copy(emissive_h.begin(), emissive_h.end(), emissive.begin());
	thrust::copy(diffuse_h.begin(), diffuse_h.end(), diffuse.begin());
	thrust::copy(specular_h.begin(), specular_h.end(), specular.begin());
	thrust::copy(exponent_h.begin(), exponent_h.end(), exponent.begin());

	MaterialsGpuParam p;
	p.emissive = thrust::raw_pointer_cast(&emissive[0]);
	p.diffuse = thrust::raw_pointer_cast(&diffuse[0]);
	p.specular = thrust::raw_pointer_cast(&specular[0]);
	p.exponent = thrust::raw_pointer_cast(&exponent[0]);
	p.numMaterials = numMaterials;
	param.reset(new cuda::CudaBuffer<MaterialsGpuParam>(&p));
}
