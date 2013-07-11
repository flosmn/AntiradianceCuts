#ifndef CUDA_GATHER_H_
#define CUDA_GATHER_H_

#include "CudaResources/cudaGraphicsResource.hpp"
#include "CudaResources/cudaBuffer.hpp"

#include "OGLResources/COGLUniformBuffer.h"

#include "ObjectClouds.h"

#include "CMaterialBuffer.h"
#include "AVPL.h"
#include "data_gpu.h"

#include "bvh.h"

#include <vector>
#include <iostream>
#include <memory>

class CConfigManager;
class SceneProbe;

class CudaGather
{
public:
	CudaGather(int width, int height, 
		GLuint glPositonTexture, GLuint glNormalTexture,
		GLuint glRadianceOutputTexture, GLuint glAntiradianceOutputTexture,
		GLuint glResultOutputTexture,
		std::vector<MATERIAL> const& materials,
		COGLUniformBuffer* ubTransform,
		CConfigManager* confManager
	);

	~CudaGather();

	void gatherAntiradiance(std::vector<AVPL> const& avpls, glm::vec3 const& cameraPosition, float photonRadius);
	void run(std::vector<AVPL> const& avpls, glm::vec3 const& cameraPosition, 
		SceneProbe* sceneProbe, float sceneExtent, bool profile);
	
	PointCloud* getPointCloud() { return m_pointCloud.get(); }
	AABBCloud* getAABBCloud() { return m_aabbCloud.get(); }

private:
	std::unique_ptr<cuda::CudaGraphicsResource> m_positionResource;
	std::unique_ptr<cuda::CudaGraphicsResource> m_normalResource;
	std::unique_ptr<cuda::CudaGraphicsResource> m_radianceOutputResource;
	std::unique_ptr<cuda::CudaGraphicsResource> m_antiradianceOutputResource;
	std::unique_ptr<cuda::CudaGraphicsResource> m_resultOutputResource;
	
	// for lightcut debug
	std::unique_ptr<PointCloud> m_pointCloud;
	std::unique_ptr<AABBCloud> m_aabbCloud;

	std::unique_ptr<MaterialsGpu> m_materialsGpu;

	COGLUniformBuffer* m_ubTransform;
	CConfigManager* m_confManager;

	int m_width;
	int m_height;
};

#endif // CUDA_GATHER_H_
