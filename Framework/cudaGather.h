#ifndef CUDA_GATHER_H_
#define CUDA_GATHER_H_

#include "CudaResources/cudaGraphicsResource.hpp"
#include "CudaResources/cudaBuffer.hpp"

#include "CMaterialBuffer.h"
#include "AVPL.h"

#include <vector>
#include <iostream>
#include <memory>

class CudaGather
{
public:
	CudaGather(int width, int height, 
		GLuint glPositonTexture, GLuint glNormalTexture,
		GLuint glRadianceOutputTexture, GLuint glAntiradianceOutputTexture,
		GLuint glResultOutputTexture,
		std::vector<MATERIAL> const& materials
	);

	~CudaGather();

	void run(std::vector<AVPL> const& avpls, glm::vec3 const& cameraPosition);

private:
	std::unique_ptr<cuda::CudaGraphicsResource> m_positionResource;
	std::unique_ptr<cuda::CudaGraphicsResource> m_normalResource;
	std::unique_ptr<cuda::CudaGraphicsResource> m_radianceOutputResource;
	std::unique_ptr<cuda::CudaGraphicsResource> m_antiradianceOutputResource;
	std::unique_ptr<cuda::CudaGraphicsResource> m_resultOutputResource;

	std::unique_ptr<cuda::CudaBuffer<NEW_AVPL>> m_avpls;
	std::unique_ptr<cuda::CudaBuffer<MATERIAL>> m_materials;

	int m_width;
	int m_height;
};

#endif // CUDA_GATHER_H_
