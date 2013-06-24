#ifndef CUDA_GRAPHICSRESOURCE_H_
#define CUDA_GRAPHICSRESOURCE_H_

#include "cudaUtil.hpp"
#include "cudaSurfaceObject.hpp"

#include "GL/glew.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include <memory>

namespace cuda
{

class CudaGraphicsResource
{
public:
	// creates and registers an opengl texture to cuda
	CudaGraphicsResource(GLuint image, GLenum target, unsigned int flags)
	{
		CUDA_CALL(cudaGraphicsGLRegisterImage(&m_resource, image, target, flags));
	}

	// creates and registers an opengl buffer to cuda
	CudaGraphicsResource(GLuint buffer, unsigned int flags)
	{
		CUDA_CALL(cudaGraphicsGLRegisterBuffer(&m_resource, buffer, flags));
	}

	~CudaGraphicsResource()
	{
		CUDA_CALL(cudaGraphicsUnregisterResource(m_resource));
	}

	cudaGraphicsResource_t getResource() const { return m_resource; }

private:
	cudaGraphicsResource_t m_resource;
};

class CudaGraphicsResourceMapped
{
public:
	CudaGraphicsResourceMapped(CudaGraphicsResource* resource)
		: m_resource(resource)
	{
		cudaGraphicsResource_t cudaGraphicsResource = m_resource->getResource();
		CUDA_CALL(cudaGraphicsMapResources(1, &cudaGraphicsResource));
	}

	virtual ~CudaGraphicsResourceMapped()
	{
		cudaGraphicsResource_t cudaGraphicsResource = m_resource->getResource();
		CUDA_CALL(cudaGraphicsUnmapResources(1, &cudaGraphicsResource));
	}

protected:
	CudaGraphicsResource* m_resource;
};

class CudaGraphicsResourceMappedArray : public CudaGraphicsResourceMapped
{
public:
	CudaGraphicsResourceMappedArray(CudaGraphicsResource* resource)
		: CudaGraphicsResourceMapped(resource)
	{
		memset(&m_resDesc, 0, sizeof(m_resDesc));
		m_resDesc.resType = cudaResourceTypeArray;

		CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&m_resDesc.res.array.array, 
			m_resource->getResource(), 0, 0));
	}

	cudaArray_t getCudaArray() { return m_resDesc.res.array.array; }

	cudaSurfaceObject_t getCudaSurfaceObject()
	{
		m_surfaceObject.reset(new CudaSurfaceObject(m_resDesc));
		return m_surfaceObject->getSurfaceObject();
	}

private:
	struct cudaResourceDesc m_resDesc;
	std::unique_ptr<CudaSurfaceObject> m_surfaceObject;
};

}

#endif // CUDA_GRAPHICSRESOURCE_H_
