#ifndef CUDA_SURFACEOBJECT_H_
#define CUDA_SURFACEOBJECT_H_

#include "cudaUtil.hpp"
#include "cuda_runtime.h"

namespace cuda
{

class CudaSurfaceObject
{
public:
	CudaSurfaceObject(cudaResourceDesc desc)
	{
		CUDA_CALL(cudaCreateSurfaceObject(&m_surfaceObject, &desc));
	}

	~CudaSurfaceObject()
	{
		CUDA_CALL(cudaDestroySurfaceObject(m_surfaceObject));
	}

	cudaSurfaceObject_t getSurfaceObject() { return m_surfaceObject; }

private:
	cudaSurfaceObject_t m_surfaceObject;
};

}

#endif // CUDA_SURFACEOBJECT_H_
