#ifndef CUDA_CONTEXT_H_
#define CUDA_CONTEXT_H_

#include "cuda_runtime.h"
#include "cudaUtil.hpp"

#include <iostream>

namespace cuda
{

class CudaContext
{
public:
	CudaContext()
	{
		memset(&m_bestCudaDeviceProp, 0, sizeof(m_bestCudaDeviceProp));

		int numCudaDevices = 0;
		cudaGetDeviceCount(&numCudaDevices);

		if (numCudaDevices == 0) {
			std::cout << "No cuda device found." << std::endl;
			return;
		}

		for (int i = 0; i < numCudaDevices; ++i) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);

			if (m_bestCudaDeviceProp.major < prop.major || 
					((m_bestCudaDeviceProp.major == prop.major) &&
					 (m_bestCudaDeviceProp.minor < prop.minor)))
			{
				m_bestCudaDevice = i;
				m_bestCudaDeviceProp = prop;
			}
		}

		std::cout << "Properties of chosen cuda device" << std::endl;
		std::cout << "   Compute Capability: " << m_bestCudaDeviceProp.major 
			<< "." << m_bestCudaDeviceProp.minor << std::endl;

		CUDA_CALL(cudaSetDevice(m_bestCudaDevice));
	}

	~CudaContext()
	{
	}

private:
	int m_bestCudaDevice;
	cudaDeviceProp m_bestCudaDeviceProp;
};

}

#endif // CUDA_CONTEXT_H_
