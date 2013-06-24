#ifndef CUDA_TIMER_H_
#define CUDA_TIMER_H_

#include "cudaUtil.hpp"
#include "cuda_runtime.h"

namespace cuda
{

class CudaTimer
{
public:
	CudaTimer() 
	{
		CUDA_CALL(cudaEventCreate(&m_start));
		CUDA_CALL(cudaEventCreate(&m_stop));
	}
	
	~CudaTimer()
	{
		CUDA_CALL(cudaEventDestroy(m_start));
		CUDA_CALL(cudaEventDestroy(m_stop));
	}

	void start()
	{
		CUDA_CALL(cudaEventRecord(m_start));
	}

	void stop()
	{
		CUDA_CALL(cudaEventRecord(m_stop));
	}

	// returns the elapsed time in milliseconds
	float getTime()
	{
		float time = 0.f;

		CUDA_CALL(cudaEventSynchronize(m_stop));
		CUDA_CALL(cudaEventElapsedTime(&time, m_start, m_stop));
		
		return time;
	}

private:
	cudaEvent_t m_start;
	cudaEvent_t m_stop;
};

}

#endif // CUDA_TIMER_H_
