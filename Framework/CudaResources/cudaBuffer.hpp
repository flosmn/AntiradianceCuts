#ifndef CUDA_BUFFER_H_
#define CUDA_BUFFER_H_

#include "cudaUtil.hpp"

#include <vector>

namespace cuda
{

template <typename T>
class CudaBuffer
{
public:
	CudaBuffer(std::vector<T> const& data)
	{
		m_numElements = data.size();
		m_size = m_numElements * sizeof(T);
		CUDA_CALL(cudaMalloc(&m_devPtr, m_size));
		CUDA_CALL(cudaMemcpy(m_devPtr, data.data(), m_size, cudaMemcpyHostToDevice));
	}

	~CudaBuffer()
	{
		CUDA_CALL(cudaFree(m_devPtr));
	}

	void getContent(std::vector<T>& data)
	{
		data.resize(m_numElements);
		CUDA_CALL(cudaMemcpy(data.data(), m_devPtr, m_size, cudaMemcpyDeviceToHost));
	}

	T* getDevicePtr() { return reinterpret_cast<T*>(m_devPtr); }

private:
	void* m_devPtr;
	size_t m_size;
	size_t m_numElements;
};

}

#endif // CUDA_BUFFER_H_
