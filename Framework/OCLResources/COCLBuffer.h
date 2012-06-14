#ifndef _C_OCL_BUFFER_H_
#define _C_OCL_BUFFER_H_

#include "COCLResource.h"

#include "CL\cl.h"

class COCLContext;

#include <string>

class COCLBuffer : public COCLResource
{
public:
	COCLBuffer(COCLContext* pContext, const std::string& debugName);
	~COCLBuffer();

	virtual bool Init(size_t size, cl_mem_flags flags);
	virtual void Release();

	void SetBufferData(void* pData, bool blocking);
	void GetBufferData(void* pData, bool blocking);

	const cl_mem* GetCLBuffer() { CheckInitialized("COCLBuffer.GetCLBuffer()"); return &m_Buffer; }

private:
	cl_mem m_Buffer;
	size_t m_Size;

	COCLContext* m_pContext;
};

#endif // _C_OCL_BUFFER_H_