#ifndef _C_OCL_KERNEL_H_
#define _C_OCL_KERNEL_H_

typedef unsigned int uint;

#include "CL/cl.h"

#include "COCLResource.h"

class COCLContext;
class COCLProgram;

#include <string>

class COCLKernel : public COCLResource
{
public:
	COCLKernel(COCLContext* pContext, COCLProgram* pProgram, const std::string& debugName);
	~COCLKernel();

	virtual bool Init(const std::string& kernelName);
	virtual void Release();

	void SetKernelArg(uint slot, size_t size, const void* value);

	void CallKernel(uint work_dim, size_t* global_work_offset, size_t* global_work_size, size_t* local_work_size);

private:
	cl_kernel m_Kernel;

	COCLContext* m_pContext;
	COCLProgram* m_pProgram;
};

#endif // _C_OCL_KERNEL_H_