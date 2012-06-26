#include "COCLKernel.h"

#include "OCLUtil.h"

#include "COCLContext.h"
#include "COCLProgram.h"

#include <iostream>

COCLKernel::COCLKernel(COCLContext* pContext, COCLProgram* pProgram, const std::string& debugName)
	: COCLResource(debugName), m_pContext(pContext), m_pProgram(pProgram)
{
}

COCLKernel::~COCLKernel()
{
	CheckNotInitialized("COCLKernel.~COCLKernel()");
}

bool COCLKernel::Init(const std::string& kernelName)
{
	cl_int err;

	V_RET_FOF(m_pProgram->CheckInitialized("COCLKernel.Init()"));

	m_Kernel = clCreateKernel(*m_pProgram->GetCLProgram(), kernelName.c_str(), &err);

	V_RET_FOF(CHECK_CL_SUCCESS(err, "clCreateKernel()"));

	V_RET_FOF(COCLResource::Init());

	return true;
}

void COCLKernel::Release()
{
	COCLResource::Release();
}

void COCLKernel::SetKernelArg(uint slot, size_t size, const void* value)
{
	CheckInitialized("COCLKernel.SetKernelArg()");

	CHECK_CL_SUCCESS(clSetKernelArg(m_Kernel, slot, size, value), "clSetKernelArg()");
}

void COCLKernel::CallKernel(uint work_dim, size_t* global_work_offset, size_t* global_work_size, size_t* local_work_size)
{
	CheckInitialized("COCLKernel.CallKernel()");

	m_pContext->CheckInitialized("COCLKernel.CallKernel()");
	
	//cl_event profileEvent;
	//cl_ulong startTime, endTime;

	cl_int err = clEnqueueNDRangeKernel(*m_pContext->GetCLCommandQueue(), m_Kernel,
		work_dim, global_work_offset, global_work_size, local_work_size, NULL, NULL, NULL);

	CHECK_CL_SUCCESS(err, "clEnqueueNDRangeKernel()");
	
	err = clFinish(*m_pContext->GetCLCommandQueue());
	
	CHECK_CL_SUCCESS(err, "clFinish()");
	
	//clGetEventProfilingInfo(profileEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
	//clGetEventProfilingInfo(profileEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
	//cl_ulong kernelExecTimeNs = endTime-startTime;

	//clReleaseEvent(profileEvent);

	//std::cout << "Kernel " << m_DebugName << " execution time: " << kernelExecTimeNs * 1.f/1000000.f << "ms." << std::endl;
}