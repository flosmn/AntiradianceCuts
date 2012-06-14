#include "COCLProgram.h"

#include "COCLContext.h"

#include "OCLUtil.h"

#include <iostream>

COCLProgram::COCLProgram(COCLContext* pContext, const std::string& debugName)
	: COCLResource(debugName), m_pContext(pContext)
{
}

COCLProgram::~COCLProgram()
{
	CheckNotInitialized("COCLProgram.~COCLProgram()");
}

bool COCLProgram::Init(const std::string& sourceFile)
{
	V_RET_FOF(m_pContext->CheckInitialized("COCLProgram.Init()"));

	cl_int err;
	std::string sourceData = ReadKernelFile(sourceFile);
	const char* data = sourceData.c_str();
	const size_t size = sourceData.size();
	m_Program = clCreateProgramWithSource(*m_pContext->GetCLContext(), 1, &data, &size, &err);

	V_RET_FOF(CHECK_CL_SUCCESS(err, "clCreateProgramWithSource"));

	if (clBuildProgram(m_Program, 1, m_pContext->GetCLDeviceId(), "", NULL, NULL) != CL_SUCCESS) {
		char buffer[10240];
		clGetProgramBuildInfo(m_Program, *m_pContext->GetCLDeviceId(), CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
		std::cout << "CL Compilation failed:\n%s" << buffer << std::endl;
		return false;
	}

	V_RET_FOF(CHECK_CL_SUCCESS(clUnloadCompiler(), "clUnloadCompiler"));

	V_RET_FOF(COCLResource::Init());

	return true;
}

void COCLProgram::Release()
{
	CHECK_CL_SUCCESS(clReleaseProgram(m_Program), "clReleaseProgram");

	COCLResource::Release();
}