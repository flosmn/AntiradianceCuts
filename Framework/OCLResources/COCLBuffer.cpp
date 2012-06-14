#include "COCLBuffer.h"

#include "COCLContext.h"

#include "OCLUtil.h"

COCLBuffer::COCLBuffer(COCLContext* pContext, const std::string& debugName)
	: COCLResource(debugName), m_pContext(pContext)
{
}

COCLBuffer::~COCLBuffer()
{
	CheckNotInitialized("COCLBuffer.~COCLBuffer()");
}

bool COCLBuffer::Init(size_t size, cl_mem_flags flags)
{
	V_RET_FOF(m_pContext->CheckInitialized("COCLBuffer.Init()"));

	m_Size = size;

	cl_int err;
	m_Buffer = clCreateBuffer(*m_pContext->GetCLContext(), flags, m_Size, NULL, &err);

	V_RET_FOF(CHECK_CL_SUCCESS(err, "clCreateBuffer()"));

	V_RET_FOF(COCLResource::Init());

	return true;
}

void COCLBuffer::Release()
{
	CHECK_CL_SUCCESS(clReleaseMemObject(m_Buffer), "clReleaseMemObject()");

	COCLResource::Release();
}

void COCLBuffer::SetBufferData(void* pData, bool blocking)
{
	CheckInitialized("COCLBuffer.SetBufferData()");
	
	CHECK_CL_SUCCESS(clEnqueueWriteBuffer(*m_pContext->GetCLCommandQueue(), m_Buffer, blocking, 0, m_Size, pData, 0, NULL, NULL), "clEnqueueWriteBuffer()");
}

void COCLBuffer::GetBufferData(void* pData, bool blocking)
{
	CheckInitialized("COCLBuffer.GetBufferData()");

	CHECK_CL_SUCCESS(clEnqueueReadBuffer(*m_pContext->GetCLCommandQueue(), m_Buffer, blocking, 0, m_Size, pData, 0, NULL, NULL), "clEnqueueReadBuffer()");	
}