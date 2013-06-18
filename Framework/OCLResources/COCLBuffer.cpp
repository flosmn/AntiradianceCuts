#include "COCLBuffer.h"

#include "CL\cl_gl.h"

#include "COCLContext.h"

#include "..\OGLResources\COGLTextureBuffer.h"

#include "OCLUtil.h"

COCLBuffer::COCLBuffer(COCLContext* pContext, const std::string& debugName)
	: COCLResource(debugName), m_pContext(pContext), m_pOGLTextureBuffer(nullptr)
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

bool COCLBuffer::Init(COGLTextureBuffer* pGLTextureBuffer, cl_mem_flags flags)
{
	V_RET_FOF(m_pContext->CheckInitialized("COCLBuffer.Init()"));

	m_pOGLTextureBuffer = pGLTextureBuffer;

	cl_int err;
	m_Size = pGLTextureBuffer->GetSize();
	m_Buffer = clCreateFromGLBuffer(*m_pContext->GetCLContext(), flags, m_pOGLTextureBuffer->GetResourceIdentifier(), &err);

	V_RET_FOF(CHECK_CL_SUCCESS(err, "clCreateFromGLBuffer()"));

	V_RET_FOF(COCLResource::Init());

	return true;
}

void COCLBuffer::Release()
{
	CHECK_CL_SUCCESS(clReleaseMemObject(m_Buffer), "clReleaseMemObject()");

	COCLResource::Release();
}

void COCLBuffer::SetBufferData(void* pData, size_t size, bool blocking)
{
	CheckInitialized("COCLBuffer.SetBufferData()");
		
	if(size > m_Size)
	{
		std::cout << "COCLBuffer::SetBufferData(): Warning: size of data greater than buffer size." << std::endl;
		size = std::min(size, m_Size);
	}	
	
	CHECK_CL_SUCCESS(clEnqueueWriteBuffer(*m_pContext->GetCLCommandQueue(), m_Buffer, blocking, 0, size, pData, 0, NULL, NULL), "clEnqueueWriteBuffer()");
}

void COCLBuffer::GetBufferData(void* pData, size_t size, bool blocking)
{
	CheckInitialized("COCLBuffer.GetBufferData()");

	if(size > m_Size)
	{
		std::cout << "COCLBuffer::GetBufferData(): Warning: size of data greater than buffer size." << std::endl;
		size = std::min(size, m_Size);
	}	

	CHECK_CL_SUCCESS(clEnqueueReadBuffer(*m_pContext->GetCLCommandQueue(), m_Buffer, blocking, 0, size, pData, 0, NULL, NULL), "clEnqueueReadBuffer()");	
}

void COCLBuffer::Lock()
{
	if(m_pOGLTextureBuffer)
	{
		glFlush();

		cl_int err = clEnqueueAcquireGLObjects(*m_pContext->GetCLCommandQueue(), 1, &m_Buffer, 0, 0, 0);

		CHECK_CL_SUCCESS(err, "clEnqueueAcquireGLObjects()");
	}
	else
	{
		std::cout << "COCLBuffer.Lock: Warning: No OGLBuffer attached to this OCLBuffer." << std::endl;
	}
}

void COCLBuffer::Unlock()
{
	if(m_pOGLTextureBuffer)
	{
		cl_uint err;
		
		err = clEnqueueReleaseGLObjects(*m_pContext->GetCLCommandQueue(), 1, &m_Buffer, 0, 0, 0);
		
		CHECK_CL_SUCCESS(err, "clEnqueueReleaseGLObjects()");

		err = clFlush(*m_pContext->GetCLCommandQueue());

		CHECK_CL_SUCCESS(err, "clFlush()");
	}
	else
	{
		std::cout << "COCLBuffer.Lock: Warning: No OGLBuffer attached to this OCLBuffer." << std::endl;
	}
}