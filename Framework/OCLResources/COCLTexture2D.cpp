#include "COCLTexture2D.h"

#include "CL/cl_gl.h"

#include "COCLContext.h"

#include "OCLUtil.h"

#include "..\OGLResources\COGLTexture2D.h"

#include "..\Defines.h"

COCLTexture2D::COCLTexture2D(COCLContext* pContext, const std::string& debugName)
	: COCLResource(debugName), m_pContext(pContext)
{
}

COCLTexture2D::~COCLTexture2D()
{
}

bool COCLTexture2D::Init(COGLTexture2D* pGLTexture)
{
	m_pGLTexture = pGLTexture;

	cl_int err;
	m_Texture = clCreateFromGLTexture2D(*m_pContext->GetCLContext(), CL_MEM_READ_WRITE,
		GL_TEXTURE_2D, 0, m_pGLTexture->GetResourceIdentifier(), &err);
	
	V_RET_FOF(CHECK_CL_SUCCESS(err, "clCreateFromGLTexture2D"));
	
	V_RET_FOF(COCLResource::Init());

	return true;
}

void COCLTexture2D::Release()
{
	clReleaseMemObject(m_Texture);

	COCLResource::Release();
}

void COCLTexture2D::Lock()
{
	glFlush();

	cl_int err = clEnqueueAcquireGLObjects(*m_pContext->GetCLCommandQueue(), 1, &m_Texture, 0, 0, 0);

	CHECK_CL_SUCCESS(err, "clEnqueueAcquireGLObjects()");
}

void COCLTexture2D::Unlock()
{
	cl_uint err;
	
	err = clEnqueueReleaseGLObjects(*m_pContext->GetCLCommandQueue(), 1, &m_Texture, 0, 0, 0);
	
	CHECK_CL_SUCCESS(err, "clEnqueueReleaseGLObjects()");

	err = clFlush(*m_pContext->GetCLCommandQueue());

	CHECK_CL_SUCCESS(err, "clFlush()");
}

