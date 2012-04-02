#include "CAccumulationBuffer.h"

#include "Macros.h"

#include "CGLResources\CGLTexture2D.h"
#include "CGLResources\CGLRenderBuffer.h"
#include "CGLResources\CGLFrameBuffer.h"

CAccumulationBuffer::CAccumulationBuffer()
	: m_pGLFBRenderTarget(0), m_pGLTAccumTexture(0), m_pGLRBDepthRenderBuffer(0),
	  m_Width(0), m_Height(0)
{
	m_pGLFBRenderTarget = new CGLFrameBuffer("CAccumulationBuffer.m_pGLFBRenderTarget");
	m_pGLTAccumTexture = new CGLTexture2D("CAccumulationBuffer.m_pGLTAccumTexture");
	m_pGLRBDepthRenderBuffer = new CGLRenderBuffer("CAccumulationBuffer.m_pGLRBDepthRenderBuffer");
}

CAccumulationBuffer::~CAccumulationBuffer() 
{
	SAFE_DELETE(m_pGLRBDepthRenderBuffer);
	SAFE_DELETE(m_pGLTAccumTexture);
	SAFE_DELETE(m_pGLFBRenderTarget);
}
	
bool CAccumulationBuffer::Init(GLuint width, GLuint height)
{
	m_Width = width;
	m_Height = height;
	
	V_RET_FOF(m_pGLTAccumTexture->Init(m_Width, m_Height, 
		GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));
	
	V_RET_FOF(m_pGLRBDepthRenderBuffer->Init(m_Width, m_Height, GL_DEPTH_COMPONENT));

	V_RET_FOF(m_pGLFBRenderTarget->Init());

	m_pGLFBRenderTarget->AttachRenderBuffer(m_pGLRBDepthRenderBuffer, GL_DEPTH_ATTACHMENT);
	m_pGLFBRenderTarget->AttachTexture2D(m_pGLTAccumTexture, GL_COLOR_ATTACHMENT0);

	V_RET_FOF(m_pGLFBRenderTarget->CheckFrameBufferComplete());

	return true;
}

void CAccumulationBuffer::Release()
{
	m_pGLTAccumTexture->Release();
	m_pGLRBDepthRenderBuffer->Release();
	m_pGLFBRenderTarget->Release();
}