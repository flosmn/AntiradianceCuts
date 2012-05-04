#include "CAccumulationBuffer.h"

#include "Macros.h"

#include "CGLResources\CGLTexture2D.h"
#include "CGLResources\CGLRenderBuffer.h"
#include "CGLResources\CGLFrameBuffer.h"

CAccumulationBuffer::CAccumulationBuffer()
	: m_pGLFBRenderTarget(0), m_pGLTAccumTexture(0), m_pDepthBuffer(0),
	  m_Width(0), m_Height(0), m_ExternalDepthBuffer(false)
{
	m_pGLFBRenderTarget = new CGLFrameBuffer("CAccumulationBuffer.m_pGLFBRenderTarget");
	m_pGLTAccumTexture = new CGLTexture2D("CAccumulationBuffer.m_pGLTAccumTexture");
}

CAccumulationBuffer::~CAccumulationBuffer() 
{
	if(!m_ExternalDepthBuffer)
		SAFE_DELETE(m_pDepthBuffer);

	SAFE_DELETE(m_pGLTAccumTexture);
	SAFE_DELETE(m_pGLFBRenderTarget);
}
	
bool CAccumulationBuffer::Init(GLuint width, GLuint height, CGLTexture2D* pDepthBuffer)
{
	if(pDepthBuffer != 0)
	{
		m_ExternalDepthBuffer = true;
		m_pDepthBuffer = pDepthBuffer;
	}
	else
	{
		m_pDepthBuffer = new CGLTexture2D("CAccumulationBuffer.m_pDepthBuffer");
		V_RET_FOF(m_pDepthBuffer->Init(width, height, GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, false));

	}

	m_Width = width;
	m_Height = height;
	
	V_RET_FOF(m_pGLTAccumTexture->Init(m_Width, m_Height, 
		GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));
	
	V_RET_FOF(m_pGLFBRenderTarget->Init());

	m_pGLFBRenderTarget->AttachTexture2D(m_pDepthBuffer, GL_DEPTH_ATTACHMENT);
	m_pGLFBRenderTarget->AttachTexture2D(m_pGLTAccumTexture, GL_COLOR_ATTACHMENT0);

	V_RET_FOF(m_pGLFBRenderTarget->CheckFrameBufferComplete());

	return true;
}

void CAccumulationBuffer::Release()
{
	if(!m_ExternalDepthBuffer)
		m_pDepthBuffer->Release();
	
	m_pGLTAccumTexture->Release();
	m_pGLFBRenderTarget->Release();
}