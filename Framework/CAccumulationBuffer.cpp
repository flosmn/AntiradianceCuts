#include "CAccumulationBuffer.h"

#include "Macros.h"

#include "OGLResources\COGLTexture2D.h"
#include "OGLResources\COGLRenderBuffer.h"
#include "OGLResources\COGLFrameBuffer.h"

CAccumulationBuffer::CAccumulationBuffer(GLuint width, GLuint height, COGLTexture2D* pDepthBuffer)
	: m_Width(width), m_Height(height), m_ExternalDepthBuffer(false)
{
	m_renderTarget.reset(new COGLFrameBuffer("CAccumulationBuffer.m_pGLFBRenderTarget"));
	m_accumTexture.reset(new COGLTexture2D(m_Width, m_Height, GL_RGBA16F, GL_RGBA, 
		GL_FLOAT, 1, false, "CAccumulationBuffer.m_pGLTAccumTexture"));

	if (pDepthBuffer == 0)
	{
		m_ExternalDepthBuffer = false;
		m_depthBuffer.reset(new COGLTexture2D(m_Width, m_Height, GL_DEPTH_COMPONENT32F, 
			GL_DEPTH_COMPONENT, GL_FLOAT, 1, false, "CAccumulationBuffer.m_pDepthBuffer"));
		m_renderTarget->AttachTexture2D(m_depthBuffer.get(), GL_DEPTH_ATTACHMENT);
	}
	else
	{
		m_ExternalDepthBuffer = true;
		m_renderTarget->AttachTexture2D(pDepthBuffer, GL_DEPTH_ATTACHMENT);
	}
	
	m_renderTarget->AttachTexture2D(m_accumTexture.get(), GL_COLOR_ATTACHMENT0);
	m_renderTarget->CheckFrameBufferComplete();
}

CAccumulationBuffer::~CAccumulationBuffer() 
{
}