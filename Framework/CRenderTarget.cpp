#include "CRenderTarget.h"

#include "Macros.h"

#include "CGLResources\CGLFrameBuffer.h"
#include "CGLResources\CGLTexture2D.h"

#include <sstream>

CRenderTarget::CRenderTarget()
	: m_pFrameBuffer(0), m_Width(0), m_Height(0), m_ExternalDepthBuffer(false)
{
	m_pFrameBuffer = new CGLFrameBuffer("CRenderTarget.m_pFrameBuffer");
}

CRenderTarget::~CRenderTarget()
{
	SAFE_DELETE(m_pFrameBuffer);
}
	

bool CRenderTarget::Init(uint width, uint height, uint nBuffers, CGLTexture2D* pDepthBuffer)
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

	m_nBuffers = nBuffers;
	m_pBuffers = new GLenum[nBuffers];
	
	m_Width = width;
	m_Height = height;
	
	V_RET_FOF(m_pFrameBuffer->Init());
	m_pFrameBuffer->AttachTexture2D(m_pDepthBuffer, GL_DEPTH_ATTACHMENT);

	for(uint i = 0; i < m_nBuffers; ++i)
	{
		std::stringstream ss;
		ss << "CRenderTarget.buffer" << i;
		CGLTexture2D* buffer = new CGLTexture2D(ss.str());
		V_RET_FOF(buffer->Init(m_Width, m_Height, GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));
		m_vTargetTextures.push_back(buffer);
		m_pFrameBuffer->AttachTexture2D(buffer, GL_COLOR_ATTACHMENT0 + i);
		m_pBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
	}
	
	V_RET_FOF(m_pFrameBuffer->CheckFrameBufferComplete());

	return true;
}


void CRenderTarget::Release()
{
	if(!m_ExternalDepthBuffer)
	{
		m_pDepthBuffer->Release();
		SAFE_DELETE(m_pDepthBuffer);
	}
	
	for(uint i = 0; i < m_nBuffers; ++i)
	{
		m_vTargetTextures[i]->Release();
		SAFE_DELETE(m_vTargetTextures[i]);
	}

	m_pFrameBuffer->Release();

	delete [] m_pBuffers;
}

CGLTexture2D* CRenderTarget::GetBuffer(uint i)
{
	if(i >= m_nBuffers)
	{
		std::cout << "CRenderTarget.GetBuffer(): index to large" << std::endl;
		return 0;
	}

	return m_vTargetTextures[i];
}

void CRenderTarget::Bind()
{
	m_pFrameBuffer->Bind(CGL_FRAMEBUFFER_SLOT);

	glDrawBuffers(m_nBuffers, m_pBuffers);
}

void CRenderTarget::Unbind()
{
	m_pFrameBuffer->Unbind();

	glDrawBuffer(GL_BACK);
}