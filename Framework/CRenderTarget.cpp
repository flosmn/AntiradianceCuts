#include "CRenderTarget.h"

#include "Macros.h"

#include "OGLResources\COGLFrameBuffer.h"
#include "OGLResources\COGLTexture2D.h"

#include <sstream>

CRenderTarget::CRenderTarget(uint width, uint height, uint numTargets, COGLTexture2D* pDepthBuffer)
	: m_Width(width), m_Height(height), m_numTargets(numTargets), m_externalDepthBuffer(pDepthBuffer)
{
	m_frameBuffer.reset(new COGLFrameBuffer("CRenderTarget.m_pFrameBuffer"));

	m_targets.resize(m_numTargets);
	
	if(m_externalDepthBuffer == 0) {
		m_depthBuffer.reset(new COGLTexture2D(m_Width, m_Height, GL_DEPTH_COMPONENT32F, 
			GL_DEPTH_COMPONENT, GL_FLOAT, 1, false, "CAccumulationBuffer.m_pDepthBuffer"));
		m_frameBuffer->AttachTexture2D(m_depthBuffer.get(), GL_DEPTH_ATTACHMENT);
	} else {
		m_frameBuffer->AttachTexture2D(m_externalDepthBuffer, GL_DEPTH_ATTACHMENT);
	}

	for(uint i = 0; i < m_numTargets; ++i)
	{
		m_targetTextures.emplace_back(std::unique_ptr<COGLTexture2D>(new COGLTexture2D(
			m_Width, m_Height, GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false)));
		m_frameBuffer->AttachTexture2D(m_targetTextures.back().get(), GL_COLOR_ATTACHMENT0 + i);
		m_targets[i] = GL_COLOR_ATTACHMENT0 + i;
	}
	
	m_frameBuffer->CheckFrameBufferComplete();
}

CRenderTarget::~CRenderTarget()
{
}

COGLTexture2D* CRenderTarget::GetDepthBuffer()
{
	if (m_externalDepthBuffer == 0) {
		return m_depthBuffer.get();
	} else {
		return m_externalDepthBuffer;
	}
}

COGLTexture2D* CRenderTarget::GetTarget(uint i)
{
	if(i >= m_targetTextures.size())
	{
		std::cout << "CRenderTarget.GetTargets(): index to large" << std::endl;
		return 0;
	}

	return m_targetTextures[i].get();
}

void CRenderTarget::Bind()
{
	m_frameBuffer->Bind(COGL_FRAMEBUFFER_SLOT);

	glDrawBuffers(m_numTargets, m_targets.data());
}

void CRenderTarget::Unbind()
{
	m_frameBuffer->Unbind();

	glDrawBuffer(GL_BACK);
}
