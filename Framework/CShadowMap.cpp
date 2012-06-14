#include "CShadowMap.h"

#include <glm/gtc/type_ptr.hpp>

#include "Macros.h"

#include "OGLResources\COGLTexture2D.h"
#include "OGLResources\COGLFrameBuffer.h"

#include <iostream>

CShadowMap::CShadowMap() 
{
	m_pRenderTarget = new COGLFrameBuffer("CShadowMap.m_pRenderTarget");
	m_pShadowMapTexture = new COGLTexture2D("CShadowMap.m_pShadowMapTexture");
}

CShadowMap::~CShadowMap() 
{
	SAFE_DELETE(m_pRenderTarget);
	SAFE_DELETE(m_pShadowMapTexture);
}

bool CShadowMap::Init(uint size)
{
	m_ShadowMapSize = size;

	V_RET_FOF(m_pShadowMapTexture->Init(m_ShadowMapSize, m_ShadowMapSize, GL_DEPTH_COMPONENT24, 
		GL_DEPTH_COMPONENT, GL_FLOAT, 1, false));

	V_RET_FOF(m_pRenderTarget->Init());

	m_pRenderTarget->AttachTexture2D(m_pShadowMapTexture, GL_DEPTH_ATTACHMENT);

	V_RET_FOF(m_pRenderTarget->CheckFrameBufferComplete());

	return true;
}

void CShadowMap::Release()
{
	m_pRenderTarget->Release();
	m_pShadowMapTexture->Release();
}
