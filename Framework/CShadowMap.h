#ifndef _C_SHADOW_MAP_H_
#define _C_SHADOW_MAP_H_

typedef unsigned int uint; 

#include "GL/glew.h"

#include <glm/gtc/type_ptr.hpp>

#include "Macros.h"

#include "OGLResources\COGLTexture2D.h"
#include "OGLResources\COGLFrameBuffer.h"

#include <iostream>
#include <memory>

class CShadowMap 
{
public:
	CShadowMap(uint size) : m_shadowMapSize(size) 
	{
		m_renderTarget.reset(new COGLFrameBuffer("CShadowMap.m_pRenderTarget"));
		m_shadowMapTexture.reset(new COGLTexture2D(m_shadowMapSize, m_shadowMapSize, GL_DEPTH_COMPONENT24, 
			GL_DEPTH_COMPONENT, GL_FLOAT, 1, false, "CShadowMap.m_pShadowMapTexture"));

		m_renderTarget->AttachTexture2D(m_shadowMapTexture.get(), GL_DEPTH_ATTACHMENT);
		m_renderTarget->CheckFrameBufferComplete();
	};

	~CShadowMap() {}

	COGLTexture2D* GetShadowMapTexture() { return m_shadowMapTexture.get(); }
	COGLFrameBuffer* GetRenderTarget() { return m_renderTarget.get(); }

	uint GetShadowMapSize() { return m_shadowMapSize; }

private:
	std::unique_ptr<COGLFrameBuffer> m_renderTarget;
	std::unique_ptr<COGLTexture2D> m_shadowMapTexture;

	uint m_shadowMapSize;
};

#endif // _C_SHADOW_MAP_H_
