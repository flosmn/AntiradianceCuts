#ifndef _C_RENDER_TARGET_H_
#define _C_RENDER_TARGET_H_

#include <vector>
#include <memory>

#include <GL/glew.h>
#include <GL/gl.h>

typedef unsigned int uint;

class COGLFrameBuffer;
class COGLTexture2D;

class CRenderTarget
{
	friend class CRenderTargetLock;

public:
	CRenderTarget(uint width, uint height, uint numTargets, COGLTexture2D* pDepthBuffer = 0);
	~CRenderTarget();
	
	COGLFrameBuffer* GetFrameBuffer() { return m_frameBuffer.get(); }

	COGLTexture2D* GetTarget(uint i);
	COGLTexture2D* GetDepthBuffer();
		
private:
	void Bind();
	void Unbind();

	std::vector<std::unique_ptr<COGLTexture2D>> m_targetTextures;

	std::unique_ptr<COGLTexture2D> m_depthBuffer;
	COGLTexture2D* m_externalDepthBuffer;
	
	std::unique_ptr<COGLFrameBuffer> m_frameBuffer;
	
	uint m_numTargets;
	std::vector<GLenum> m_targets;

	uint m_Width;
	uint m_Height;
};

class CRenderTargetLock
{
public:
	CRenderTargetLock(CRenderTarget* pRenderTarget)
		: m_pRenderTarget(pRenderTarget)
	{
		m_pRenderTarget->Bind();
	}

	~CRenderTargetLock()
	{
		m_pRenderTarget->Unbind();
	}

private:
	CRenderTarget* m_pRenderTarget;

	// noncopyable
	CRenderTargetLock(const CRenderTargetLock&);
	CRenderTargetLock& operator=(const CRenderTargetLock&);
};

#endif // _C_RENDER_TARGET_H_
