#ifndef _C_RENDER_TARGET_H_
#define _C_RENDER_TARGET_H_

#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>

typedef unsigned int uint;

class COGLFrameBuffer;
class COGLTexture2D;

class CRenderTarget
{
	friend class CRenderTargetLock;

public:
	CRenderTarget();
	~CRenderTarget();
	
	bool Init(uint width, uint height, uint nBuffers, COGLTexture2D* pDepthBuffer);
	void Release();
		
	COGLFrameBuffer* GetFrameBuffer() { return m_pFrameBuffer; }

	COGLTexture2D* GetBuffer(uint i);
	COGLTexture2D* GetDepthBuffer();
		
private:
	void Bind();
	void Unbind();

	std::vector<COGLTexture2D*> m_vTargetTextures;
	COGLTexture2D* m_pDepthBuffer;
	
	COGLFrameBuffer* m_pFrameBuffer;
	
	uint m_nBuffers;
	GLenum* m_pBuffers;

	uint m_Width;
	uint m_Height;

	bool m_ExternalDepthBuffer;
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