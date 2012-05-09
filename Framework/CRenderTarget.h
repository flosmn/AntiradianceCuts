#ifndef _C_RENDER_TARGET_H_
#define _C_RENDER_TARGET_H_

#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>

typedef unsigned int uint;

class CGLFrameBuffer;
class CGLTexture2D;

class CRenderTarget
{
	friend class CRenderTargetLock;

public:
	CRenderTarget();
	~CRenderTarget();
	
	bool Init(uint width, uint height, uint nBuffers, CGLTexture2D* pDepthBuffer);
	void Release();
		
	CGLFrameBuffer* GetFrameBuffer() { return m_pFrameBuffer; }

	CGLTexture2D* GetBuffer(uint i);
	CGLTexture2D* GetDepthBuffer();
		
private:
	void Bind();
	void Unbind();

	std::vector<CGLTexture2D*> m_vTargetTextures;
	CGLTexture2D* m_pDepthBuffer;
	
	CGLFrameBuffer* m_pFrameBuffer;
	
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