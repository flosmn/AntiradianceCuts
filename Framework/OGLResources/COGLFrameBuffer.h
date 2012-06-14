#ifndef _C_GL_FRAME_BUFFER_H_
#define _C_GL_FRAME_BUFFER_H_

#include "COGLResource.h"

class COGLRenderBuffer;
class COGLTexture2D;
class COGLRenderTargetConfig;

class COGLFrameBuffer : public COGLResource
{
	friend class CRenderTarget;
	friend class COGLRenderTargetLock;

public:
	COGLFrameBuffer(std::string debugName);
	~COGLFrameBuffer();

	virtual bool Init();
	virtual void Release();

	void AttachRenderBuffer(COGLRenderBuffer* renderBuffer, GLenum attachmentSlot);
	void AttachTexture2D(COGLTexture2D* texture, GLenum attachmentSlot);
	
	bool CheckFrameBufferComplete();

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();
};

class COGLRenderTargetLock
{
public:
	COGLRenderTargetLock(COGLFrameBuffer* pFrameBuffer, GLuint nBuffers, GLenum* pBuffers)
		: m_pFrameBuffer(pFrameBuffer)
	{
		m_pFrameBuffer->Bind(COGL_FRAMEBUFFER_SLOT);
		glDrawBuffers(nBuffers, pBuffers);
	}

	~COGLRenderTargetLock()
	{
		m_pFrameBuffer->Unbind();
		glDrawBuffer(GL_BACK);
	}

private:
	COGLFrameBuffer* m_pFrameBuffer;

	// noncopyable
	COGLRenderTargetLock(const COGLRenderTargetLock&);
	COGLRenderTargetLock& operator=(const COGLRenderTargetLock&);
};

#endif // _C_GL_FRAME_BUFFER_H_