#ifndef _C_GL_FRAME_BUFFER_H_
#define _C_GL_FRAME_BUFFER_H_

#include "CGLResource.h"

class CGLRenderBuffer;
class CGLTexture2D;
class CGLRenderTargetConfig;

class CGLFrameBuffer : public CGLResource
{
	friend class CGLRenderTargetLock;

public:
	CGLFrameBuffer(std::string debugName);
	~CGLFrameBuffer();

	virtual bool Init();
	virtual void Release();

	void AttachRenderBuffer(CGLRenderBuffer* renderBuffer, GLenum attachmentSlot);
	void AttachTexture2D(CGLTexture2D* texture, GLenum attachmentSlot);
	
	bool CheckFrameBufferComplete();

private:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();
};

class CGLRenderTargetLock
{
public:
	CGLRenderTargetLock(CGLFrameBuffer* pFrameBuffer, GLuint nBuffers, GLenum* pBuffers)
		: m_pFrameBuffer(pFrameBuffer)
	{
		m_pFrameBuffer->Bind(CGL_FRAMEBUFFER_SLOT);
		glDrawBuffers(nBuffers, pBuffers);
	}

	~CGLRenderTargetLock()
	{
		m_pFrameBuffer->Unbind();
		glDrawBuffer(GL_BACK);
	}

private:
	CGLFrameBuffer* m_pFrameBuffer;

	// noncopyable
	CGLRenderTargetLock(const CGLRenderTargetLock&);
	CGLRenderTargetLock& operator=(const CGLRenderTargetLock&);
};

#endif // _C_GL_FRAME_BUFFER_H_