#include "CGLFrameBuffer.h"

#include "CGLRenderBuffer.h"
#include "CGLTexture2D.h"

#include "..\Macros.h"

#include "..\CUtils\GLErrorUtil.h"

#include <assert.h>
#include <iostream>


CGLFrameBuffer::CGLFrameBuffer(std::string debugName)
	: CGLResource(CGL_FRAMEBUFFER, debugName)
{

}

CGLFrameBuffer::~CGLFrameBuffer()
{
	CGLResource::~CGLResource();
}

bool CGLFrameBuffer::Init()
{
	V_RET_FOF(CGLResource::Init());

	glGenFramebuffers(1, &m_Resource);
	
	V_RET_FOT(CheckGLError(m_DebugName, "CGLFrameBuffer::Init()"));

	return true;
}

void CGLFrameBuffer::Release()
{
	CGLResource::Release();
	
	glDeleteFramebuffers(1, &m_Resource);
	
	CheckGLError(m_DebugName, "CGLFrameBuffer::Release()");
}

void CGLFrameBuffer::AttachRenderBuffer(CGLRenderBuffer* renderBuffer, 
	GLenum attachmentSlot)
{
	CGLBindLock lock(this, CGL_FRAMEBUFFER_SLOT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachmentSlot, GL_RENDERBUFFER, 
		renderBuffer->GetResourceIdentifier());

	CheckGLError(m_DebugName, "CGLFrameBuffer::AttachRenderBuffer()");
}

void CGLFrameBuffer::AttachTexture2D(CGLTexture2D* texture, 
	GLenum attachmentSlot)
{
	CGLBindLock lock(this, CGL_FRAMEBUFFER_SLOT);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachmentSlot, GL_TEXTURE_2D, 
		texture->GetResourceIdentifier(), 0);

	CheckGLError(m_DebugName, "CGLFrameBuffer::AttachTexture2D()");
}

void CGLFrameBuffer::Bind(CGLBindSlot slot)
{
	CGLResource::Bind(slot);

	assert(m_Slot == CGL_FRAMEBUFFER_SLOT);

	glBindFramebuffer(GetGLSlot(m_Slot), m_Resource);

	CheckGLError(m_DebugName, "CGLFrameBuffer::Bind()");
}

void CGLFrameBuffer::Unbind()
{
	CGLResource::Unbind();

	assert(m_Slot == CGL_FRAMEBUFFER_SLOT);

	glBindFramebuffer(GetGLSlot(m_Slot), 0);

	CheckGLError(m_DebugName, "CGLFrameBuffer::Unbind()");
}

bool CGLFrameBuffer::CheckFrameBufferComplete()
{
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(status != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << "Framebuffer status not complete." << std::endl;
		return false;
	}

	return true;
}
