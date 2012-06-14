#include "COGLFrameBuffer.h"

#include "COGLRenderBuffer.h"
#include "COGLTexture2D.h"

#include "..\Macros.h"

#include "..\Utils\GLErrorUtil.h"

#include <assert.h>
#include <iostream>


COGLFrameBuffer::COGLFrameBuffer(std::string debugName)
	: COGLResource(COGL_FRAMEBUFFER, debugName)
{

}

COGLFrameBuffer::~COGLFrameBuffer()
{
	COGLResource::~COGLResource();
}

bool COGLFrameBuffer::Init()
{
	V_RET_FOF(COGLResource::Init());

	glGenFramebuffers(1, &m_Resource);
	
	V_RET_FOT(CheckGLError(m_DebugName, "COGLFrameBuffer::Init()"));

	return true;
}

void COGLFrameBuffer::Release()
{
	COGLResource::Release();
	
	glDeleteFramebuffers(1, &m_Resource);
	
	CheckGLError(m_DebugName, "COGLFrameBuffer::Release()");
}

void COGLFrameBuffer::AttachRenderBuffer(COGLRenderBuffer* renderBuffer, 
	GLenum attachmentSlot)
{
	COGLBindLock lock(this, COGL_FRAMEBUFFER_SLOT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachmentSlot, GL_RENDERBUFFER, 
		renderBuffer->GetResourceIdentifier());

	CheckGLError(m_DebugName, "COGLFrameBuffer::AttachRenderBuffer()");
}

void COGLFrameBuffer::AttachTexture2D(COGLTexture2D* texture, 
	GLenum attachmentSlot)
{
	COGLBindLock lock(this, COGL_FRAMEBUFFER_SLOT);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachmentSlot, GL_TEXTURE_2D, 
		texture->GetResourceIdentifier(), 0);

	CheckGLError(m_DebugName, "COGLFrameBuffer::AttachTexture2D()");
}

void COGLFrameBuffer::Bind(COGLBindSlot slot)
{
	COGLResource::Bind(slot);

	assert(m_Slot == COGL_FRAMEBUFFER_SLOT);

	glBindFramebuffer(GetGLSlot(m_Slot), m_Resource);

	CheckGLError(m_DebugName, "COGLFrameBuffer::Bind()");
}

void COGLFrameBuffer::Unbind()
{
	COGLResource::Unbind();

	assert(m_Slot == COGL_FRAMEBUFFER_SLOT);

	glBindFramebuffer(GetGLSlot(m_Slot), 0);

	CheckGLError(m_DebugName, "COGLFrameBuffer::Unbind()");
}

bool COGLFrameBuffer::CheckFrameBufferComplete()
{
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(status != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << "Framebuffer status not complete." << std::endl;
		return false;
	}

	return true;
}
