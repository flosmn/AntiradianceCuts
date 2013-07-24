#include "COGLRenderBuffer.h"

#include <assert.h>

COGLRenderBuffer::COGLRenderBuffer(GLuint width, GLuint height, 
	GLenum internalFormat, std::string const& debugName)
	: COGLResource(COGL_RENDERBUFFER, debugName)
{
	m_Width = width;
	m_Height = height;
	m_InternalFormat = internalFormat;

	glGenRenderbuffers(1, &m_Resource);
	{
		COGLBindLock lock(this, COGL_RENDERBUFFER_SLOT);
		glRenderbufferStorage(GL_RENDERBUFFER, m_InternalFormat, m_Width, m_Height);
	}

	CheckGLError(m_DebugName, "COGLRenderBuffer::COGLRenderBuffer()");
}

COGLRenderBuffer::~COGLRenderBuffer()
{
	glDeleteRenderbuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLRenderBuffer::~COGLRenderBuffer()");
}

void COGLRenderBuffer::Bind(COGLBindSlot slot)
{
	COGLResource::Bind(slot);

	assert(m_Slot == COGL_RENDERBUFFER_SLOT);

	glBindRenderbuffer(GetGLSlot(m_Slot), m_Resource);
}

void COGLRenderBuffer::Unbind()
{
	COGLResource::Unbind();

	assert(m_Slot == COGL_RENDERBUFFER_SLOT);

	glBindRenderbuffer(GetGLSlot(m_Slot), 0);
}