#include "COGLRenderBuffer.h"

#include "..\Macros.h"

#include "..\Utils\GLErrorUtil.h"

#include <assert.h>


COGLRenderBuffer::COGLRenderBuffer(std::string debugName)
	: COGLResource(COGL_RENDERBUFFER, debugName)
{

}

COGLRenderBuffer::~COGLRenderBuffer()
{
	COGLResource::~COGLResource();
}

bool COGLRenderBuffer::Init(GLuint width, GLuint height, GLenum internalFormat)
{
	V_RET_FOF(COGLResource::Init());

	m_Width = width;
	m_Height = height;
	m_InternalFormat = internalFormat;

	glGenRenderbuffers(1, &m_Resource);
	{
		COGLBindLock lock(this, COGL_RENDERBUFFER_SLOT);
		glRenderbufferStorage(GL_RENDERBUFFER, m_InternalFormat, m_Width, m_Height);
	}

	V_RET_FOT(CheckGLError(m_DebugName, "COGLRenderBuffer::Init()"));
	
	return true;
}

void COGLRenderBuffer::Release()
{
	COGLResource::Release();

	glDeleteRenderbuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLRenderBuffer::Release()");
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