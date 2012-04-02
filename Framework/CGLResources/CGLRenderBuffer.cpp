#include "CGLRenderBuffer.h"

#include "Macros.h"
#include "GLErrorUtil.h"

#include <assert.h>


CGLRenderBuffer::CGLRenderBuffer(std::string debugName)
	: CGLResource(CGL_RENDERBUFFER, debugName)
{

}

CGLRenderBuffer::~CGLRenderBuffer()
{
	CGLResource::~CGLResource();
}

bool CGLRenderBuffer::Init(GLuint width, GLuint height, GLenum internalFormat)
{
	V_RET_FOF(CGLResource::Init());

	m_Width = width;
	m_Height = height;
	m_InternalFormat = internalFormat;

	glGenRenderbuffers(1, &m_Resource);
	{
		CGLBindLock lock(this, CGL_RENDERBUFFER_SLOT);
		glRenderbufferStorage(GL_RENDERBUFFER, m_InternalFormat, m_Width, m_Height);
	}

	V_RET_FOT(CheckGLError(m_DebugName, "CGLRenderBuffer::Init()"));
	
	return true;
}

void CGLRenderBuffer::Release()
{
	CGLResource::Release();

	glDeleteRenderbuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "CGLRenderBuffer::Release()");
}

void CGLRenderBuffer::Bind(CGLBindSlot slot)
{
	CGLResource::Bind(slot);

	assert(m_Slot == CGL_RENDERBUFFER_SLOT);

	glBindRenderbuffer(GetGLSlot(m_Slot), m_Resource);
}

void CGLRenderBuffer::Unbind()
{
	CGLResource::Unbind();

	assert(m_Slot == CGL_RENDERBUFFER_SLOT);

	glBindRenderbuffer(GetGLSlot(m_Slot), 0);
}