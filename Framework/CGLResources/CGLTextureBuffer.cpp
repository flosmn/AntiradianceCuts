#include "CGLTextureBuffer.h"

#include "..\Macros.h"

#include "..\CUtils\GLErrorUtil.h"

CGLTextureBuffer::CGLTextureBuffer(std::string debugName)
	: CGLResource(CGL_TEXTURE_BUFFER, debugName)
{

}

CGLTextureBuffer::~CGLTextureBuffer()
{

}

bool CGLTextureBuffer::Init()
{
	V_RET_FOF(CGLResource::Init());

	glGenTextures(1, &m_TextureBufferTexture);
	glGenBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "CGLTextureBuffer::Init()");

	return true;
}

void CGLTextureBuffer::Release()
{
	CGLResource::Release();

	glDeleteTextures(1, &m_TextureBufferTexture);
	glDeleteBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "CGLTextureBuffer::Release()");
}

void CGLTextureBuffer::SetContent(GLuint elementSize, GLuint numElements, void* content, GLenum usage)
{
	CheckInitialized("CGLTextureBuffer.SetContent()");
	CheckResourceNotNull("CGLTextureBuffer.SetContent()");

	glBindBuffer(GL_TEXTURE_BUFFER, m_Resource);
	glBufferData(GL_TEXTURE_BUFFER, elementSize * numElements, content, usage);
	glBindBuffer(GL_TEXTURE_BUFFER, 0);

	CheckGLError(m_DebugName, "CGLTextureBuffer::SetContent()");
}

void CGLTextureBuffer::Bind(CGLBindSlot slot)
{
	CGLResource::Bind(slot);

	glActiveTexture(GetGLSlot(m_Slot));
	glBindTexture(GL_TEXTURE_BUFFER, m_TextureBufferTexture);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, m_Resource);

	CheckGLError(m_DebugName, "CGLTextureBuffer::Bind()");
}

void CGLTextureBuffer::Unbind()
{
	CGLResource::Unbind();
		
	glBindTexture(GL_TEXTURE_BUFFER, 0);

	CheckGLError(m_DebugName, "CGLTextureBuffer::Unbind()");
}
