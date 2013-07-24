#include "COGLTextureBuffer.h"

COGLTextureBuffer::COGLTextureBuffer(GLenum type, std::string const& debugName)
	: COGLResource(COGL_TEXTURE_BUFFER, debugName), m_Type(type)
{
	glGenTextures(1, &m_TextureBufferTexture);
	glGenBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLTextureBuffer::COGLTextureBuffer()");
}

COGLTextureBuffer::~COGLTextureBuffer()
{
	glDeleteTextures(1, &m_TextureBufferTexture);
	glDeleteBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLTextureBuffer::~COGLTextureBuffer()");
}

void COGLTextureBuffer::SetContent(size_t size, GLenum usage, void* content)
{
	m_size = size;

	glBindBuffer(GL_TEXTURE_BUFFER, m_Resource);
	glBufferData(GL_TEXTURE_BUFFER, size, content, usage);
	glBindBuffer(GL_TEXTURE_BUFFER, 0);

	CheckGLError(m_DebugName, "COGLTextureBuffer::SetContent()");
}

void COGLTextureBuffer::Bind(COGLBindSlot slot)
{
	COGLResource::Bind(slot);

	glActiveTexture(GetGLSlot(m_Slot));
	glBindTexture(GL_TEXTURE_BUFFER, m_TextureBufferTexture);
	glTexBuffer(GL_TEXTURE_BUFFER, m_Type, m_Resource);

	CheckGLError(m_DebugName, "COGLTextureBuffer::Bind()");
}

void COGLTextureBuffer::Unbind()
{
	COGLResource::Unbind();
		
	glBindTexture(GL_TEXTURE_BUFFER, 0);

	CheckGLError(m_DebugName, "COGLTextureBuffer::Unbind()");
}
