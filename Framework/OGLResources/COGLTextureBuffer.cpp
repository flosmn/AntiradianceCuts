#include "COGLTextureBuffer.h"

#include "..\Macros.h"

#include "..\Utils\GLErrorUtil.h"

COGLTextureBuffer::COGLTextureBuffer(std::string debugName)
	: COGLResource(COGL_TEXTURE_BUFFER, debugName)
{

}

COGLTextureBuffer::~COGLTextureBuffer()
{

}

bool COGLTextureBuffer::Init()
{
	V_RET_FOF(COGLResource::Init());

	glGenTextures(1, &m_TextureBufferTexture);
	glGenBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLTextureBuffer::Init()");

	return true;
}

void COGLTextureBuffer::Release()
{
	COGLResource::Release();

	glDeleteTextures(1, &m_TextureBufferTexture);
	glDeleteBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLTextureBuffer::Release()");
}

void COGLTextureBuffer::SetContent(GLuint elementSize, GLuint numElements, void* content, GLenum usage)
{
	CheckInitialized("COGLTextureBuffer.SetContent()");
	CheckResourceNotNull("COGLTextureBuffer.SetContent()");

	glBindBuffer(GL_TEXTURE_BUFFER, m_Resource);
	glBufferData(GL_TEXTURE_BUFFER, elementSize * numElements, content, usage);
	glBindBuffer(GL_TEXTURE_BUFFER, 0);

	CheckGLError(m_DebugName, "COGLTextureBuffer::SetContent()");
}

void COGLTextureBuffer::Bind(COGLBindSlot slot)
{
	COGLResource::Bind(slot);

	glActiveTexture(GetGLSlot(m_Slot));
	glBindTexture(GL_TEXTURE_BUFFER, m_TextureBufferTexture);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, m_Resource);

	CheckGLError(m_DebugName, "COGLTextureBuffer::Bind()");
}

void COGLTextureBuffer::Unbind()
{
	COGLResource::Unbind();
		
	glBindTexture(GL_TEXTURE_BUFFER, 0);

	CheckGLError(m_DebugName, "COGLTextureBuffer::Unbind()");
}
