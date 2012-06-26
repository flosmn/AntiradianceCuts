#include "COGLTextureBuffer.h"

#include "..\Macros.h"

#include "..\Utils\GLErrorUtil.h"

COGLTextureBuffer::COGLTextureBuffer(std::string debugName)
	: COGLResource(COGL_TEXTURE_BUFFER, debugName), m_Size(0)
{

}

COGLTextureBuffer::~COGLTextureBuffer()
{

}

bool COGLTextureBuffer::Init(size_t size, GLenum usage)
{
	V_RET_FOF(COGLResource::Init());

	glGenTextures(1, &m_TextureBufferTexture);
	glGenBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLTextureBuffer::Init()");

	m_Usage = usage;
	m_Size = size;
	SetContent(NULL, m_Size);

	return true;
}

void COGLTextureBuffer::Release()
{
	COGLResource::Release();

	glDeleteTextures(1, &m_TextureBufferTexture);
	glDeleteBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLTextureBuffer::Release()");
}

void COGLTextureBuffer::SetContent(void* content, size_t size)
{
	CheckInitialized("COGLTextureBuffer.SetContent()");
	CheckResourceNotNull("COGLTextureBuffer.SetContent()");

	if(size > m_Size)
	{
		std::cout << "COGLTextureBuffer::SetContent(): Warning: size of data greater than buffer size." << std::endl;
		size = std::min(size, m_Size);
	}

	glBindBuffer(GL_TEXTURE_BUFFER, m_Resource);
	glBufferData(GL_TEXTURE_BUFFER, size, content, m_Usage);
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

size_t COGLTextureBuffer::GetSize()
{
	return m_Size;
}
