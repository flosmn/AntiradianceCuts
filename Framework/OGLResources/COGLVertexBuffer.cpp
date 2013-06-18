#include "COGLVertexBuffer.h"

#include "COGLBindLock.h"

#include "..\Macros.h"

#include "..\Utils\GLErrorUtil.h"

#include <assert.h>


COGLVertexBuffer::COGLVertexBuffer(std::string const& debugName)
	: COGLResource(COGL_VERTEXBUFFER, debugName)
{
	glGenBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLVertexBuffer::COGLVertexBuffer()");
}

COGLVertexBuffer::~COGLVertexBuffer()
{
	glDeleteBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLArrayBuffer::~COGLVertexBuffer()");
}

bool COGLVertexBuffer::SetContent(GLuint size, void* data, GLenum usage)
{
	COGLBindLock lock(this, COGL_ARRAY_BUFFER_SLOT);
	glBufferData(GL_ARRAY_BUFFER, size, data, usage);

	V_RET_FOT(CheckGLError(m_DebugName, "COGLVertexBuffer::SetContent()"));

	return true;
}

void COGLVertexBuffer::Bind(COGLBindSlot slot)
{
	COGLResource::Bind(slot);

	assert(m_Slot == COGL_ARRAY_BUFFER_SLOT || m_Slot == COGL_ELEMENT_ARRAY_BUFFER_SLOT);

	glBindBuffer(GetGLSlot(m_Slot), m_Resource);
}

void COGLVertexBuffer::Unbind()
{
	COGLResource::Unbind();

	assert(m_Slot == COGL_ARRAY_BUFFER_SLOT || m_Slot == COGL_ELEMENT_ARRAY_BUFFER_SLOT);

	glBindBuffer(GetGLSlot(m_Slot), 0);
}