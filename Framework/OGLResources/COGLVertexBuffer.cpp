#include "COGLVertexBuffer.h"

#include "COGLBindLock.h"

#include "..\Macros.h"

#include "..\Utils\GLErrorUtil.h"

#include <assert.h>


COGLVertexBuffer::COGLVertexBuffer(std::string debugName)
	: COGLResource(COGL_VERTEXBUFFER, debugName)
{

}

COGLVertexBuffer::~COGLVertexBuffer()
{

}

bool COGLVertexBuffer::Init()
{
	V_RET_FOF(COGLResource::Init());

	glGenBuffers(1, &m_Resource);

	V_RET_FOT(CheckGLError(m_DebugName, "COGLVertexBuffer::Init()"));

	return true;
}

void COGLVertexBuffer::Release()
{
	COGLResource::Release();

	glDeleteBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLArrayBuffer::Release()");
}

bool COGLVertexBuffer::SetContent(GLuint size, void* data, GLenum usage)
{
	CheckInitialized("COGLVertexBuffer::SetContent()");

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