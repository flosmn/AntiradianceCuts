#include "CGLVertexBuffer.h"

#include "CGLBindLock.h"

#include "..\Macros.h"

#include "..\CUtils\GLErrorUtil.h"

#include <assert.h>


CGLVertexBuffer::CGLVertexBuffer(std::string debugName)
	: CGLResource(CGL_VERTEXBUFFER, debugName)
{

}

CGLVertexBuffer::~CGLVertexBuffer()
{

}

bool CGLVertexBuffer::Init()
{
	V_RET_FOF(CGLResource::Init());

	glGenBuffers(1, &m_Resource);

	V_RET_FOT(CheckGLError(m_DebugName, "CGLVertexBuffer::Init()"));

	return true;
}

void CGLVertexBuffer::Release()
{
	CGLResource::Release();

	glDeleteBuffers(1, &m_Resource);

	CheckGLError(m_DebugName, "CGLArrayBuffer::Release()");
}

bool CGLVertexBuffer::SetContent(GLuint size, void* data, GLenum usage)
{
	CheckInitialized("CGLVertexBuffer::SetContent()");

	CGLBindLock lock(this, CGL_ARRAY_BUFFER_SLOT);
	glBufferData(GL_ARRAY_BUFFER, size, data, usage);

	V_RET_FOT(CheckGLError(m_DebugName, "CGLVertexBuffer::SetContent()"));

	return true;
}

void CGLVertexBuffer::Bind(CGLBindSlot slot)
{
	CGLResource::Bind(slot);

	assert(m_Slot == CGL_ARRAY_BUFFER_SLOT || m_Slot == CGL_ELEMENT_ARRAY_BUFFER_SLOT);

	glBindBuffer(GetGLSlot(m_Slot), m_Resource);
}

void CGLVertexBuffer::Unbind()
{
	CGLResource::Unbind();

	assert(m_Slot == CGL_ARRAY_BUFFER_SLOT || m_Slot == CGL_ELEMENT_ARRAY_BUFFER_SLOT);

	glBindBuffer(GetGLSlot(m_Slot), 0);
}