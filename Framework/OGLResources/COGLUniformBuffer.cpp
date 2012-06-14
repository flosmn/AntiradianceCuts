#include "COGLUniformBuffer.h"

#include "..\Macros.h"

#include "..\Utils\GLErrorUtil.h"

#include <assert.h>


uint COGLUniformBuffer::static_GlobalBindingPoint = 0;


COGLUniformBuffer::COGLUniformBuffer(std::string debugName)
	: COGLResource(COGL_UNIFORMBUFFER, debugName), m_Size(0)
{

}

COGLUniformBuffer::~COGLUniformBuffer()
{
	COGLResource::~COGLResource();
}

bool COGLUniformBuffer::Init(uint size, void* data, GLenum usage)
{
	V_RET_FOF(COGLResource::Init());

	m_Size = size;
	m_GlobalBindingPoint = GetUniqueGlobalBindingPoint();

	glGenBuffers(1, &m_Resource);
	
	CheckGLError("COGLUniformBuffer", "Init() 1");

	COGLBindLock lock(this, COGL_UNIFORM_BUFFER_SLOT);
	
	CheckGLError("COGLUniformBuffer", "Init() 2");

	glBufferData(GL_UNIFORM_BUFFER, m_Size, data, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, m_GlobalBindingPoint, m_Resource, 0, m_Size);
	
	CheckGLError("COGLUniformBuffer", "Init() 3");

	return true;
}	

void COGLUniformBuffer::Release()
{
	COGLResource::Release();

	glDeleteBuffers(1, &m_Resource);
}

void COGLUniformBuffer::UpdateData(void* data)
{
	CheckInitialized("COGLUniformBuffer::UpdateData()");
	CheckNotBound("COGLUniformBuffer::UpdateData()");
	
	COGLBindLock lock(this, COGL_UNIFORM_BUFFER_SLOT);
	
	CheckGLError("COGLUniformBuffer", "UpdateData() 1");

	glBufferData(GL_UNIFORM_BUFFER, m_Size, data, GL_DYNAMIC_DRAW);
	
	CheckGLError("COGLUniformBuffer", "UpdateData() 2");
}

uint COGLUniformBuffer::GetGlobalBindingPoint()
{
	CheckInitialized("COGLUniformBuffer::GetGlobalBindingPoint()");

	return m_GlobalBindingPoint;
}

void COGLUniformBuffer::Bind(COGLBindSlot slot)
{
	COGLResource::Bind(slot);

	assert(m_Slot == COGL_UNIFORM_BUFFER_SLOT);

	glBindBuffer(GetGLSlot(m_Slot), m_Resource);
}

void COGLUniformBuffer::Unbind()
{
	COGLResource::Unbind();

	assert(m_Slot == COGL_UNIFORM_BUFFER_SLOT);

	glBindBuffer(GetGLSlot(m_Slot), 0);
}

uint COGLUniformBuffer::GetUniqueGlobalBindingPoint()
{
	uint uniqueBP = static_GlobalBindingPoint;
	static_GlobalBindingPoint++;
	return uniqueBP;
}