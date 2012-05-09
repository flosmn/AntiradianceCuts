#include "CGLUniformBuffer.h"

#include "..\Macros.h"

#include "..\CUtils\GLErrorUtil.h"

#include <assert.h>


uint CGLUniformBuffer::static_GlobalBindingPoint = 0;


CGLUniformBuffer::CGLUniformBuffer(std::string debugName)
	: CGLResource(CGL_UNIFORMBUFFER, debugName), m_Size(0)
{

}

CGLUniformBuffer::~CGLUniformBuffer()
{
	CGLResource::~CGLResource();
}

bool CGLUniformBuffer::Init(uint size, void* data, GLenum usage)
{
	V_RET_FOF(CGLResource::Init());

	m_Size = size;
	m_GlobalBindingPoint = GetUniqueGlobalBindingPoint();

	glGenBuffers(1, &m_Resource);
	
	CheckGLError("CGLUniformBuffer", "Init() 1");

	CGLBindLock lock(this, CGL_UNIFORM_BUFFER_SLOT);
	
	CheckGLError("CGLUniformBuffer", "Init() 2");

	glBufferData(GL_UNIFORM_BUFFER, m_Size, data, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, m_GlobalBindingPoint, m_Resource, 0, m_Size);
	
	CheckGLError("CGLUniformBuffer", "Init() 3");

	return true;
}	

void CGLUniformBuffer::Release()
{
	CGLResource::Release();

	glDeleteBuffers(1, &m_Resource);
}

void CGLUniformBuffer::UpdateData(void* data)
{
	CheckInitialized("CGLUniformBuffer::UpdateData()");
	CheckNotBound("CGLUniformBuffer::UpdateData()");
	
	CGLBindLock lock(this, CGL_UNIFORM_BUFFER_SLOT);
	
	CheckGLError("CGLUniformBuffer", "UpdateData() 1");

	glBufferData(GL_UNIFORM_BUFFER, m_Size, data, GL_DYNAMIC_DRAW);
	
	CheckGLError("CGLUniformBuffer", "UpdateData() 2");
}

uint CGLUniformBuffer::GetGlobalBindingPoint()
{
	CheckInitialized("CGLUniformBuffer::GetGlobalBindingPoint()");

	return m_GlobalBindingPoint;
}

void CGLUniformBuffer::Bind(CGLBindSlot slot)
{
	CGLResource::Bind(slot);

	assert(m_Slot == CGL_UNIFORM_BUFFER_SLOT);

	glBindBuffer(GetGLSlot(m_Slot), m_Resource);
}

void CGLUniformBuffer::Unbind()
{
	CGLResource::Unbind();

	assert(m_Slot == CGL_UNIFORM_BUFFER_SLOT);

	glBindBuffer(GetGLSlot(m_Slot), 0);
}

uint CGLUniformBuffer::GetUniqueGlobalBindingPoint()
{
	uint uniqueBP = static_GlobalBindingPoint;
	static_GlobalBindingPoint++;
	return uniqueBP;
}