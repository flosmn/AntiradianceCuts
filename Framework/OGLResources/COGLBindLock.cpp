#include "COGLBindLock.h"

#include "COGLResource.h"

COGLBindLock::COGLBindLock(COGLResource *resource, COGLBindSlot slot) 
	: m_pResource(resource)
{	
	m_pResource->Bind(slot);
}

COGLBindLock::~COGLBindLock()
{
	m_pResource->Unbind();
}
