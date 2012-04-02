#include "CGLBindLock.h"

#include "CGLResource.h"

CGLBindLock::CGLBindLock(CGLResource *resource, CGLBindSlot slot) 
	: m_pResource(resource)
{	
	m_pResource->Bind(slot);
}

CGLBindLock::~CGLBindLock()
{
	m_pResource->Unbind();
}
