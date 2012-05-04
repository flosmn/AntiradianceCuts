#include "CGLResource.h"

#include "..\CUtils\GLErrorUtil.h"

#include <iostream>
#include <string>

CGLResource::CGLResource(CGLResourceType resourceType, std::string debugName)
	: m_Resource(0), m_DebugName(debugName), 
	  m_ResourceType(resourceType), m_IsBound(false), m_IsInitialized(false)
{
	if(m_DebugName == "") 
	{
		std::cout << "CGLResource: " 
			<< " no valid debug name for a resource!!!"	<< std::endl;
	}
}

CGLResource::~CGLResource()
{
	CheckNotInitialized("CGLResource::~CGLResource()");
	CheckNotBound("CGLResource::~CGLResource()");
}

bool CGLResource::Init()
{
	CheckNotInitialized("CGLResource::Init()");
	CheckNotBound("CGLResource::Init()");

	m_IsInitialized = true;
	return true;
}

void CGLResource::Release()
{
	CheckInitialized("CGLResource::Release()");
	CheckNotBound("CGLResource::Release()");

	m_IsInitialized = false;
}

void CGLResource::Bind(CGLBindSlot slot)
{
	CheckInitialized("CGLResource::Bind()");
	CheckNotBound("CGLResource::Bind()");

	m_Slot = slot;

	std::map<CGLBindSlot, std::string>& mapSlotsToResources = 
		GetMapForResourceType(m_ResourceType);

	std::string resNameToBind = GetDebugName();
	std::string resNameBound = mapSlotsToResources[slot];
	
	if(!(resNameBound == "" || resNameBound == resNameToBind)) 
	{
		std::cout << "Warning! Trying to bind CGLResource: " << resNameToBind 
			<< " to slot " << GetStringForBindSlot(m_Slot) << " but the CGLResource " 
			<< resNameBound << " is still bound to this slot!!!" << std::endl;
	}
	mapSlotsToResources[slot] = resNameToBind;

	m_IsBound = true;
}

void CGLResource::Unbind()
{
	CheckInitialized("CGLResource::Unbind()");
	CheckBound("CGLResource::Unbind()");

	std::map<CGLBindSlot, std::string>& mapSlotsToResources = 
		GetMapForResourceType(m_ResourceType);

	std::string resNameToUnbind = GetDebugName();
	std::string resNameBound = mapSlotsToResources[m_Slot];
	if(resNameBound != resNameToUnbind) 
	{
		std::cout << "Warning! Trying to unbind CGLResource: " << resNameToUnbind 
			<< " from slot " << GetStringForBindSlot(m_Slot) << " but the CGLResource " 
			<< resNameBound << " is bound to this slot!!!" << std::endl;
	}
	if(resNameBound == "") 
	{
		std::cout << "Warning! Trying to unbind CGLResource: " << resNameToUnbind 
			<< " from slot " << GetStringForBindSlot(m_Slot) << " but there is nothing bound!!!" 
			<< std::endl;
	}
	
	mapSlotsToResources[m_Slot] = "";

	m_IsBound = false;
}


GLuint CGLResource::GetResourceIdentifier()
{ 
	CheckInitialized("CGLResource::GetResourceIdentifier");

	return m_Resource;
}

bool CGLResource::CheckInitialized(std::string checker)
{
	if (!m_IsInitialized)
	{
		std::cout << checker << ": "
			<< "CGLResource " << m_DebugName 
			<< " is not initialized!!!" << std::endl;
		return false;
	}

	return CheckResourceNotNull(checker);
}

bool CGLResource::CheckNotInitialized(std::string checker)
{
	if (m_IsInitialized)
	{
		std::cout << checker << ": "
			<< "CGLResource " << m_DebugName 
			<< " is still initialized!!!" << std::endl;
		return false;
	}

	return true;
}

bool CGLResource::CheckBound(std::string checker)
{
	if (!m_IsBound)
	{
		std::cout << checker << ": "
			<< "CGLResource " << m_DebugName 
			<< " is not bound!!!" << std::endl;
		return false;
	}

	return true;
}

bool CGLResource::CheckNotBound(std::string checker)
{
	if (m_IsBound)
	{
		std::cout << checker << ": "
			<< "CGLResource " << m_DebugName 
			<< " is still bound!!!" << std::endl;
		return false;
	}

	return true;
}

bool CGLResource::CheckResourceNotNull(std::string checker)
{
	if (m_Resource == 0)
	{
		std::cout << checker << ": "
			<< "CGLResource " << m_DebugName 
			<< " is null!!!" << std::endl;
		return false;
	}

	return true;
}

std::string CGLResource::GetDebugName()
{
	return m_DebugName;
}

std::map<CGLBindSlot, std::string>& CGLResource::GetMapForResourceType(CGLResourceType type)
{
	switch(type)
	{
	case CGL_TEXTURE_2D: return mapSlotsToCGLTEXTURE2D;
	case CGL_TEXTURE_BUFFER: return mapSlotsToCGLTEXTUREBUFFER;
	case CGL_FRAMEBUFFER: return mapSlotsToCGLFRAMEBUFFER;
	case CGL_RENDERBUFFER: return mapSlotsToCGLRENDERBUFFER;
	case CGL_VERTEXBUFFER: return mapSlotsToCGLVERTEXBUFFER;
	case CGL_VERTEXARRAY: return mapSlotsToCGLVERTEXARRAY;
	case CGL_UNIFORMBUFFER: return mapSlotsToCGLUNIFORMBUFFER;
	case CGL_PROGRAM: return mapSlotsToCGLPROGRAM;

	default: std::cout << "Invalid resource type" << std::endl;
			 return mapSlotsToCGLTEXTURE2D;
	}
}

void CGLResource::PrintWarning(std::string warning )
{
	std::cout << "The CGLResource" << m_DebugName 
		<< " has a warning:" << warning << std::endl;
}

std::string CGLResource::GetStringForBindSlot( CGLBindSlot slot )
{
	switch(slot)
	{
	case CGL_TEXTURE0_SLOT : return "GL_TEXTURE0";
	case CGL_TEXTURE1_SLOT : return "GL_TEXTURE1";
	case CGL_TEXTURE2_SLOT : return "GL_TEXTURE2";
	case CGL_TEXTURE3_SLOT : return "GL_TEXTURE3";
	case CGL_TEXTURE4_SLOT : return "GL_TEXTURE4";
	case CGL_TEXTURE5_SLOT : return "GL_TEXTURE5";
	case CGL_TEXTURE6_SLOT : return "GL_TEXTURE6";
	case CGL_TEXTURE7_SLOT : return "GL_TEXTURE7";
	case CGL_TEXTURE8_SLOT : return "GL_TEXTURE8";
	case CGL_TEXTURE9_SLOT : return "GL_TEXTURE9";
	
	case CGL_FRAMEBUFFER_SLOT : return "GL_FRAMEBUFFER";
	case CGL_RENDERBUFFER_SLOT : return "GL_RENDERBUFFER";
	case CGL_ARRAY_BUFFER_SLOT : return "GL_ARRAY_BUFFER";
	case CGL_ELEMENT_ARRAY_BUFFER_SLOT : return "GL_ELEMENT_ARRAY_BUFFER";
	case CGL_VERTEX_ARRAY_SLOT : return "GL_VERTEX_ARRAY";

	default : std::cout << "No string for GLenum found!!!" << std::endl; return "";
	}
}

std::map<CGLBindSlot, std::string> CGLResource::mapSlotsToCGLTEXTURE2D;
std::map<CGLBindSlot, std::string> CGLResource::mapSlotsToCGLTEXTUREBUFFER;
std::map<CGLBindSlot, std::string> CGLResource::mapSlotsToCGLFRAMEBUFFER;
std::map<CGLBindSlot, std::string> CGLResource::mapSlotsToCGLRENDERBUFFER;
std::map<CGLBindSlot, std::string> CGLResource::mapSlotsToCGLVERTEXBUFFER;
std::map<CGLBindSlot, std::string> CGLResource::mapSlotsToCGLVERTEXARRAY;
std::map<CGLBindSlot, std::string> CGLResource::mapSlotsToCGLUNIFORMBUFFER;
std::map<CGLBindSlot, std::string> CGLResource::mapSlotsToCGLPROGRAM;