#include "COGLResource.h"

#include <iostream>
#include <string>

COGLResource::COGLResource(COGLResourceType resourceType, std::string const& debugName)
	: m_Resource(0), m_DebugName(debugName), 
	  m_ResourceType(resourceType), m_IsBound(false), m_IsInitialized(false)
{
}

COGLResource::~COGLResource()
{	
	CheckNotBound("COGLResource::~COGLResource()");
}

void COGLResource::Bind(COGLBindSlot slot)
{
	CheckNotBound("COGLResource::Bind()");

	m_Slot = slot;

	if (GetDebugName() != "")
	{
		std::map<COGLBindSlot, std::string>& mapSlotsToResources = 
			GetMapForResourceType(m_ResourceType);

		std::string resNameToBind = GetDebugName();
		std::string resNameBound = mapSlotsToResources[slot];
		
		if(!(resNameBound == "" || resNameBound == resNameToBind)) 
		{
			std::cout << "Warning! Trying to bind COGLResource: " << resNameToBind 
				<< " to slot " << GetStringForBindSlot(m_Slot) << " but the COGLResource " 
				<< resNameBound << " is still bound to this slot!!!" << std::endl;
		}
		mapSlotsToResources[slot] = resNameToBind;
	}

	m_IsBound = true;
}

void COGLResource::Unbind()
{
	CheckBound("COGLResource::Unbind()");

	if (GetDebugName() != "")
	{
		std::map<COGLBindSlot, std::string>& mapSlotsToResources = 
			GetMapForResourceType(m_ResourceType);

		std::string resNameToUnbind = GetDebugName();
		std::string resNameBound = mapSlotsToResources[m_Slot];
		if(resNameBound != resNameToUnbind) 
		{
			std::cout << "Warning! Trying to unbind COGLResource: " << resNameToUnbind 
				<< " from slot " << GetStringForBindSlot(m_Slot) << " but the COGLResource " 
				<< resNameBound << " is bound to this slot!!!" << std::endl;
		}
		if(resNameBound == "") 
		{
			std::cout << "Warning! Trying to unbind COGLResource: " << resNameToUnbind 
				<< " from slot " << GetStringForBindSlot(m_Slot) << " but there is nothing bound!!!" 
				<< std::endl;
		}
		
		mapSlotsToResources[m_Slot] = "";
	}

	m_IsBound = false;
}


GLuint COGLResource::GetResourceIdentifier()
{ 
	return m_Resource;
}

bool COGLResource::CheckBound(std::string checker)
{
	if (!m_IsBound)
	{
		std::cout << checker << ": "
			<< "COGLResource " << m_DebugName 
			<< " is not bound!!!" << std::endl;
		return false;
	}

	return true;
}

bool COGLResource::CheckNotBound(std::string checker)
{
	if (m_IsBound)
	{
		std::cout << checker << ": "
			<< "COGLResource " << m_DebugName 
			<< " is still bound!!!" << std::endl;
		return false;
	}

	return true;
}

std::string COGLResource::GetDebugName()
{
	return m_DebugName;
}

std::map<COGLBindSlot, std::string>& COGLResource::GetMapForResourceType(COGLResourceType type)
{
	switch(type)
	{
	case COGL_TEXTURE_2D: return mapSlotsToCOGLTEXTURE2D;
	case COGL_TEXTURE_BUFFER: return mapSlotsToCOGLTEXTUREBUFFER;
	case COGL_FRAMEBUFFER: return mapSlotsToCOGLFRAMEBUFFER;
	case COGL_RENDERBUFFER: return mapSlotsToCOGLRENDERBUFFER;
	case COGL_VERTEXBUFFER: return mapSlotsToCOGLVERTEXBUFFER;
	case COGL_VERTEXARRAY: return mapSlotsToCOGLVERTEXARRAY;
	case COGL_UNIFORMBUFFER: return mapSlotsToCOGLUNIFORMBUFFER;
	case COGL_PROGRAM: return mapSlotsToCOGLPROGRAM;
	case COGL_CUBE_MAP: return mapSlotsToCOGLCUBEMAP;

	default: std::cout << "Invalid resource type" << std::endl;
			 return mapSlotsToCOGLTEXTURE2D;
	}
}

void COGLResource::PrintWarning(std::string warning )
{
	std::cout << "The COGLResource" << m_DebugName 
		<< " has a warning:" << warning << std::endl;
}

std::string COGLResource::GetStringForBindSlot( COGLBindSlot slot )
{
	switch(slot)
	{
	case COGL_TEXTURE0_SLOT : return "GL_TEXTURE0";
	case COGL_TEXTURE1_SLOT : return "GL_TEXTURE1";
	case COGL_TEXTURE2_SLOT : return "GL_TEXTURE2";
	case COGL_TEXTURE3_SLOT : return "GL_TEXTURE3";
	case COGL_TEXTURE4_SLOT : return "GL_TEXTURE4";
	case COGL_TEXTURE5_SLOT : return "GL_TEXTURE5";
	case COGL_TEXTURE6_SLOT : return "GL_TEXTURE6";
	case COGL_TEXTURE7_SLOT : return "GL_TEXTURE7";
	case COGL_TEXTURE8_SLOT : return "GL_TEXTURE8";
	case COGL_TEXTURE9_SLOT : return "GL_TEXTURE9";
	
	case COGL_FRAMEBUFFER_SLOT : return "GL_FRAMEBUFFER";
	case COGL_RENDERBUFFER_SLOT : return "GL_RENDERBUFFER";
	case COGL_ARRAY_BUFFER_SLOT : return "GL_ARRAY_BUFFER";
	case COGL_ELEMENT_ARRAY_BUFFER_SLOT : return "GL_ELEMENT_ARRAY_BUFFER";
	case COGL_VERTEX_ARRAY_SLOT : return "GL_VERTEX_ARRAY";
	case COGL_PROGRAM_SLOT : return "GL_PROGRAM";

	default : std::cout << "No string for GLenum found!!!" << std::endl; return "";
	}
}

std::map<COGLBindSlot, std::string> COGLResource::mapSlotsToCOGLTEXTURE2D;
std::map<COGLBindSlot, std::string> COGLResource::mapSlotsToCOGLTEXTUREBUFFER;
std::map<COGLBindSlot, std::string> COGLResource::mapSlotsToCOGLFRAMEBUFFER;
std::map<COGLBindSlot, std::string> COGLResource::mapSlotsToCOGLRENDERBUFFER;
std::map<COGLBindSlot, std::string> COGLResource::mapSlotsToCOGLVERTEXBUFFER;
std::map<COGLBindSlot, std::string> COGLResource::mapSlotsToCOGLVERTEXARRAY;
std::map<COGLBindSlot, std::string> COGLResource::mapSlotsToCOGLUNIFORMBUFFER;
std::map<COGLBindSlot, std::string> COGLResource::mapSlotsToCOGLPROGRAM;
std::map<COGLBindSlot, std::string> COGLResource::mapSlotsToCOGLCUBEMAP;
