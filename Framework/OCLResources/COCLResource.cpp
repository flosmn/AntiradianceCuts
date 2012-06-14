#include "COCLResource.h"

#include <iostream>
#include <string>

COCLResource::COCLResource(std::string debugName)
	: m_DebugName(debugName), m_IsInitialized(false)
{
	if(m_DebugName == "") 
	{
		std::cout << "COCLResource: "  << " no valid debug name for a resource!!!"	<< std::endl;
	}
}

COCLResource::~COCLResource()
{
	CheckNotInitialized("COCLResource::~COCLResource()");
}

bool COCLResource::Init()
{
	CheckNotInitialized("COCLResource::Init()");
	
	m_IsInitialized = true;
	return true;
}

void COCLResource::Release()
{
	CheckInitialized("COCLResource::Release()");
	
	m_IsInitialized = false;
}

bool COCLResource::CheckInitialized(std::string checker)
{
	if (!m_IsInitialized)
	{
		std::cout << checker << ": " << "COCLResource " << m_DebugName 	<< " is not initialized!!!" << std::endl;
		return false;
	}

	return true;
}

bool COCLResource::CheckNotInitialized(std::string checker)
{
	if (m_IsInitialized)
	{
		std::cout << checker << ": " << "COCLResource " << m_DebugName << " is still initialized!!!" << std::endl;
		return false;
	}

	return true;
}

std::string COCLResource::GetDebugName()
{
	return m_DebugName;
}