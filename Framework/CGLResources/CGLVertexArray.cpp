#include "CGLVertexArray.h"

#include "CGLVertexBuffer.h"
#include "CGLBindLock.h"

#include "..\Macros.h"

#include "..\CUtils\GLErrorUtil.h"

#include <assert.h>
#include <iostream>


CGLVertexArray::CGLVertexArray(std::string debugName)
	: CGLResource(CGL_VERTEXARRAY, debugName), m_pGLVBIndexData(nullptr), m_HasIndexData(false)
{
	m_pGLVBIndexData = new CGLVertexBuffer("CGLVertexArray.m_pGLVBIndexData");

	memset(m_ChannelsUsed, 0, 10 * sizeof(uint));
}

CGLVertexArray::~CGLVertexArray()
{
	SAFE_DELETE(m_pGLVBIndexData);

	for(uint i = 0; i < 10; ++i)
	{
		if(m_ChannelsUsed[i] != 0)
			SAFE_DELETE(m_VertexDataChannels[i]);
	}
}

bool CGLVertexArray::Init(GLenum primitiveType)
{
	V_RET_FOF(CGLResource::Init());

	m_PrimitiveType = primitiveType;

	glGenVertexArrays(1, &m_Resource);

	V_RET_FOT(CheckGLError(m_DebugName, "CGLVertexArray::Init()"));

	return true;
}

void CGLVertexArray::Release()
{
	CGLResource::Release();

	for(uint i = 0; i < 10; ++i)
	{
		if(m_ChannelsUsed[i] != 0)
			m_VertexDataChannels[i]->Release();
	}

	if(m_HasIndexData)
		m_pGLVBIndexData->Release();

	glDeleteVertexArrays(1, &m_Resource);

	CheckGLError(m_DebugName, "CGLVertexArray::Release()");
}

bool CGLVertexArray::AddVertexDataChannel(uint index, uint elements)
{
	CheckInitialized("CGLVertexArray::AddVertexDataChannel()");
	CheckNotBound("CGLVertexArray::AddVertexDataChannel()");

	CHANNEL_INFO info;
	info.index = index;
	info.elements = elements;
	
	m_VertexDataChannelInfo[index] = info;
	m_ChannelsUsed[index] = 1;

	CGLVertexBuffer* pBuffer = new CGLVertexBuffer("CGLVertexArray.pBuffer");
	V_RET_FOF(pBuffer->Init());

	m_VertexDataChannels[index] = pBuffer;

	return true;
}

bool CGLVertexArray::AddVertexData(uint index, uint size, void* pData)
{
	CheckInitialized("CGLVertexArray::AddVertexData()");
	CheckNotBound("CGLVertexArray::AddVertexData()");
		
	V_RET_FOF(m_VertexDataChannels[index]->SetContent(size, pData, GL_STATIC_DRAW));
	
	return true;
}

bool CGLVertexArray::AddIndexDataChannel()
{
	CheckInitialized("CGLVertexArray::AddIndexData()");
	CheckNotBound("CGLVertexArray::AddIndexData()");

	V_RET_FOF(m_pGLVBIndexData->Init());
	
	CheckGLError(m_DebugName, "CGLVertexArray::AddIndexDataChannel()");
	
	m_HasIndexData = true;

	return true;
}

bool CGLVertexArray::AddIndexData(GLuint size, void* pData)
{
	CheckInitialized("CGLVertexArray::AddIndexData()");
	CheckNotBound("CGLVertexArray::AddIndexData()");

	V_RET_FOF(m_pGLVBIndexData->SetContent(size, pData, GL_STATIC_DRAW));

	CheckGLError(m_DebugName, "CGLVertexArray::AddIndexData()");
	
	return true;
}

void CGLVertexArray::Finish()
{
	CheckInitialized("CGLVertexArray::Finish()");
	CheckNotBound("CGLVertexArray::Finish()");
	
	if(!m_HasIndexData)
	{
		std::cout << "CGLVertexArray: " << m_DebugName 
			<< " has no index data!!!"	<< std::endl;
	}

	CGLBindLock lock(this, CGL_VERTEX_ARRAY_SLOT);
	
	for(uint i = 0; i < 10; ++i)
	{
		if(m_ChannelsUsed[i] == 0)
			continue;

		CGLVertexBuffer* pBuffer = m_VertexDataChannels[i];
					
		CHANNEL_INFO info = m_VertexDataChannelInfo[i];
			
		CGLBindLock lock(pBuffer, CGL_ARRAY_BUFFER_SLOT);
		glEnableVertexAttribArray(info.index);
		glVertexAttribPointer(info.index, info.elements, GL_FLOAT, GL_FALSE, 0, 0);
	}

	CGLBindLock lockIndexData(m_pGLVBIndexData, CGL_ELEMENT_ARRAY_BUFFER_SLOT);
	
	CheckGLError(m_DebugName, "CGLVertexArray::Finish()");

	glBindVertexArray(0);
}

void CGLVertexArray::Draw(GLuint count)
{
	CheckInitialized("CGLVertexArray::Draw()");
	CheckNotBound("CGLVertexArray::Draw()");
	
	CheckGLError(m_DebugName, "CGLVertexArray::Draw() before glDrawElements");

	CGLBindLock lock(this, CGL_VERTEX_ARRAY_SLOT);

	glDrawElements(m_PrimitiveType, count, GL_UNSIGNED_SHORT, 0);

	CheckGLError(m_DebugName, "CGLVertexArray::Draw() after glDrawElements");
}

void CGLVertexArray::Bind(CGLBindSlot slot)
{	
	CGLResource::Bind(slot);

	assert(m_Slot == CGL_VERTEX_ARRAY_SLOT);

	glBindVertexArray(m_Resource);
}

void CGLVertexArray::Unbind()
{
	CGLResource::Unbind();

	assert(m_Slot == CGL_VERTEX_ARRAY_SLOT);

	glBindVertexArray(0);
}
