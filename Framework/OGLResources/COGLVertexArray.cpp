#include "COGLVertexArray.h"

#include "COGLVertexBuffer.h"
#include "COGLBindLock.h"

#include "..\Macros.h"

#include "..\Utils\GLErrorUtil.h"

#include <assert.h>
#include <iostream>


COGLVertexArray::COGLVertexArray(std::string debugName)
	: COGLResource(COGL_VERTEXARRAY, debugName), m_pGLVBIndexData(nullptr), m_HasIndexData(false)
{
	m_pGLVBIndexData = new COGLVertexBuffer("COGLVertexArray.m_pGLVBIndexData");

	memset(m_ChannelsUsed, 0, 10 * sizeof(uint));
}

COGLVertexArray::~COGLVertexArray()
{
	SAFE_DELETE(m_pGLVBIndexData);

	for(uint i = 0; i < 10; ++i)
	{
		if(m_ChannelsUsed[i] != 0)
			SAFE_DELETE(m_VertexDataChannels[i]);
	}
}

bool COGLVertexArray::Init(GLenum primitiveType)
{
	V_RET_FOF(COGLResource::Init());

	m_PrimitiveType = primitiveType;

	glGenVertexArrays(1, &m_Resource);

	V_RET_FOT(CheckGLError(m_DebugName, "COGLVertexArray::Init()"));

	return true;
}

void COGLVertexArray::Release()
{
	COGLResource::Release();

	for(uint i = 0; i < 10; ++i)
	{
		if(m_ChannelsUsed[i] != 0)
			m_VertexDataChannels[i]->Release();
	}

	if(m_HasIndexData)
		m_pGLVBIndexData->Release();

	glDeleteVertexArrays(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLVertexArray::Release()");
}

bool COGLVertexArray::AddVertexDataChannel(uint index, uint elements)
{
	CheckInitialized("COGLVertexArray::AddVertexDataChannel()");
	CheckNotBound("COGLVertexArray::AddVertexDataChannel()");

	CHANNEL_INFO info;
	info.index = index;
	info.elements = elements;
	
	m_VertexDataChannelInfo[index] = info;
	m_ChannelsUsed[index] = 1;

	COGLVertexBuffer* pBuffer = new COGLVertexBuffer("COGLVertexArray.pBuffer");
	V_RET_FOF(pBuffer->Init());

	m_VertexDataChannels[index] = pBuffer;

	return true;
}

bool COGLVertexArray::AddVertexData(uint index, uint size, void* pData)
{
	CheckInitialized("COGLVertexArray::AddVertexData()");
	CheckNotBound("COGLVertexArray::AddVertexData()");
		
	V_RET_FOF(m_VertexDataChannels[index]->SetContent(size, pData, GL_STATIC_DRAW));
	
	return true;
}

bool COGLVertexArray::AddIndexDataChannel()
{
	CheckInitialized("COGLVertexArray::AddIndexData()");
	CheckNotBound("COGLVertexArray::AddIndexData()");

	V_RET_FOF(m_pGLVBIndexData->Init());
	
	CheckGLError(m_DebugName, "COGLVertexArray::AddIndexDataChannel()");
	
	m_HasIndexData = true;

	return true;
}

bool COGLVertexArray::AddIndexData(GLuint size, void* pData)
{
	CheckInitialized("COGLVertexArray::AddIndexData()");
	CheckNotBound("COGLVertexArray::AddIndexData()");

	V_RET_FOF(m_pGLVBIndexData->SetContent(size, pData, GL_STATIC_DRAW));

	CheckGLError(m_DebugName, "COGLVertexArray::AddIndexData()");
	
	return true;
}

void COGLVertexArray::Finish()
{
	CheckInitialized("COGLVertexArray::Finish()");
	CheckNotBound("COGLVertexArray::Finish()");
	
	if(!m_HasIndexData)
	{
		std::cout << "COGLVertexArray: " << m_DebugName 
			<< " has no index data!!!"	<< std::endl;
	}

	COGLBindLock lock(this, COGL_VERTEX_ARRAY_SLOT);
	
	for(uint i = 0; i < 10; ++i)
	{
		if(m_ChannelsUsed[i] == 0)
			continue;

		COGLVertexBuffer* pBuffer = m_VertexDataChannels[i];
					
		CHANNEL_INFO info = m_VertexDataChannelInfo[i];
			
		COGLBindLock lock(pBuffer, COGL_ARRAY_BUFFER_SLOT);
		glEnableVertexAttribArray(info.index);
		glVertexAttribPointer(info.index, info.elements, GL_FLOAT, GL_FALSE, 0, 0);
	}

	COGLBindLock lockIndexData(m_pGLVBIndexData, COGL_ELEMENT_ARRAY_BUFFER_SLOT);
	
	CheckGLError(m_DebugName, "COGLVertexArray::Finish()");

	glBindVertexArray(0);
}

void COGLVertexArray::Draw(GLuint count)
{
	CheckInitialized("COGLVertexArray::Draw()");
	CheckNotBound("COGLVertexArray::Draw()");
	
	CheckGLError(m_DebugName, "COGLVertexArray::Draw() before glDrawElements");

	COGLBindLock lock(this, COGL_VERTEX_ARRAY_SLOT);

	glDrawElements(m_PrimitiveType, count, GL_UNSIGNED_INT, 0);

	CheckGLError(m_DebugName, "COGLVertexArray::Draw() after glDrawElements");
}

void COGLVertexArray::Bind(COGLBindSlot slot)
{	
	COGLResource::Bind(slot);

	assert(m_Slot == COGL_VERTEX_ARRAY_SLOT);

	glBindVertexArray(m_Resource);
}

void COGLVertexArray::Unbind()
{
	COGLResource::Unbind();

	assert(m_Slot == COGL_VERTEX_ARRAY_SLOT);

	glBindVertexArray(0);
}
