#include "COGLVertexArray.h"

#include "COGLVertexBuffer.h"
#include "COGLBindLock.h"

#include "..\Macros.h"

#include "..\Utils\GLErrorUtil.h"

#include <assert.h>
#include <iostream>


COGLVertexArray::COGLVertexArray(GLenum primitiveType, std::string const& debugName)
	: COGLResource(COGL_VERTEXARRAY, debugName), m_HasIndexData(false)
{
	m_PrimitiveType = primitiveType;

	glGenVertexArrays(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLVertexArray::COGLVertexArray()");
}

COGLVertexArray::~COGLVertexArray()
{
	glDeleteVertexArrays(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLVertexArray::~COGLVertexArray()");
}

bool COGLVertexArray::AddVertexData(uint index, uint elements, uint size, void* pData)
{
	CheckNotBound("COGLVertexArray::AddVertexDataChannel()");

	CHANNEL_INFO info;
	info.index = index;
	info.elements = elements;
	
	m_VertexDataChannelInfo.push_back(info);
	m_VertexDataChannels.emplace_back(std::unique_ptr<COGLVertexBuffer>(new COGLVertexBuffer()));
	
	V_RET_FOF(m_VertexDataChannels.back()->SetContent(size, pData, GL_STATIC_DRAW));
	
	return true;
}

bool COGLVertexArray::AddIndexData(GLuint size, void* pData)
{
	CheckNotBound("COGLVertexArray::AddIndexData()");

	m_IndexData.reset(new COGLVertexBuffer());
	V_RET_FOF(m_IndexData->SetContent(size, pData, GL_STATIC_DRAW));

	CheckGLError(m_DebugName, "COGLVertexArray::AddIndexData()");
	
	m_HasIndexData = true;

	return true;
}

void COGLVertexArray::Finish()
{
	CheckNotBound("COGLVertexArray::Finish()");
	
	if(!m_HasIndexData)
	{
		std::cout << "COGLVertexArray: " << m_DebugName 
			<< " has no index data!!!"	<< std::endl;
	}

	COGLBindLock lock(this, COGL_VERTEX_ARRAY_SLOT);
	
	for(uint i = 0; i < m_VertexDataChannels.size(); ++i)
	{				
		CHANNEL_INFO const& info = m_VertexDataChannelInfo[i];
			
		COGLBindLock lock(m_VertexDataChannels[i].get(), COGL_ARRAY_BUFFER_SLOT);
		glEnableVertexAttribArray(info.index);
		glVertexAttribPointer(info.index, info.elements, GL_FLOAT, GL_FALSE, 0, 0);
	}

	COGLBindLock lockIndexData(m_IndexData.get(), COGL_ELEMENT_ARRAY_BUFFER_SLOT);
	
	CheckGLError(m_DebugName, "COGLVertexArray::Finish()");

	glBindVertexArray(0);
}

void COGLVertexArray::Draw(GLuint count)
{
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
