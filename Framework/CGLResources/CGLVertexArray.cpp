#include "CGLVertexArray.h"

#include "CGLVertexBuffer.h"
#include "CGLBindLock.h"

#include "..\Macros.h"

#include "..\CUtils\GLErrorUtil.h"

#include <assert.h>
#include <iostream>


CGLVertexArray::CGLVertexArray(std::string debugName)
	: CGLResource(CGL_VERTEXARRAY, debugName), m_pGLVBPositionData(nullptr), 
	  m_pGLVBNormalData(nullptr), m_pGLVBIndexData(nullptr),
	  m_HasPositionData(false), m_HasNormalData(false), m_HasTextureData(false),
	  m_HasIndexData(false)
{
	m_pGLVBPositionData = new CGLVertexBuffer("CGLVertexArray.m_pGLVBPositionData");
	m_pGLVBNormalData = new CGLVertexBuffer("CGLVertexArray.m_pGLVBNormalData");
	m_pGLVBTextureData = new CGLVertexBuffer("CGLVertexArray.m_pGLVBTextureData");
	m_pGLVBIndexData = new CGLVertexBuffer("CGLVertexArray.m_pGLVBIndexData");
}

CGLVertexArray::~CGLVertexArray()
{
	SAFE_DELETE(m_pGLVBPositionData);
	SAFE_DELETE(m_pGLVBNormalData);
	SAFE_DELETE(m_pGLVBTextureData);
	SAFE_DELETE(m_pGLVBIndexData);
}

bool CGLVertexArray::Init()
{
	V_RET_FOF(CGLResource::Init());

	glGenVertexArrays(1, &m_Resource);

	V_RET_FOT(CheckGLError(m_DebugName, "CGLVertexArray::Init()"));

	return true;
}

void CGLVertexArray::Release()
{
	CGLResource::Release();

	if(m_HasPositionData)
		m_pGLVBPositionData->Release();

	if(m_HasNormalData)
		m_pGLVBNormalData->Release();

	if(m_HasTextureData)
		m_pGLVBTextureData->Release();

	if(m_HasIndexData)
		m_pGLVBIndexData->Release();

	glDeleteVertexArrays(1, &m_Resource);

	CheckGLError(m_DebugName, "CGLVertexArray::Release()");
}

bool CGLVertexArray::AddPositionData(GLuint size, void* pData)
{
	CheckInitialized("CGLVertexArray::AddPositionData()");
	CheckNotBound("CGLVertexArray::AddPositionData()");

	V_RET_FOF(m_pGLVBPositionData->Init(size, pData, GL_STATIC_DRAW));

	CheckGLError(m_DebugName, "CGLVertexArray::AddPositionData()");

	m_HasPositionData = true;

	return true;
}

bool CGLVertexArray::AddNormalData(GLuint size, void* pData)
{
	CheckInitialized("CGLVertexArray::AddNormalData()");
	CheckNotBound("CGLVertexArray::AddNormalData()");
	
	V_RET_FOF(m_pGLVBNormalData->Init(size, pData, GL_STATIC_DRAW));

	CheckGLError(m_DebugName, "CGLVertexArray::AddNormalData()");

	m_HasNormalData = true;
	
	return true;
}

bool CGLVertexArray::AddTextureData(GLuint size, void* pData)
{
	CheckInitialized("CGLVertexArray::AddTextureData()");
	CheckNotBound("CGLVertexArray::AddTextureData()");

	V_RET_FOF(m_pGLVBTextureData->Init(size, pData, GL_STATIC_DRAW));

	CheckGLError(m_DebugName, "CGLVertexArray::AddTextureData()");

	m_HasTextureData = true;

	return true;
}

bool CGLVertexArray::AddIndexData(GLuint size, void* pData)
{
	CheckInitialized("CGLVertexArray::AddIndexData()");
	CheckNotBound("CGLVertexArray::AddIndexData()");

	V_RET_FOF(m_pGLVBIndexData->Init(size, pData, GL_STATIC_DRAW));

	CheckGLError(m_DebugName, "CGLVertexArray::AddIndexData()");

	m_HasIndexData = true;

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
	
	if(m_HasPositionData)
	{
		CGLBindLock lockPositionData(m_pGLVBPositionData, CGL_ARRAY_BUFFER_SLOT);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
	}

	if(m_HasNormalData)
	{
		CGLBindLock lockNormalData(m_pGLVBNormalData, CGL_ARRAY_BUFFER_SLOT);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	}

	if(m_HasTextureData)
	{
		CGLBindLock lockTextureData(m_pGLVBTextureData, CGL_ARRAY_BUFFER_SLOT);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
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

	glDrawElements(GL_TRIANGLES, count, GL_UNSIGNED_SHORT, 0);

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