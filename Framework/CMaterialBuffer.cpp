#include "CMaterialBuffer.h"

#include "Macros.h"

#include "OGLResources\COGLTextureBuffer.h"

/*
material index is counted from 1 to numMaterials.

on index 0 in the material buffer a dummy material is inserted.

material index 0 means that this material does not exist jet
*/

CMaterialBuffer::CMaterialBuffer()
{
	MATERIAL mat;
	m_Materials.push_back(mat);
	m_pOGLMaterialBuffer = new COGLTextureBuffer("CMaterialBuffer.m_pOGLMaterialBuffer");
}

CMaterialBuffer::~CMaterialBuffer()
{
	m_MapMatNameToIndex.clear();
	m_Materials.clear();

	SAFE_DELETE(m_pOGLMaterialBuffer);
}

// add the material mat to the material buffer and returns the index of the material in the buffer
int CMaterialBuffer::AddMaterial(const std::string& name, const MATERIAL& mat)
{
	if(m_MapMatNameToIndex[name] != 0)
	{
		return GetIndexOfMaterial(name);
	}
	else
	{	
		const int index = (int)m_Materials.size();
		m_Materials.push_back(mat);

		m_MapMatNameToIndex[name] = index;
		return index;
	}
}

int CMaterialBuffer::GetIndexOfMaterial(const std::string& name)
{
	return m_MapMatNameToIndex[name];
}

// returns the material with the index i
MATERIAL* CMaterialBuffer::GetMaterial(int i)
{
	return &(m_Materials[i]);
}

bool CMaterialBuffer::InitOGLMaterialBuffer()
{
	V_RET_FOF(m_pOGLMaterialBuffer->Init((m_Materials.size()+1) * sizeof(MATERIAL), GL_STATIC_DRAW, GL_RGBA32F));

	MATERIAL* pData = new MATERIAL[(m_Materials.size()+1)];
	memset(pData, 0, (m_Materials.size()+1) * sizeof(MATERIAL));
	for(int i = 0; i < (m_Materials.size()+1); ++i)
	{
		pData[i] = m_Materials[i];
	}
	m_pOGLMaterialBuffer->SetContent(pData, (m_Materials.size()+1) * sizeof(MATERIAL));
	delete [] pData;

	return true;
}

void CMaterialBuffer::ReleaseOGLMaterialBuffer()
{
	m_pOGLMaterialBuffer->Release();
}

COGLTextureBuffer* CMaterialBuffer::GetOGLMaterialBuffer()
{
	m_pOGLMaterialBuffer->CheckInitialized("CMaterialBuffer.GetOGLMaterialBuffer()");
	return m_pOGLMaterialBuffer;
}