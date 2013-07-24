#include "CMaterialBuffer.h"

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
}

CMaterialBuffer::~CMaterialBuffer()
{
	m_MapMatNameToIndex.clear();
	m_Materials.clear();
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

bool CMaterialBuffer::FillOGLMaterialBuffer()
{
	m_oglMaterialBuffer.reset(new COGLTextureBuffer(GL_RGBA32F));
	m_oglMaterialBuffer->SetContent(m_Materials.size() * sizeof(MATERIAL), GL_STATIC_DRAW, m_Materials.data());
	
	return true;
}
