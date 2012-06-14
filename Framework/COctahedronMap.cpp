#include "COctahedronMap.h"

#include <glm/glm.hpp>

#include "Defines.h"
#include "Macros.h"

#include "OGLResources\COGLTexture2D.h"

/*
	face layout

	|---------|
	|6 / | \ 5|
	| /2 | 1\ |
	|---------|
	| \3 | 4/ |
	|7 \ | / 8|
	|---------|
*/

COctahedronMap::COctahedronMap()
{
	m_pTexture = new COGLTexture2D("COctahedronMap.m_pTexture");
}

COctahedronMap::~COctahedronMap()
{
	SAFE_DELETE(m_pTexture);
}

bool COctahedronMap::Init(uint dimension)
{
	m_Dimension = dimension;
	
	V_RET_FOF(m_pTexture->Init(m_Dimension, m_Dimension, GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));

	return true;
}

void COctahedronMap::Release()
{
	m_pTexture->Release();
}

COGLTexture2D* COctahedronMap::GetTexture()
{
	m_pTexture->CheckInitialized("COctahedronMap.GetTexture()");

	return m_pTexture;
}

void COctahedronMap::FillWithDebugData()
{
	m_pTexture->CheckInitialized("COctahedronMap.GetTexture()");

	glm::vec4* pData = new glm::vec4[m_Dimension * m_Dimension];
	memset(pData, 0, sizeof(glm::vec4) * m_Dimension * m_Dimension);

	
	for(uint x = 1; x < m_Dimension - 1; ++x)
	{
		for(uint y = 1; y < m_Dimension - 1; ++y)
		{
			if(x < m_Dimension / 2 && y < m_Dimension / 2)
			{
				if(x < m_Dimension / 2 - y) {
					// fill face 7
					*AccessMap(x, y, pData) = glm::vec4(0.f, 0.f, 1.f, 1.f);
				}
				else{
					// fill face 3
					*AccessMap(x, y, pData) = glm::vec4(1.f, 0.f, 0.f, 1.f);
				}
			}
			else if(x < m_Dimension / 2 && y >= m_Dimension / 2)
			{
				if(x >= y - m_Dimension / 2) {
					// fill face 2
					*AccessMap(x, y, pData) = glm::vec4(1.f, 0.f, 1.f, 1.f);
				}
				else{
					// fill face 6
					*AccessMap(x, y, pData) = glm::vec4(1.f, 1.f, 1.f, 1.f);
				}
			}
			else if(x >= m_Dimension / 2 && y < m_Dimension / 2)
			{
				if(x - m_Dimension / 2 >= y) {
					// fill face 8
					*AccessMap(x, y, pData) = glm::vec4(1.f, 1.f, 0.f, 1.f);
				}
				else{
					// fill face 4
					*AccessMap(x, y, pData) = glm::vec4(0.f, 1.f, 0.f, 1.f);
				}
			}
			else
			{
				if(x - m_Dimension/2 < m_Dimension - y) {
					// fill face 1
					*AccessMap(x, y, pData) = glm::vec4(0.f, 0.f, 0.f, 1.f);
				}
				else{
					// fill face 5
					*AccessMap(x, y, pData) = glm::vec4(0.f, 1.f, 1.f, 1.f);
				}
			}
			
		}
	}

	// top border
	for(uint x = 1; x < m_Dimension - 1; ++x)
	{
		*AccessMap(x, m_Dimension-1, pData) = *AccessMap(m_Dimension-1 - x, m_Dimension-2, pData);
	}

	// bottom border
	for(uint x = 1; x < m_Dimension - 1; ++x)
	{
		*AccessMap(x, 0, pData) = *AccessMap(m_Dimension-1 - x, 1, pData);
	}

	// left border
	for(uint y = 1; y < m_Dimension - 1; ++y)
	{
		*AccessMap(0, y, pData) = *AccessMap(1, m_Dimension-1 - y, pData);
	}

	// right border
	for(uint y = 1; y < m_Dimension - 1; ++y)
	{
		*AccessMap(m_Dimension - 1,	y, pData) = *AccessMap(m_Dimension - 2, m_Dimension-1 - y, pData);
	}

	// corners
	*AccessMap(0,				0,					pData) = *AccessMap(m_Dimension-2,	m_Dimension-2,	pData);
	*AccessMap(m_Dimension - 1, m_Dimension - 1,	pData) = *AccessMap(1,				1,				pData);
	*AccessMap(0,				m_Dimension - 1,	pData) = *AccessMap(m_Dimension-2,	1,				pData);
	*AccessMap(m_Dimension - 1, 0,					pData) = *AccessMap(1,				m_Dimension-2,	pData);
	
	m_pTexture->SetPixelData((void*)pData);

	delete [] pData;
}

glm::vec4* COctahedronMap::AccessMap(uint x, uint y, glm::vec4* pMap)
{
	return &(pMap[y * m_Dimension + x]);
}