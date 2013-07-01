#include "CMeshGeometry.h"

#include "..\Macros.h"
#include "..\Utils\Util.h"
#include "..\Triangle.h"

#include <iostream>
#include <set>
#include <vector>
#include <map>

CMeshGeometry::CMeshGeometry()
{
	m_pPositionData = nullptr;
	m_pNormalData = nullptr;
	m_pTexCoordData = nullptr;
	m_pIndexData = nullptr;
}

CMeshGeometry::~CMeshGeometry()
{
	SAFE_DELETE_ARRAY(m_pIndexData);
	SAFE_DELETE_ARRAY(m_pPositionData);
	SAFE_DELETE_ARRAY(m_pNormalData);
	SAFE_DELETE_ARRAY(m_pTexCoordData);
}

void CMeshGeometry::PrintGeometryData()
{
	std::cout << "#vertices " << m_nVertices << std::endl;

	if(m_pPositionData == 0)
	{
		std::cout << "no position data" << std::endl;
	}
	else
	{		
		for(uint i = 0; i < m_nVertices; ++i)
		{
			glm::vec4 p = m_pPositionData[i];
			std::cout << "v: (" << AsString(p) << ")" << std::endl;
		}
	}

	if(m_pNormalData == 0)
	{
		std::cout << "no position data" << std::endl;
	}
	else
	{
		for(uint i = 0; i < m_nVertices; ++i)
		{
			glm::vec3 p = m_pNormalData[i];
			std::cout << "n: (" << AsString(p) << ")" << std::endl;
		}
	}

	if(m_pTexCoordData == 0)
	{
		std::cout << "no texcoord data" << std::endl;
	}
	else
	{
		for(uint i = 0; i < m_nVertices; ++i)
		{
			glm::vec3 p = m_pTexCoordData[i];
			std::cout << "v: (" << AsString(p) << ")" << std::endl;
		}
	}

	std::cout << "#faces " << m_nFaces << std::endl;
	
	if(m_pIndexData == 0)
	{
		std::cout << "no face data" << std::endl;
	}
	else
	{
		for(uint i = 0; i < m_nFaces; ++i)
		{
			uint i0 = m_pIndexData[3 * i + 0];
			uint i1 = m_pIndexData[3 * i + 1];
			uint i2 = m_pIndexData[3 * i + 2];
			std::cout << "f: (" << i0 << ", " << i1 << ", " << i2 << ")" << std::endl;
		}
	}

	std::cout << "material index: " << m_MaterialIndex << std::endl;
}