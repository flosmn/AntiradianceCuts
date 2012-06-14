#include "CMeshGeometry.h"

#include "..\Macros.h"

#include "CMeshMaterial.h"

#include "..\Utils\Util.h"

#include "..\Triangle.h"

#include <iostream>
#include <set>
#include <vector>
#include <map>

CMeshGeometry::CMeshGeometry()
{
	m_pMeshMaterial = nullptr;
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

void CMeshGeometry::CreateTriangleData(std::vector<Triangle*>& triangles)
{
	std::vector<Triangle*>::iterator it;
	for (it = triangles.begin() ; it < triangles.end(); it++ )
	{
		SAFE_DELETE(*it);
	}
	triangles.clear();
		
	for(uint i = 0; i < m_nFaces; i++)
	{
		uint i1 = m_pIndexData[i * 3 + 0];
		uint i2 = m_pIndexData[i * 3 + 1];
		uint i3 = m_pIndexData[i * 3 + 2];
		glm::vec3 p1 = glm::vec3(m_pPositionData[i1]);
		glm::vec3 p2 = glm::vec3(m_pPositionData[i2]);
		glm::vec3 p3 = glm::vec3(m_pPositionData[i3]);

		glm::vec3 normal = glm::normalize(glm::cross(p3-p1, p2-p1));

		Triangle* triangle = new Triangle(p1, p2, p3, normal);
		triangles.push_back(triangle);														 
	}
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
			ushort i0 = m_pIndexData[3 * i + 0];
			ushort i1 = m_pIndexData[3 * i + 1];
			ushort i2 = m_pIndexData[3 * i + 2];
			std::cout << "f: (" << i0 << ", " << i1 << ", " << i2 << ")" << std::endl;
		}
	}

	if(!m_pMeshMaterial)
	{
		std::cout << "no material data" << std::endl;
	}
	else
	{
		std::cout << "material: " << m_pMeshMaterial->GetMaterialName() << std::endl;
	}
}