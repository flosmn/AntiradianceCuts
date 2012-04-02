#ifndef _C_MESH_GEOMETRY_H_
#define _C_MESH_GEOMETRY_H_

typedef unsigned int uint;
typedef unsigned short ushort;

#include "glm/glm.hpp"

#include <vector>

class CMeshMaterial;
class Triangle;

class CMeshVertex
{
public:
	CMeshVertex() 
	{
		positionDataIndex = 0;
		normalDataIndex = 0;
		texCoordDataIndex = 0;
	}

	bool operator<(const CMeshVertex& rhs) const
	{
		if(positionDataIndex < rhs.positionDataIndex)	return true;
		if(positionDataIndex > rhs.positionDataIndex)	return false;
		if(normalDataIndex < rhs.normalDataIndex)		return true;
		if(normalDataIndex > rhs.normalDataIndex)		return false;
		if(texCoordDataIndex < rhs.texCoordDataIndex)	return true;
		if(texCoordDataIndex > rhs.texCoordDataIndex)	return false;
		return false;
	}

	bool operator<(const CMeshVertex* rhs) const
	{
		if(positionDataIndex < rhs->positionDataIndex)	return true;
		if(positionDataIndex > rhs->positionDataIndex)	return false;
		if(normalDataIndex < rhs->normalDataIndex)		return true;
		if(normalDataIndex > rhs->normalDataIndex)		return false;
		if(texCoordDataIndex < rhs->texCoordDataIndex)	return true;
		if(texCoordDataIndex > rhs->texCoordDataIndex)	return false;
		return false;
	}
	
	int positionDataIndex;
	int normalDataIndex;
	int texCoordDataIndex;
};

class CMeshTriangleFace
{
public:
	CMeshVertex* vertex0;
	CMeshVertex* vertex1;
	CMeshVertex* vertex2;
};

struct CMeshVertex_Compare {
    bool operator() (const CMeshVertex& lhs, const CMeshVertex& rhs) const{
		if(lhs.positionDataIndex < rhs.positionDataIndex)	return true;
		if(lhs.positionDataIndex > rhs.positionDataIndex)	return false;
		if(lhs.normalDataIndex < rhs.normalDataIndex)		return true;
		if(lhs.normalDataIndex > rhs.normalDataIndex)		return false;
		if(lhs.texCoordDataIndex < rhs.texCoordDataIndex)	return true;
		if(lhs.texCoordDataIndex > rhs.texCoordDataIndex)	return false;
		return false;
    }

	bool operator() (const CMeshVertex* lhs, const CMeshVertex* rhs) const{
		if(lhs->positionDataIndex < rhs->positionDataIndex)	return true;
		if(lhs->positionDataIndex > rhs->positionDataIndex)	return false;
		if(lhs->normalDataIndex < rhs->normalDataIndex)		return true;
		if(lhs->normalDataIndex > rhs->normalDataIndex)		return false;
		if(lhs->texCoordDataIndex < rhs->texCoordDataIndex)	return true;
		if(lhs->texCoordDataIndex > rhs->texCoordDataIndex)	return false;
		return false;
    }
};

class CMeshGeometry
{
public:
	CMeshGeometry();
	~CMeshGeometry();
		
	// sets a data vector to NULL if no data of that kind is provided
	const glm::vec4* GetPositionData() { return m_pPositionData; }
	const glm::vec3* GetNormalData() { return m_pNormalData; }
	const glm::vec3* GetTexCoordData() { return m_pTexCoordData; }
	const ushort* GetIndexData() { return m_pIndexData; }

	void SetPositionData(glm::vec4* pPositionData) { m_pPositionData = pPositionData; }
	void SetNormalData(glm::vec3* pNormalData) { m_pNormalData = pNormalData; }
	void SetTexCoordData(glm::vec3* pTexCoordData) { m_pTexCoordData = pTexCoordData; }
	void SetIndexData(ushort* pIndexData) { m_pIndexData = pIndexData; }

	uint GetNumberOfVertices() { return m_nVertices; }
	uint GetNumberOfFaces() { return m_nFaces; }

	void SetNumberOfVertices(uint nVertices) { m_nVertices = nVertices; }
	void SetNumberOfFaces(uint nFaces) { m_nFaces = nFaces; }

	CMeshMaterial* GetMeshMaterial() { return m_pMeshMaterial; }
	void SetMeshMaterial(CMeshMaterial* pMeshMaterial) { m_pMeshMaterial = pMeshMaterial; }

	void CreateTriangleData(std::vector<Triangle*>& triangles);

	void PrintGeometryData();
	
private:
	CMeshMaterial* m_pMeshMaterial;
		
	uint m_nVertices;
	uint m_nFaces;
	glm::vec4* m_pPositionData;
	glm::vec3* m_pNormalData;
	glm::vec3* m_pTexCoordData;
	ushort* m_pIndexData;	
};

#endif // _C_MESH_GEOMETRY_H_