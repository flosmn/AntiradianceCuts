#include "CSubModel.h"

#include "CMesh.h"
#include "CMeshGeometry.h"

#include "..\Macros.h"
#include "..\Structs.h"

#include "..\Utils\GLErrorUtil.h"

#include "..\CMaterialBuffer.h"

#include "..\OGLResources\COGLVertexBuffer.h"
#include "..\OGLResources\COGLVertexArray.h"
#include "..\OGLResources\COGLBindLock.h"
#include "..\OGLResources\COGLUniformBuffer.h"


CSubModel::CSubModel(CMesh* mesh)
{
	m_vertexArray.reset(new COGLVertexArray(GL_TRIANGLES, "vertexData"));

	uint nVertices = mesh->GetNumberOfVertices();
	m_nTriangles = mesh->GetNumberOfFaces();
	
	GLuint positionDataSize = sizeof(glm::vec4) * nVertices;
	GLuint normalDataSize = sizeof(glm::vec3) * nVertices;
	GLuint indexDataSize = sizeof(GLuint) * 3 * m_nTriangles;
	
	m_vertexArray->AddVertexData(0, 4, positionDataSize, (void*)mesh->GetPositionData());
	m_vertexArray->AddVertexData(1, 3, normalDataSize, (void*)mesh->GetNormalData());
	m_vertexArray->AddIndexData(indexDataSize, (void*)mesh->GetIndexData());
	
	m_vertexArray->Finish();

	CreateTriangleData(mesh->GetNumberOfFaces(), mesh->GetIndexData(),
		mesh->GetPositionData(), mesh->GetNormalData());
}

CSubModel::CSubModel(const CMeshGeometry& meshGeometry, CMaterialBuffer* materialBuffer)
{
	m_vertexArray.reset(new COGLVertexArray(GL_TRIANGLES, "vertexData"));

	m_materialBuffer = materialBuffer;
	uint nVertices = meshGeometry.GetNumberOfVertices();
	m_nTriangles = meshGeometry.GetNumberOfFaces();
	m_MaterialIndex = meshGeometry.GetMaterialIndex();

	GLuint positionDataSize = sizeof(glm::vec4) * nVertices;
	GLuint normalDataSize = sizeof(glm::vec3) * nVertices;
	GLuint indexDataSize = sizeof(GLuint) * 3 * m_nTriangles;

	m_vertexArray->AddVertexData(0, 4, positionDataSize, (void*)meshGeometry.GetPositionData());
	m_vertexArray->AddVertexData(1, 3, normalDataSize, (void*)meshGeometry.GetNormalData());
	m_vertexArray->AddIndexData(indexDataSize, (void*)meshGeometry.GetIndexData());
	
	m_vertexArray->Finish();

	CreateTriangleData(meshGeometry.GetNumberOfFaces(), meshGeometry.GetIndexData(),
		meshGeometry.GetPositionData(), meshGeometry.GetNormalData());
}

CSubModel::~CSubModel()
{
}

void CSubModel::Draw(COGLUniformBuffer* ubMaterial) 
{
	MATERIAL material = *(m_materialBuffer->GetMaterial(m_MaterialIndex));
	ubMaterial->UpdateData(&material);
	
	Draw();
}

void CSubModel::Draw() 
{
	m_vertexArray->Draw(3 * m_nTriangles);
}

std::vector<Triangle>& CSubModel::GetTrianglesOS()
{
	return m_trianglesOS;
}

std::vector<Triangle>& CSubModel::GetTrianglesWS()
{
	return m_trianglesWS;
}

void CSubModel::SetWorldTransform(const glm::mat4& transform)
{
	m_worldTransform = transform;
	m_trianglesWS.clear();
	
	for(uint i = 0; i < m_trianglesOS.size(); ++i)
	{
		const Triangle& triangleOS = m_trianglesOS[i];
		Triangle triangleWS(
			glm::vec3(m_worldTransform * glm::vec4(triangleOS.P0(), 1.f)),
			glm::vec3(m_worldTransform * glm::vec4(triangleOS.P1(), 1.f)),
			glm::vec3(m_worldTransform * glm::vec4(triangleOS.P2(), 1.f)));
		triangleWS.setMaterialIndex(m_MaterialIndex);
		m_trianglesWS.push_back(triangleWS);
	}	
}


void CSubModel::CreateTriangleData(uint nFaces, const uint* pIndexData, 
	const glm::vec4* pPositionData, const glm::vec3* pNormalData)
{
	m_trianglesOS.clear();
	m_trianglesWS.clear();

	for(uint i = 0; i < nFaces; i++)
	{
		uint i1 = pIndexData[i * 3 + 0];
		uint i2 = pIndexData[i * 3 + 1];
		uint i3 = pIndexData[i * 3 + 2];
		glm::vec3 p1 = glm::vec3(pPositionData[i1]);
		glm::vec3 p2 = glm::vec3(pPositionData[i2]);
		glm::vec3 p3 = glm::vec3(pPositionData[i3]);

		m_trianglesOS.push_back(Triangle(p1, p2, p3));
		m_trianglesOS.back().setMaterialIndex(m_MaterialIndex);
	}
} 
