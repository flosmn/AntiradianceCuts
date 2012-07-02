#include "CSubModel.h"

#include "CMesh.h"
#include "CMeshGeometry.h"
#include "CMeshMaterial.h"

#include "..\Macros.h"
#include "..\Structs.h"
#include "..\Camera.h"

#include "..\Utils\GLErrorUtil.h"

#include "..\OGLResources\COGLVertexBuffer.h"
#include "..\OGLResources\COGLVertexArray.h"
#include "..\OGLResources\COGLBindLock.h"
#include "..\OGLResources\COGLUniformBuffer.h"


CSubModel::CSubModel()
	: m_pGLVARenderData(nullptr)
{
	m_pGLVARenderData = new COGLVertexArray("CSubModel.m_pGLVARenderData");
}

CSubModel::~CSubModel()
{
	SAFE_DELETE(m_pGLVARenderData);
};

bool CSubModel::Init(CMeshGeometry* pMeshGeometry) 
{
	V_RET_FOF(m_pGLVARenderData->Init(GL_TRIANGLES));

	uint nVertices = pMeshGeometry->GetNumberOfVertices();
	m_nTriangles = pMeshGeometry->GetNumberOfFaces();
	m_pMaterial = pMeshGeometry->GetMeshMaterial();

	GLuint positionDataSize = sizeof(glm::vec4) * nVertices;
	GLuint normalDataSize = sizeof(glm::vec3) * nVertices;
	GLuint indexDataSize = sizeof(GLshort) * 3 * m_nTriangles;
	
	V_RET_FOF(m_pGLVARenderData->AddVertexDataChannel(0, 4));
	V_RET_FOF(m_pGLVARenderData->AddVertexData(0, positionDataSize, (void*)pMeshGeometry->GetPositionData()));

	V_RET_FOF(m_pGLVARenderData->AddVertexDataChannel(1, 3));
	V_RET_FOF(m_pGLVARenderData->AddVertexData(1, normalDataSize, (void*)pMeshGeometry->GetNormalData()));

	V_RET_FOF(m_pGLVARenderData->AddIndexDataChannel());
	V_RET_FOF(m_pGLVARenderData->AddIndexData(indexDataSize, (void*)pMeshGeometry->GetIndexData()));
	
	m_pGLVARenderData->Finish();

	CreateTriangleData(pMeshGeometry->GetNumberOfFaces(), pMeshGeometry->GetIndexData(),
		pMeshGeometry->GetPositionData(), pMeshGeometry->GetNormalData());

	return true;
}

bool CSubModel::Init(CMesh* pMesh) 
{
	V_RET_FOF(m_pGLVARenderData->Init(GL_TRIANGLES));

	uint nVertices = pMesh->GetNumberOfVertices();
	m_nTriangles = pMesh->GetNumberOfFaces();
	m_pMaterial = new CMeshMaterial();

	GLuint positionDataSize = sizeof(glm::vec4) * nVertices;
	GLuint normalDataSize = sizeof(glm::vec3) * nVertices;
	GLuint indexDataSize = sizeof(GLshort) * 3 * m_nTriangles;
	
	V_RET_FOF(m_pGLVARenderData->AddVertexDataChannel(0, 4));
	V_RET_FOF(m_pGLVARenderData->AddVertexData(0, positionDataSize, (void*)pMesh->GetPositionData()));

	V_RET_FOF(m_pGLVARenderData->AddVertexDataChannel(1, 3));
	V_RET_FOF(m_pGLVARenderData->AddVertexData(1, normalDataSize, (void*)pMesh->GetNormalData()));

	V_RET_FOF(m_pGLVARenderData->AddIndexDataChannel());
	V_RET_FOF(m_pGLVARenderData->AddIndexData(indexDataSize, (void*)pMesh->GetIndexData()));
	
	m_pGLVARenderData->Finish();

	CreateTriangleData(pMesh->GetNumberOfFaces(), pMesh->GetIndexData(),
		pMesh->GetPositionData(), pMesh->GetNormalData());
	
	return true;
}

void CSubModel::Release()
{
	m_pGLVARenderData->Release();

	for(uint i = 0; i < m_TrianglesOS.size(); ++i)
	{
		if(m_TrianglesOS[i])
			delete m_TrianglesOS[i];
	}

	for(uint i = 0; i < m_TrianglesWS.size(); ++i)
	{
		if(m_TrianglesWS[i])
			delete m_TrianglesWS[i];
	}

	m_TrianglesWS.clear();
	m_TrianglesOS.clear();
}

void CSubModel::Draw(COGLUniformBuffer* pUBMaterial) 
{
	MATERIAL material = m_pMaterial->GetMaterialData();
	pUBMaterial->UpdateData(&material);
	
	Draw();
}

void CSubModel::Draw() 
{
	m_pGLVARenderData->Draw(3 * m_nTriangles);
}

const std::vector<CTriangle*>& CSubModel::GetTrianglesOS() const
{
	return m_TrianglesOS;
}

const std::vector<CTriangle*>& CSubModel::GetTrianglesWS() const
{
	return m_TrianglesWS;
}

void CSubModel::SetMaterial(const MATERIAL& mat)
{
	m_pMaterial->SetMaterialData(mat); 
}

MATERIAL CSubModel::GetMaterial() const
{
	return m_pMaterial->GetMaterialData(); 
}

void CSubModel::SetWorldTransform(const glm::mat4& transform)
{
	m_WorldTransform = transform;

	for(uint i = 0; i < m_TrianglesWS.size(); ++i)
	{
		if(m_TrianglesWS[i])
			delete m_TrianglesWS[i];
	}
	m_TrianglesWS.clear();
	
	for(uint i = 0; i < m_TrianglesOS.size(); ++i)
	{
		const CTriangle& triangleOS = *m_TrianglesOS[i];
		CTriangle* triangleWS = new CTriangle(
			glm::vec3(m_WorldTransform * glm::vec4(triangleOS.P0(), 1.f)),
			glm::vec3(m_WorldTransform * glm::vec4(triangleOS.P1(), 1.f)),
			glm::vec3(m_WorldTransform * glm::vec4(triangleOS.P2(), 1.f)));
		triangleWS->SetModel(triangleOS.GetModel());
		m_TrianglesWS.push_back(triangleWS);
	}	
}


void CSubModel::CreateTriangleData(uint nFaces, const ushort* pIndexData, 
	const glm::vec4* pPositionData, const glm::vec3* pNormalData)
{
	for(uint i = 0; i < nFaces; i++)
	{
		uint i1 = pIndexData[i * 3 + 0];
		uint i2 = pIndexData[i * 3 + 1];
		uint i3 = pIndexData[i * 3 + 2];
		glm::vec3 p1 = glm::vec3(pPositionData[i1]);
		glm::vec3 p2 = glm::vec3(pPositionData[i2]);
		glm::vec3 p3 = glm::vec3(pPositionData[i3]);
		
		glm::vec3 normal;
		if(pNormalData)
			normal = 1.f/3.f * (pNormalData[i1] + pNormalData[i2] + pNormalData[i3]);
		else
			normal = glm::normalize(glm::cross(p3-p1, p2-p1));

		CTriangle* triangle = new CTriangle(p1, p2, p3);
		triangle->SetModel(this);
		m_TrianglesOS.push_back(triangle);														 
	}
} 
