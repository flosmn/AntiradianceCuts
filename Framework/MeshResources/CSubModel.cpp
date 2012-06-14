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

	pMeshGeometry->CreateTriangleData(m_Triangles);

	return true;
}

bool CSubModel::Init(CMesh* pMesh) 
{
	V_RET_FOF(m_pGLVARenderData->Init(GL_TRIANGLES));

	uint nVertices = pMesh->GetNumberOfVertices();
	m_nTriangles = pMesh->GetNumberOfTriangles();
	m_pMaterial = new CMeshMaterial();

	GLuint positionDataSize = sizeof(glm::vec4) * nVertices;
	GLuint normalDataSize = sizeof(glm::vec3) * nVertices;
	GLuint indexDataSize = sizeof(GLshort) * 3 * m_nTriangles;
	
	V_RET_FOF(m_pGLVARenderData->AddVertexDataChannel(0, 4));
	V_RET_FOF(m_pGLVARenderData->AddVertexData(0, positionDataSize, (void*)pMesh->GetVertexPositions()));

	V_RET_FOF(m_pGLVARenderData->AddVertexDataChannel(1, 3));
	V_RET_FOF(m_pGLVARenderData->AddVertexData(1, normalDataSize, (void*)pMesh->GetVertexNormals()));

	V_RET_FOF(m_pGLVARenderData->AddIndexDataChannel());
	V_RET_FOF(m_pGLVARenderData->AddIndexData(indexDataSize, (void*)pMesh->GetIndexData()));
	
	m_pGLVARenderData->Finish();

	pMesh->CreateTriangleData(m_Triangles);

	return true;
}

void CSubModel::Release()
{
	m_pGLVARenderData->Release();
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

std::vector<Triangle*> CSubModel::GetTriangles() 
{
	return m_Triangles;
}

void CSubModel::SetMaterial(MATERIAL& mat)
{
	m_pMaterial->SetMaterialData(mat); 
}

MATERIAL& CSubModel::GetMaterial()
{
	return m_pMaterial->GetMaterialData(); 
}
	
