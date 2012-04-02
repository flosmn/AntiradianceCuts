#include "CFullScreenQuad.h"

#include "..\Macros.h"

#include "CMesh.h"

#include "..\CGLResources\CGLVertexArray.h"


CFullScreenQuad::CFullScreenQuad()
{
	m_pFullScreenQuadMesh = new CFullScreenQuadMesh();
	m_pGLVARenderData = new CGLVertexArray("CFullScreenQuad.m_pVARenderData");

}

CFullScreenQuad::~CFullScreenQuad()
{
	SAFE_DELETE(m_pFullScreenQuadMesh);
	SAFE_DELETE(m_pGLVARenderData);
}

bool CFullScreenQuad::Init()
{
	V_RET_FOF(m_pGLVARenderData->Init());

	GLuint positionDataSize = sizeof(glm::vec4) * m_pFullScreenQuadMesh->GetNumberOfVertices();
	GLuint textureDataSize = sizeof(glm::vec3) * m_pFullScreenQuadMesh->GetNumberOfVertices();
	GLuint indexDataSize = sizeof(GLshort) * 3 * m_pFullScreenQuadMesh->GetNumberOfTriangles();

	V_RET_FOF(m_pGLVARenderData->AddPositionData(positionDataSize,
		(void*)m_pFullScreenQuadMesh->GetVertexPositions()));

	V_RET_FOF(m_pGLVARenderData->AddTextureData(textureDataSize,
		(void*)m_pFullScreenQuadMesh->GetVertexTexCoords()));

	V_RET_FOF(m_pGLVARenderData->AddIndexData(indexDataSize,
		(void*)m_pFullScreenQuadMesh->GetIndexData()));

	m_pGLVARenderData->Finish();

	return true;
}

void CFullScreenQuad::Release()
{
	m_pGLVARenderData->Release();
}

void CFullScreenQuad::Draw()
{
	m_pGLVARenderData->Draw(3 * m_pFullScreenQuadMesh->GetNumberOfTriangles());
}