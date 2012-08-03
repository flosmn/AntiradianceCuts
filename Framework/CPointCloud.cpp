#include "CPointCloud.h"

#include "Macros.h"

#include "OGLResources\COGLVertexArray.h"
#include "OGLResources\COGLProgram.h"

CPointCloud::CPointCloud()
{
	m_pVertexArray = new COGLVertexArray("CPointCloud.m_pVertexArray");
}

CPointCloud::~CPointCloud()
{
	SAFE_DELETE(m_pVertexArray);
}

bool CPointCloud::Init()
{
	V_RET_FOF(m_pVertexArray->Init(GL_POINTS));
	
	m_pVertexArray->AddVertexDataChannel(0, 4);
	m_pVertexArray->AddVertexDataChannel(1, 4);

	m_pVertexArray->AddIndexDataChannel();

	return true;
}

void CPointCloud::Release()
{
	m_pVertexArray->Release();
}

void CPointCloud::Draw(glm::vec4* positionData, glm::vec4* colorData, uint nPoints)
{
	m_pVertexArray->AddVertexData(0, nPoints * sizeof(glm::vec4), positionData);
	m_pVertexArray->AddVertexData(1, nPoints * sizeof(glm::vec4), colorData);

	GLuint* indexData = new GLuint[nPoints];
	for(uint i = 0; i < nPoints; ++i) indexData[i] = GLuint(i);

	m_pVertexArray->AddIndexData(nPoints * sizeof(GLuint), indexData);

	m_pVertexArray->Finish();

	glPointSize(3.0f);
			
	m_pVertexArray->Draw(nPoints);	

	delete [] indexData;
}