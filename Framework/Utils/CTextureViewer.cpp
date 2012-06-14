#include "CTextureViewer.h"

#include "..\Macros.h"

#include "..\Utils\ShaderUtil.h"

#include "..\OGLResources\COGLProgram.h"
#include "..\OGLResources\COGLTexture2D.h"
#include "..\OGLResources\COGLBindLock.h"
#include "..\OGLResources\COGLSampler.h"

#include "..\MeshResources\CFullScreenQuad.h"

CTextureViewer::CTextureViewer()
{
	m_pProgram = new CProgram("TextureViewer.m_pGLProgram", "Shaders\\DrawTexture.vert", "Shaders\\DrawTexture.frag");
	m_pSampler = new COGLSampler("CTextureViewer.m_pSampler");
	m_pFullScreenQuad = new CFullScreenQuad();
}

CTextureViewer::~CTextureViewer()
{
	SAFE_DELETE(m_pProgram);
	SAFE_DELETE(m_pSampler);
	SAFE_DELETE(m_pFullScreenQuad);
}

bool CTextureViewer::Init()
{
	V_RET_FOF(m_pProgram->Init());
	V_RET_FOF(m_pFullScreenQuad->Init());
	
	V_RET_FOF(m_pSampler->Init(GL_LINEAR, GL_LINEAR, GL_REPEAT, GL_REPEAT));	
	
	return true;
}

void CTextureViewer::Release()
{
	m_pProgram->Release();
	m_pSampler->Release();
	m_pFullScreenQuad->Release();
}

void CTextureViewer::DrawTexture(COGLTexture2D* pTexture, GLuint x, GLuint y, 
	GLuint width, GLuint height) 
{		
	COGLBindLock lockProgram(m_pProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
		
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glViewport(x, y, width, height);
	
	COGLBindLock lockTexture(pTexture, COGL_TEXTURE0_SLOT);
	
	m_pFullScreenQuad->Draw();

}