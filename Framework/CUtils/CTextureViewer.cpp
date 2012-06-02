#include "CTextureViewer.h"

#include "..\Macros.h"

#include "..\CUtils\ShaderUtil.h"

#include "..\CGLResources\CGLProgram.h"
#include "..\CGLResources\CGLTexture2D.h"
#include "..\CGLResources\CGLBindLock.h"

#include "..\CMeshResources\CFullScreenQuad.h"

CTextureViewer::CTextureViewer()
{
	m_pProgram = new CProgram("TextureViewer.m_pGLProgram", "Shaders\\DrawTexture.vert", "Shaders\\DrawTexture.frag");
	m_pFullScreenQuad = new CFullScreenQuad();
}

CTextureViewer::~CTextureViewer()
{
	SAFE_DELETE(m_pProgram);
	SAFE_DELETE(m_pFullScreenQuad);
}

bool CTextureViewer::Init()
{
	V_RET_FOF(m_pProgram->Init());
	V_RET_FOF(m_pFullScreenQuad->Init());
	
	return true;
}

void CTextureViewer::Release()
{
	m_pProgram->Release();
	m_pFullScreenQuad->Release();
}

void CTextureViewer::DrawTexture(CGLTexture2D* pTexture, GLuint x, GLuint y, 
	GLuint width, GLuint height) 
{		
	CGLBindLock lockProgram(m_pProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glViewport(x, y, width, height);
	
	CGLBindLock lockTexture(pTexture, CGL_TEXTURE0_SLOT);
	
	m_pFullScreenQuad->Draw();

}