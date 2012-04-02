#include "CTextureViewer.h"

#include "..\Macros.h"

#include "..\CUtils\ShaderUtil.h"

#include "..\CGLResources\CGLProgram.h"
#include "..\CGLResources\CGLTexture2D.h"
#include "..\CGLResources\CGLBindLock.h"

#include "..\CMeshResources\CFullScreenQuad.h"

CTextureViewer::CTextureViewer()
	: CProgram("TextureViewer.m_pGLProgram", "Shaders\\DrawTexture.vert", "Shaders\\DrawTexture.frag")
{
	m_pFullScreenQuad = new CFullScreenQuad();
}

CTextureViewer::~CTextureViewer()
{
	CProgram::~CProgram();

	SAFE_DELETE(m_pFullScreenQuad);
}

bool CTextureViewer::Init()
{
	V_RET_FOF(CProgram::Init());

	V_RET_FOF(m_pFullScreenQuad->Init());

	glGenSamplers(1, &sampler);

	return true;
}

void CTextureViewer::Release()
{
	CProgram::Release();

	m_pFullScreenQuad->Release();
}

void CTextureViewer::DrawTexture(CGLTexture2D* pTexture, GLuint x, GLuint y, 
	GLuint width, GLuint height) 
{		
	CGLBindLock lockProgram(GetGLProgram(), CGL_PROGRAM_SLOT);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glViewport(x, y, width, height);
	
	CGLBindLock lockTexture(pTexture, CGL_TEXTURE0_SLOT);
	
	m_pFullScreenQuad->Draw();

}