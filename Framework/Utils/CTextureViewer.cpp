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
	m_program.reset(new CProgram("Shaders\\DrawTexture.vert", "Shaders\\DrawTexture.frag", "Program"));
	m_fullScreenQuad.reset(new CFullScreenQuad());
}

CTextureViewer::~CTextureViewer()
{
}

void CTextureViewer::DrawTexture(COGLTexture2D* pTexture, GLuint x, GLuint y, 
	GLuint width, GLuint height) 
{		
	COGLBindLock lockProgram(m_program->GetGLProgram(), COGL_PROGRAM_SLOT);
		
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glViewport(x, y, width, height);
	
	COGLBindLock lockTexture(pTexture, COGL_TEXTURE0_SLOT);
	
	m_fullScreenQuad->Draw();
}
