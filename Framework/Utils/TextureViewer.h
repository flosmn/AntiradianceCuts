#ifndef TEXTUREVIEWER_H_
#define TEXTUREVIEWER_H_

#include "GL/glew.h"

#include "../CProgram.h"

#include "../FullScreenQuad.h"

#include "../OGLResources/COGLTexture2D.h"
#include "../OGLResources/COGLSampler.h"

class TextureViewer
{
public:
	TextureViewer() 
	{
		m_program.reset(new CProgram("Shaders\\DrawTexture.vert", "Shaders\\DrawTexture.frag", "Program"));
		m_fullScreenQuad.reset(new FullScreenQuad());
	}
	
	void drawTexture(COGLTexture2D* pTexture, GLuint x, GLuint y, GLuint width, GLuint height) 
	{
		COGLBindLock lockProgram(m_program->GetGLProgram(), COGL_PROGRAM_SLOT);
		
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		glViewport(x, y, width, height);
		
		COGLBindLock lockTexture(pTexture, COGL_TEXTURE0_SLOT);
		
		m_fullScreenQuad->draw();
	}

private:
	std::unique_ptr<CProgram> m_program;
	std::unique_ptr<FullScreenQuad> m_fullScreenQuad;
};

#endif // TEXTUREVIEWER_H_
