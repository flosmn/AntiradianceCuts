#ifndef _C_TEXTURE_VIEWER_H_
#define _C_TEXTURE_VIEWER_H_

#include "GL/glew.h"

#include "..\CProgram.h"

class COGLTexture2D;
class COGLSampler;
class CFullScreenQuad;

class CTextureViewer {
public:
	CTextureViewer();
	~CTextureViewer();

	void DrawTexture(COGLTexture2D* pTexture, GLuint x, GLuint y, 
		GLuint width, GLuint height);

private:
	std::unique_ptr<CProgram> m_program;
	std::unique_ptr<CFullScreenQuad> m_fullScreenQuad;
};

#endif // _C_TEXTURE_VIEWER_H_
