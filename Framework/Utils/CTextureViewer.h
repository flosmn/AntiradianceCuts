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

	bool Init();
	void Release();

	void DrawTexture(COGLTexture2D* pTexture, GLuint x, GLuint y, 
		GLuint width, GLuint height);

private:
	CProgram* m_pProgram;
	COGLSampler* m_pSampler;
	CFullScreenQuad* m_pFullScreenQuad;
};

#endif // _C_TEXTURE_VIEWER_H_