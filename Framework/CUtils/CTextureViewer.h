#ifndef _C_TEXTURE_VIEWER_H_
#define _C_TEXTURE_VIEWER_H_

#include "GL/glew.h"

#include "..\CProgram.h"

class CGLTexture2D;
class CFullScreenQuad;

class CTextureViewer : public CProgram {
public:
	CTextureViewer();
	~CTextureViewer();

	bool Init();
	void Release();

	void DrawTexture(CGLTexture2D* pTexture, GLuint x, GLuint y, 
		GLuint width, GLuint height);

private:
	CFullScreenQuad* m_pFullScreenQuad;
};

#endif // _C_TEXTURE_VIEWER_H_