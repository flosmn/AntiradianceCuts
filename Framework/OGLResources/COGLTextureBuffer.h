#ifndef _C_GL_TEXTURE_BUFFER_H_
#define _C_GL_TEXTURE_BUFFER_H_

#include "GL/glew.h"
#include "GL/gl.h"

#include "COGLResource.h"

class COGLTextureBuffer : public COGLResource
{
public:
	COGLTextureBuffer(std::string debugName);
	~COGLTextureBuffer();

	virtual bool Init();
	virtual void Release();
	
	void SetContent(GLuint elementSize, GLuint numElements, void* content, GLenum usage);

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();

	GLenum m_InternalFormat;
	GLuint m_TextureBufferTexture;
};

#endif // _C_GL_TEXTURE_BUFFER_H_