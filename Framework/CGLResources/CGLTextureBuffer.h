#ifndef _C_GL_TEXTURE_BUFFER_H_
#define _C_GL_TEXTURE_BUFFER_H_

#include "GL/glew.h"
#include "GL/gl.h"

#include "CGLResource.h"

class CGLTextureBuffer : public CGLResource
{
public:
	CGLTextureBuffer(std::string debugName);
	~CGLTextureBuffer();

	virtual bool Init();
	virtual void Release();
	
	void SetContent(GLuint elementSize, GLuint numElements, void* content, GLenum usage);

private:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();

	GLenum m_InternalFormat;
	GLuint m_TextureBufferTexture;
};

#endif // _C_GL_TEXTURE_BUFFER_H_