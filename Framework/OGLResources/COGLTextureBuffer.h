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

	virtual bool Init(size_t size, GLenum usage, GLenum type);
	virtual void Release();
	
	void SetContent(void* content, size_t size);

	size_t GetSize();

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();

	GLenum m_Usage;
	GLenum m_Type;
	GLenum m_InternalFormat;
	GLuint m_TextureBufferTexture;

	size_t m_Size;
};

#endif // _C_GL_TEXTURE_BUFFER_H_