#ifndef _C_GL_TEXTURE_BUFFER_H_
#define _C_GL_TEXTURE_BUFFER_H_

#include "GL/glew.h"
#include "GL/gl.h"

#include "COGLResource.h"

class COGLTextureBuffer : public COGLResource
{
public:
	COGLTextureBuffer(GLenum type, std::string const& debugName = "");
	~COGLTextureBuffer();

	void SetContent(size_t size, GLenum usage, void* content);

	size_t GetSize() { return m_size; }

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();

	size_t m_size;
	GLenum m_Type;
	GLuint m_TextureBufferTexture;
};

#endif // _C_GL_TEXTURE_BUFFER_H_
