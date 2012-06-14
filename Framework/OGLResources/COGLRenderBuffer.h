#ifndef _C_GL_RENDER_BUFFER_H_
#define _C_GL_RENDER_BUFFER_H_

#include "COGLResource.h"

class COGLRenderBuffer : public COGLResource
{
public:
	COGLRenderBuffer(std::string debugName);
	~COGLRenderBuffer();

	virtual bool Init(GLuint width, GLuint height, GLenum internalFormat);
	virtual void Release();

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();

	GLuint m_Width;
	GLuint m_Height;
	GLenum m_InternalFormat;
	GLenum m_AttachmentSlot;
};

#endif // _C_GL_RENDER_BUFFER_H_