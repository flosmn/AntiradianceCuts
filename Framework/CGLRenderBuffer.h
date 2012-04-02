#ifndef _C_GL_RENDER_BUFFER_H_
#define _C_GL_RENDER_BUFFER_H_

#include "CGLResource.h"

class CGLRenderBuffer : public CGLResource
{
public:
	CGLRenderBuffer(std::string debugName);
	~CGLRenderBuffer();

	virtual bool Init(GLuint width, GLuint height, GLenum internalFormat);
	virtual void Release();

private:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();

	GLuint m_Width;
	GLuint m_Height;
	GLenum m_InternalFormat;
	GLenum m_AttachmentSlot;
};

#endif // _C_GL_RENDER_BUFFER_H_