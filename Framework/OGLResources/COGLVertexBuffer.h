#ifndef _C_GL_VERTEX_BUFFER_H_
#define _C_GL_VERTEX_BUFFER_H_

#include "COGLResource.h"

class COGLVertexBuffer : public COGLResource
{
public:
	COGLVertexBuffer(std::string debugName);
	~COGLVertexBuffer();

	virtual bool Init();
	virtual void Release();

	bool SetContent(GLuint size, void* data, GLenum usage);

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();

};

#endif // _C_GL_VERTEX_BUFFER_H_