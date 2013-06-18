#ifndef _C_GL_VERTEX_BUFFER_H_
#define _C_GL_VERTEX_BUFFER_H_

#include "COGLResource.h"

class COGLVertexBuffer : public COGLResource
{
public:
	COGLVertexBuffer(std::string const& debugName = "");
	~COGLVertexBuffer();
	
	bool SetContent(GLuint size, void* data, GLenum usage);

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();

};

#endif // _C_GL_VERTEX_BUFFER_H_