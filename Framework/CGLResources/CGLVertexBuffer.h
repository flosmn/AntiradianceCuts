#ifndef _C_GL_VERTEX_BUFFER_H_
#define _C_GL_VERTEX_BUFFER_H_

#include "CGLResource.h"

class CGLVertexBuffer : public CGLResource
{
public:
	CGLVertexBuffer(std::string debugName);
	~CGLVertexBuffer();

	virtual bool Init();
	virtual void Release();

	bool SetContent(GLuint size, void* data, GLenum usage);

private:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();

};

#endif // _C_GL_VERTEX_BUFFER_H_