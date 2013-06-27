#ifndef _C_GL_VERTEX_ARRAY_H_
#define _C_GL_VERTEX_ARRAY_H_

#include "COGLResource.h"

#include <vector>
#include <memory>

typedef unsigned int uint;

struct CHANNEL_INFO
{
	uint index;
	uint elements;
	uint divisor;
};

class COGLVertexBuffer;

class COGLVertexArray : public COGLResource
{
public:
	COGLVertexArray(GLenum primitiveType, std::string const& debugName = "");
	COGLVertexArray(GLenum primitiveType, GLuint instanceCount, std::string const& debugName = "");
	~COGLVertexArray();

	bool AddVertexData(uint index, GLuint elements, GLuint size, void* pData, GLuint divisor = 0);
	bool AddIndexData(GLuint size, void* pData);

	void Finish();

	void Draw(GLuint count);

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();
		
	bool m_hasIndexData;
	
	std::unique_ptr<COGLVertexBuffer> m_IndexData;
	std::vector<std::unique_ptr<COGLVertexBuffer>> m_VertexDataChannels;
	std::vector<CHANNEL_INFO> m_VertexDataChannelInfo;
	
	GLenum m_primitiveType;
	GLuint m_instanceCount;
};

#endif _C_GL_VERTEX_ARRAY_H_
