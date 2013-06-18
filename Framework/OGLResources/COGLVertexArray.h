#ifndef _C_GL_VERTEX_ARRAY_H_
#define _C_GL_VERTEX_ARRAY_H_

#include "COGLResource.h"

#include <vector>

typedef unsigned int uint;

struct CHANNEL_INFO
{
	uint index;
	uint elements;
};

class COGLVertexBuffer;

class COGLVertexArray : public COGLResource
{
public:
	COGLVertexArray(GLenum primitiveType, std::string const& debugName = "");
	~COGLVertexArray();

	bool AddVertexData(uint index, uint elements, uint size, void* pData);
	bool AddIndexData(GLuint size, void* pData);

	void Finish();

	void Draw(GLuint count);

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();
		
	bool m_HasIndexData;
	
	std::unique_ptr<COGLVertexBuffer> m_IndexData;
	std::vector<std::unique_ptr<COGLVertexBuffer>> m_VertexDataChannels;
	std::vector<CHANNEL_INFO> m_VertexDataChannelInfo;
	
	GLenum m_PrimitiveType;
};

#endif _C_GL_VERTEX_ARRAY_H_
