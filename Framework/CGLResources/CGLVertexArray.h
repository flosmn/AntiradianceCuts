#ifndef _C_GL_VERTEX_ARRAY_H_
#define _C_GL_VERTEX_ARRAY_H_

#include "CGLResource.h"

#include <vector>

typedef unsigned int uint;

struct CHANNEL_INFO
{
	uint index;
	uint elements;
};

class CGLVertexBuffer;

class CGLVertexArray : public CGLResource
{
public:
	CGLVertexArray(std::string debugName);
	~CGLVertexArray();

	virtual bool Init(GLenum primitiveType);
	virtual void Release();

	bool AddVertexDataChannel(uint index, uint elements);
	bool AddVertexData(uint index, uint size, void* pData);
	
	bool AddIndexDataChannel();
	bool AddIndexData(GLuint size, void* pData);

	void Finish();

	void Draw(GLuint count);

private:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();

	CGLVertexBuffer* m_pGLVBIndexData;
	bool m_HasIndexData;

	CGLVertexBuffer* m_VertexDataChannels[10];
	CHANNEL_INFO m_VertexDataChannelInfo[10];
	uint m_ChannelsUsed[10];
	
	GLenum m_PrimitiveType;
};

#endif _C_GL_VERTEX_ARRAY_H_
