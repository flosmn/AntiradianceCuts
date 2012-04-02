#ifndef _C_GL_VERTEX_ARRAY_H_
#define _C_GL_VERTEX_ARRAY_H_

#include "CGLResource.h"

class CGLVertexBuffer;

class CGLVertexArray : public CGLResource
{
public:
	CGLVertexArray(std::string debugName);
	~CGLVertexArray();

	virtual bool Init();
	virtual void Release();

	bool AddPositionData(GLuint size, void* pData);
	bool AddNormalData(GLuint size, void* pData);
	bool AddTextureData(GLuint size, void* pData);
	bool AddIndexData(GLuint size, void* pData);

	void Finish();

	void Draw(GLuint count);

private:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();

	CGLVertexBuffer* m_pGLVBPositionData;
	CGLVertexBuffer* m_pGLVBNormalData;
	CGLVertexBuffer* m_pGLVBTextureData;
	CGLVertexBuffer* m_pGLVBIndexData;

	bool m_HasPositionData;
	bool m_HasNormalData;
	bool m_HasTextureData;
	bool m_HasIndexData;
};

#endif _C_GL_VERTEX_ARRAY_H_