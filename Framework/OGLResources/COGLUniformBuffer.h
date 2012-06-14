#ifndef _C_GL_UNIFORM_BUFFER_H_
#define _C_GL_UNIFORM_BUFFER_H_

typedef unsigned int uint;

#include "COGLResource.h"

class COGLUniformBuffer : public COGLResource
{
public:
	COGLUniformBuffer(std::string debugName);
	~COGLUniformBuffer();

	virtual bool Init(uint size, void* data, GLenum usage);
	virtual void Release();

	void UpdateData(void* data);
	uint GetGlobalBindingPoint();

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();

	uint GetUniqueGlobalBindingPoint();

	uint m_Size;
	uint m_GlobalBindingPoint;

	static uint static_GlobalBindingPoint;
};

#endif // _C_GL_UNIFORM_BUFFER_H_