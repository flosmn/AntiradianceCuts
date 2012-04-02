#ifndef _C_GL_UNIFORM_BUFFER_H_
#define _C_GL_UNIFORM_BUFFER_H_

typedef unsigned int uint;

#include "CGLResource.h"

class CGLUniformBuffer : public CGLResource
{
public:
	CGLUniformBuffer(std::string debugName);
	~CGLUniformBuffer();

	virtual bool Init(uint size, void* data, GLenum usage);
	virtual void Release();

	void UpdateData(void* data);
	uint GetGlobalBindingPoint();

private:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();

	uint GetUniqueGlobalBindingPoint();

	uint m_Size;
	uint m_GlobalBindingPoint;

	static uint static_GlobalBindingPoint;
};

#endif // _C_GL_UNIFORM_BUFFER_H_