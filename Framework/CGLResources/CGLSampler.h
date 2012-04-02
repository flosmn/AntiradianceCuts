#ifndef _C_GL_SAMPLER_H_
#define _C_GL_SAMPLER_H_

#include "CGLResource.h"

class CGLSampler : public CGLResource
{
public:
	CGLSampler(std::string debugName);
	~CGLSampler();

	bool Init(GLenum minFilter, GLenum magFilter);
	void Release();

private:
	void Bind(CGLBindSlot slot) {}
	void Unbind() {}
};

#endif // _C_GL_SAMPLER_H_