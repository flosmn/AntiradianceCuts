#ifndef _C_GL_SAMPLER_H_
#define _C_GL_SAMPLER_H_

#include "COGLResource.h"

class COGLSampler : public COGLResource
{
public:
	COGLSampler(std::string debugName);
	~COGLSampler();

	bool Init(GLenum minFilter, GLenum magFilter, GLenum clampS, GLenum clampT);
	void Release();

private:
	void Bind(COGLBindSlot slot) {}
	void Unbind() {}
};

#endif // _C_GL_SAMPLER_H_