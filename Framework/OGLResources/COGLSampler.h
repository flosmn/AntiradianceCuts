#ifndef _C_GL_SAMPLER_H_
#define _C_GL_SAMPLER_H_

#include "COGLResource.h"

class COGLSampler : public COGLResource
{
public:
	COGLSampler(GLenum minFilter, GLenum magFilter, GLenum clampS, GLenum clampT, 
		std::string const& debugName = "");
	~COGLSampler();

private:
	void Bind(COGLBindSlot slot) {}
	void Unbind() {}
};

#endif // _C_GL_SAMPLER_H_