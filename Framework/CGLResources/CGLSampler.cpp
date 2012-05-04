#include "CGLSampler.h"

#include "..\Macros.h"

CGLSampler::CGLSampler(std::string debugName)
	:CGLResource(CGL_SAMPLER, debugName)
{

}

CGLSampler::~CGLSampler()
{
	CGLResource::~CGLResource();
}

bool CGLSampler::Init(GLenum minFilter, GLenum magFilter, GLenum clampS, GLenum clampT)
{
	V_RET_FOF(CGLResource::Init());

	glGenSamplers(1, &m_Resource);

	glSamplerParameteri(m_Resource, GL_TEXTURE_MIN_FILTER, minFilter);
	glSamplerParameteri(m_Resource, GL_TEXTURE_MAG_FILTER, magFilter);
	glSamplerParameteri(m_Resource, GL_TEXTURE_WRAP_S, clampS);
	glSamplerParameteri(m_Resource, GL_TEXTURE_WRAP_T, clampT);

	return true;
}

void CGLSampler::Release()
{
	CGLResource::Release();

	glDeleteSamplers(1, &m_Resource);
}