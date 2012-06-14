#include "COGLSampler.h"

#include "..\Macros.h"

COGLSampler::COGLSampler(std::string debugName)
	:COGLResource(COGL_SAMPLER, debugName)
{

}

COGLSampler::~COGLSampler()
{
	COGLResource::~COGLResource();
}

bool COGLSampler::Init(GLenum minFilter, GLenum magFilter, GLenum clampS, GLenum clampT)
{
	V_RET_FOF(COGLResource::Init());

	glGenSamplers(1, &m_Resource);

	glSamplerParameteri(m_Resource, GL_TEXTURE_MIN_FILTER, minFilter);
	glSamplerParameteri(m_Resource, GL_TEXTURE_MAG_FILTER, magFilter);
	glSamplerParameteri(m_Resource, GL_TEXTURE_WRAP_S, clampS);
	glSamplerParameteri(m_Resource, GL_TEXTURE_WRAP_T, clampT);

	return true;
}

void COGLSampler::Release()
{
	COGLResource::Release();

	glDeleteSamplers(1, &m_Resource);
}