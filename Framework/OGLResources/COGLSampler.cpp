#include "COGLSampler.h"

#include "..\Macros.h"

COGLSampler::COGLSampler(GLenum minFilter, GLenum magFilter, GLenum clampS, 
	GLenum clampT, std::string const& debugName)
	:COGLResource(COGL_SAMPLER, debugName)
{
	glGenSamplers(1, &m_Resource);

	glSamplerParameteri(m_Resource, GL_TEXTURE_MIN_FILTER, minFilter);
	glSamplerParameteri(m_Resource, GL_TEXTURE_MAG_FILTER, magFilter);
	glSamplerParameteri(m_Resource, GL_TEXTURE_WRAP_S, clampS);
	glSamplerParameteri(m_Resource, GL_TEXTURE_WRAP_T, clampT);
}

COGLSampler::~COGLSampler()
{
	glDeleteSamplers(1, &m_Resource);
}