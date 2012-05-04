#ifndef _C_GL_PROGRAM_H_
#define _C_GL_PROGRAM_H_

#include "CGLResource.h"

class CGLUniformBuffer;
class CGLSampler;

class CGLProgram : public CGLResource
{
public:
	CGLProgram(const std::string debugName, const std::string strVS, const std::string strFS);
	~CGLProgram();

	virtual bool Init();
	virtual void Release();

	void BindUniformBuffer(CGLUniformBuffer* pGLUniformBuffer, const std::string strUniformBlockName);
	void BindSampler(GLuint samplerSlot, CGLSampler* pGLSampler);

private:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();

	GLuint LoadShader(GLenum eShaderType, const std::string &strShaderFilename);
	GLuint CreateShader(GLenum eShaderType, const std::string &strShaderFile, const std::string &strFileName);
	std::string FindFileOrThrow(const std::string &strBasename);

	std::string m_VS;	// vertex shader string
	std::string m_FS;	// fragment shader string
};

#endif _C_GL_PROGRAM_H_