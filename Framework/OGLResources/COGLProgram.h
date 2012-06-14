#ifndef _C_GL_PROGRAM_H_
#define _C_GL_PROGRAM_H_

#include "COGLResource.h"

class COGLUniformBuffer;
class COGLSampler;

class COGLProgram : public COGLResource
{
public:
	COGLProgram(const std::string debugName, const std::string strVS, const std::string strFS);
	~COGLProgram();

	virtual bool Init();
	virtual void Release();

	void BindUniformBuffer(COGLUniformBuffer* pGLUniformBuffer, const std::string strUniformBlockName);
	void BindSampler(GLuint samplerSlot, COGLSampler* pGLSampler);

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();

	GLuint LoadShader(GLenum eShaderType, const std::string &strShaderFilename);
	GLuint CreateShader(GLenum eShaderType, const std::string &strShaderFile, const std::string &strFileName);
	std::string FindFileOrThrow(const std::string &strBasename);

	std::string m_VS;	// vertex shader string
	std::string m_FS;	// fragment shader string
};

#endif _C_GL_PROGRAM_H_