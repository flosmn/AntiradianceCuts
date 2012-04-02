#include "CGLProgram.h"

#include "Macros.h"
#include "Defines.h"
#include "GLErrorUtil.h"

#include "CGLUniformBuffer.h"
#include "CGLSampler.h"

#include <istream>
#include <fstream>
#include <sstream>
#include <assert.h>

CGLProgram::CGLProgram(const std::string debugName, const std::string VS, const std::string FS)
	: CGLResource(CGL_PROGRAM, debugName), m_VS(VS), m_FS(FS)
{
	
}

CGLProgram::~CGLProgram()
{
	CGLResource::~CGLResource();
}

bool CGLProgram::Init()
{
	V_RET_FOF(CGLResource::Init());

	GLuint vertexShader = LoadShader(GL_VERTEX_SHADER, m_VS);
	GLuint fragmentShader = LoadShader(GL_FRAGMENT_SHADER, m_FS);
	m_Resource = glCreateProgram();

	glAttachShader(m_Resource, vertexShader);
	glAttachShader(m_Resource, fragmentShader);

	glLinkProgram(m_Resource);

	GLint status;
	glGetProgramiv (m_Resource, GL_LINK_STATUS, &status);
	if (status == GL_FALSE)
	{
		GLint infoLogLength;
		glGetProgramiv(m_Resource, GL_INFO_LOG_LENGTH, &infoLogLength);

		GLchar *strInfoLog = new GLchar[infoLogLength + 1];
		glGetProgramInfoLog(m_Resource, infoLogLength, NULL, strInfoLog);
		fprintf(stderr, "Linker failure: %s\n", strInfoLog);
		delete[] strInfoLog;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return true;
}
	
void CGLProgram::Release()
{
	CGLResource::Release();

	glDeleteProgram(m_Resource);
}

void CGLProgram::BindUniformBuffer(CGLUniformBuffer* pGLUniformBuffer, 
	const std::string strUniformBlockName)
{
	CheckInitialized("CGLProgram::BindUniformBuffer()");

	GLuint index = glGetUniformBlockIndex(m_Resource, strUniformBlockName.c_str());

	CheckGLError("CGLProgram", "BindUniformBuffer()");

	glUniformBlockBinding(m_Resource, index, pGLUniformBuffer->GetGlobalBindingPoint());

	CheckGLError("CGLProgram", "BindUniformBuffer()");

	glBindBufferBase(GL_UNIFORM_BUFFER, pGLUniformBuffer->GetGlobalBindingPoint(), pGLUniformBuffer->GetResourceIdentifier());
}

void CGLProgram::BindSampler(GLuint samplerSlot, CGLSampler* pGLSampler)
{
	CheckInitialized("CGLProgram::BindSampler()");

	CGLBindLock lockProgram(this, CGL_PROGRAM_SLOT);

	glBindSampler(samplerSlot, pGLSampler->GetResourceIdentifier());
}

void CGLProgram::Bind(CGLBindSlot slot)
{
	CGLResource::Bind(slot);

	assert(m_Slot == CGL_PROGRAM_SLOT);

	glUseProgram(m_Resource);
}
	
void CGLProgram::Unbind()
{
	CGLResource::Unbind();

	assert(m_Slot == CGL_PROGRAM_SLOT);

	glUseProgram(0);
}

GLuint CGLProgram::LoadShader(GLenum eShaderType, const std::string &strShaderFilename)
{
	std::string strFilename = FindFileOrThrow(strShaderFilename);
	std::ifstream shaderFile(strFilename.c_str());
	std::stringstream shaderData;
	shaderData << shaderFile.rdbuf();
	shaderFile.close();

	return CreateShader(eShaderType, shaderData.str());
}

std::string CGLProgram::FindFileOrThrow( const std::string &strBasename )
{
	std::string strFilename = LOCAL_DIR + strBasename;
	std::ifstream testFile(strFilename.c_str());
	if(testFile.is_open())
		return strFilename;

	strFilename = LOCAL_FILE_DIR + strBasename;
	testFile.open(strFilename.c_str());
	if(testFile.is_open())
		return strFilename;

	throw std::runtime_error("Could not find the file " + strBasename);
}

GLuint CGLProgram::CreateShader(GLenum eShaderType, const std::string &strShaderFile)
{
	GLuint shader = glCreateShader(eShaderType);
	const char *strFileData = strShaderFile.c_str();
	glShaderSource(shader, 1, &strFileData, NULL);

	glCompileShader(shader);

	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE)
	{
		GLint infoLogLength;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

		GLchar *strInfoLog = new GLchar[infoLogLength + 1];
		glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

		const char *strShaderType = NULL;
		switch(eShaderType)
		{
		case GL_VERTEX_SHADER: strShaderType = "vertex"; break;
		case GL_GEOMETRY_SHADER: strShaderType = "geometry"; break;
		case GL_FRAGMENT_SHADER: strShaderType = "fragment"; break;
		}

		fprintf(stderr, "Compile failure in %s shader:\n%s\n", strShaderType, 
			strInfoLog);

		delete[] strInfoLog;
	}

	return shader;
}