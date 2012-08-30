#include "COGLProgram.h"

#include "..\Macros.h"
#include "..\Defines.h"

#include "..\Utils\GLErrorUtil.h"

#include "..\OGLResources\COGLUniformBuffer.h"
#include "..\OGLResources\COGLSampler.h"

#include <istream>
#include <fstream>
#include <sstream>
#include <assert.h>

COGLProgram::COGLProgram(const std::string debugName, const std::string VS, const std::string FS, std::vector<std::string> headerFiles)
	: COGLResource(COGL_PROGRAM, debugName), m_VS(VS), m_FS(FS), m_HeaderFiles(headerFiles)
{
	
}

COGLProgram::~COGLProgram()
{
	COGLResource::~COGLResource();
}

bool COGLProgram::Init()
{
	V_RET_FOF(COGLResource::Init());
	/*
	for(int i = 0; i < m_HeaderFiles.size(); ++i)
	{
		std::stringstream ss;
		ss << "/" << m_HeaderFiles[i]; 
		std::string name(ss.str());
		
		m_HeaderFileNames.push_back(name);
		std::string content = GetFileContent(m_HeaderFiles[i]);

		std::cout << "header file name: " << name.c_str() << std::endl;
		std::cout << "header file content: " << content << std::endl;

		glNamedStringARB(GL_SHADER_INCLUDE_ARB, 
			-1, name.c_str(),
			-1, content.c_str());

		CheckGLError("CGLProgram", "glNamedStringARB");
	}
	*/

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
	
void COGLProgram::Release()
{
	COGLResource::Release();

	glDeleteProgram(m_Resource);
}

void COGLProgram::BindUniformBuffer(COGLUniformBuffer* pGLUniformBuffer, 
	const std::string strUniformBlockName)
{
	CheckInitialized("COGLProgram::BindUniformBuffer()");

	GLuint index = glGetUniformBlockIndex(m_Resource, strUniformBlockName.c_str());

	CheckGLError("COGLProgram", "BindUniformBuffer()");

	glUniformBlockBinding(m_Resource, index, pGLUniformBuffer->GetGlobalBindingPoint());

	CheckGLError("COGLProgram", "BindUniformBuffer()");

	glBindBufferBase(GL_UNIFORM_BUFFER, pGLUniformBuffer->GetGlobalBindingPoint(), pGLUniformBuffer->GetResourceIdentifier());
}

void COGLProgram::BindSampler(GLuint samplerSlot, COGLSampler* pGLSampler)
{
	CheckInitialized("COGLProgram::BindSampler()");

	COGLBindLock lockProgram(this, COGL_PROGRAM_SLOT);

	glBindSampler(samplerSlot, pGLSampler->GetResourceIdentifier());
}

void COGLProgram::Bind(COGLBindSlot slot)
{
	COGLResource::Bind(slot);

	assert(m_Slot == COGL_PROGRAM_SLOT);

	glUseProgram(m_Resource);
}
	
void COGLProgram::Unbind()
{
	COGLResource::Unbind();

	assert(m_Slot == COGL_PROGRAM_SLOT);

	glUseProgram(0);
}

GLuint COGLProgram::LoadShader(GLenum eShaderType, const std::string &strShaderFilename)
{
	std::string shaderData = GetFileContent(strShaderFilename);
	return CreateShader(eShaderType, shaderData, strShaderFilename);
}

std::string COGLProgram::FindFileOrThrow( const std::string &strBasename )
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

std::string COGLProgram::GetFileContent( const std::string &strShaderFile)
{
	std::string strFilename = FindFileOrThrow(strShaderFile);
	std::ifstream shaderFile(strFilename.c_str());
	std::stringstream shaderData;
	shaderData << shaderFile.rdbuf();
	shaderFile.close();
	
	return shaderData.str();
}

GLuint COGLProgram::CreateShader(GLenum eShaderType, const std::string &strShaderFile, const std::string& strFileName)
{
	GLuint shader = glCreateShader(eShaderType);
	const char *strFileData = strShaderFile.c_str();
	glShaderSource(shader, 1, &strFileData, NULL);

	GLsizei count = (GLsizei)m_HeaderFileNames.size();
	const char** names = new const char*[count];
	for(int i = 0; i < count; ++i)
	{
		names[i] = m_HeaderFileNames[i].c_str();
	}

	if(count == 0)
		glCompileShader(shader);
	else
		glCompileShaderIncludeARB(shader, count, names, 0);

	CheckGLError("COGLProgram", "glCompileShader");

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

		fprintf(stderr, "Compile failure in %s shader:\n%s\n%s\n", strShaderType, strFileName.c_str(),
			strInfoLog);

		delete[] strInfoLog;
	}

	return shader;
}