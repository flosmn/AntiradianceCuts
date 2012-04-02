#ifndef SHADERUTIL_H
#define SHADERUTIL_H

#include "GL/glew.h"

#include "..\Defines.h"

#include <iostream>
#include <istream>
#include <vector>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>
#include <stdio.h>

static std::string FindFileOrThrow( const std::string &strBasename )
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

static GLuint CreateShader(GLenum eShaderType, const std::string &strShaderFile)
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

static GLuint LoadShader(GLenum eShaderType, const std::string &strShaderFilename)
{
		std::string strFilename = FindFileOrThrow(strShaderFilename);
		std::ifstream shaderFile(strFilename.c_str());
		std::stringstream shaderData;
		shaderData << shaderFile.rdbuf();
		shaderFile.close();
		
		return CreateShader(eShaderType, shaderData.str());
}

static GLuint CreateProgram(const std::string &strVertexShaderFilename, 
							const std::string &strFragmentShaderFilename)
{
		GLuint vertexShader = LoadShader(GL_VERTEX_SHADER, strVertexShaderFilename);
		GLuint fragmentShader = LoadShader(GL_FRAGMENT_SHADER, strFragmentShaderFilename);
		GLuint program = glCreateProgram();
		
		glAttachShader(program, vertexShader);
		glAttachShader(program, fragmentShader);
		
		glLinkProgram(program);
		
		GLint status;
		glGetProgramiv (program, GL_LINK_STATUS, &status);
		if (status == GL_FALSE)
		{
				GLint infoLogLength;
				glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
				
				GLchar *strInfoLog = new GLchar[infoLogLength + 1];
				glGetProgramInfoLog(program, infoLogLength, NULL, strInfoLog);
				fprintf(stderr, "Linker failure: %s\n", strInfoLog);
				delete[] strInfoLog;
		}
		
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		return program;
}

#endif