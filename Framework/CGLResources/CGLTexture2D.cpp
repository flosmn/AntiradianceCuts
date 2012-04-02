#include "CGLTexture2D.h"

#include "..\Macros.h"

#include "..\CUtils\GLErrorUtil.h"


CGLTexture2D::CGLTexture2D(std::string debugName)
	: CGLResource(CGL_TEXTURE_2D, debugName)
{

}

CGLTexture2D::~CGLTexture2D()
{

}

bool CGLTexture2D::Init(GLuint width, GLuint height, GLenum internalFormat,
		GLenum format, GLenum type, GLuint nMipMaps, bool genMipMaps)
{
	V_RET_FOF(CGLResource::Init());

	m_Width = width;
	m_Height = height;
	m_nMipMaps = nMipMaps;
	m_InternalFormat = internalFormat;
	m_Format = format;
	m_Type = type;
	m_GenMipMaps = genMipMaps;
	
	glGenTextures(1, &m_Resource);

	V_RET_FOT(CheckGLError(m_DebugName, "CGLTexture2D::Init()"));
	
	{
		CGLBindLock lock(this, CGL_TEXTURE0_SLOT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, nMipMaps);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, m_GenMipMaps);
		glTexImage2D(GL_TEXTURE_2D, 0, m_InternalFormat, m_Width, m_Height,
			0, m_Format, m_Type, 0);
	}

	V_RET_FOT(CheckGLError(m_DebugName, "CGLTexture2D::Init()"));
	
	return true;
}

void CGLTexture2D::Release()
{
	CGLResource::Release();

	glDeleteTextures(1, &m_Resource);

	CheckGLError(m_DebugName, "CGLTexture2D::Release()");
}

void CGLTexture2D::GenerateMipMaps()
{
	if(m_GenMipMaps)
	{
		CGLBindLock lock(this, CGL_TEXTURE0_SLOT);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::string warning = "GenerateMipMaps() was called but automatic " 
			"MipMap generation was not activated at initialization.";
		PrintWarning(warning);
	}
}

void CGLTexture2D::Bind(CGLBindSlot slot)
{
	CGLResource::Bind(slot);
		
	glActiveTexture(GetGLSlot(m_Slot));
	glBindTexture(GL_TEXTURE_2D, m_Resource);
}

void CGLTexture2D::Unbind()
{
	CGLResource::Unbind();

	glActiveTexture(GetGLSlot(m_Slot));
	glBindTexture(GL_TEXTURE_2D, 0);
}