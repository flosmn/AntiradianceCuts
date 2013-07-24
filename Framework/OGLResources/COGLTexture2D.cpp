#include "COGLTexture2D.h"

#include <assert.h>

COGLTexture2D::COGLTexture2D(GLuint width, GLuint height, GLenum internalFormat,
		GLenum format, GLenum type, GLuint nMipMaps, bool genMipMaps, std::string const& debugName)
	: COGLResource(COGL_TEXTURE_2D, debugName)
{
	m_Width = width;
	m_Height = height;
	m_nMipMaps = nMipMaps;
	m_InternalFormat = internalFormat;
	m_Format = format;
	m_Type = type;
	m_GenMipMaps = genMipMaps;
	
	glGenTextures(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLTexture2D::COGLTexture2D()");
	
	{
		COGLBindLock lock(this, COGL_TEXTURE0_SLOT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, nMipMaps);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
		glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, m_GenMipMaps);
		glTexImage2D(GL_TEXTURE_2D, 0, m_InternalFormat, m_Width, m_Height,
			0, m_Format, m_Type, NULL);
	}

	CheckGLError(m_DebugName, "COGLTexture2D::COGLTexture2D()");
}

COGLTexture2D::~COGLTexture2D()
{
	glDeleteTextures(1, &m_Resource);

	CheckGLError(m_DebugName, "COGLTexture2D::~COGLTexture2D()");
}

void COGLTexture2D::GenerateMipMaps()
{
	if(m_GenMipMaps)
	{
		COGLBindLock lock(this, COGL_TEXTURE0_SLOT);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::string warning = "GenerateMipMaps() was called but automatic " 
			"MipMap generation was not activated at initialization.";
		PrintWarning(warning);
	}
}

void COGLTexture2D::GetPixelData(void* pData)
{
	COGLBindLock lock(this, COGL_TEXTURE0_SLOT);

	glGetTexImage(GL_TEXTURE_2D, 0, m_Format, m_Type, pData);

	CheckGLError(m_DebugName, "COGLTexture2D::GetPixelData()");
}

void COGLTexture2D::SetPixelData(void* pData)
{
	COGLBindLock lock(this, COGL_TEXTURE0_SLOT);

	CheckGLError(m_DebugName, "COGLTexture2D::SetPixelData(), before glTexImage2D");

	glTexImage2D(GL_TEXTURE_2D, 0, m_InternalFormat, m_Width, m_Height, 0, m_Format, m_Type, pData);
	
	CheckGLError(m_DebugName, "COGLTexture2D::SetPixelData(), after glTexImage2D");
}

void COGLTexture2D::CopyData(COGLTexture2D* pTexture)
{
	uint src_width = pTexture->GetWidth();
	uint src_height = pTexture->GetHeight();

	if(src_width != m_Width || src_height != m_Height)
	{
		std::cout << "COGLTexture2D.CopyData(): Cannot copy because dimensions are not the same" << std::endl;
	}

	GLenum src_internalformat = pTexture->GetInternalFormat();
	GLenum src_format = pTexture->GetFormat();
	GLenum src_type = pTexture->GetType();

	assert(src_internalformat == GL_RGBA32F);
	assert(m_InternalFormat == GL_RGBA32F);

	try
	{
		float* pData = new float[4 * sizeof(float) * m_Width * m_Height];
		
		{
			COGLBindLock lock(pTexture, COGL_TEXTURE0_SLOT);
			glGetTexImage(GL_TEXTURE_2D, 0, m_Format, m_Type, pData);
		}

		{
			COGLBindLock lock(this, COGL_TEXTURE0_SLOT);
			glTexImage2D(GL_TEXTURE_2D, 0, m_InternalFormat, m_Width, m_Height, 0, m_Format, m_Type, pData);
		}

		delete [] pData;
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at COGLTexture2D.CopyData()" << std::endl;
	}
}

void COGLTexture2D::Bind(COGLBindSlot slot)
{
	COGLResource::Bind(slot);
		
	glActiveTexture(GetGLSlot(m_Slot));
	glBindTexture(GL_TEXTURE_2D, m_Resource);
}

void COGLTexture2D::Unbind()
{
	COGLResource::Unbind();

	glActiveTexture(GetGLSlot(m_Slot));
	glBindTexture(GL_TEXTURE_2D, 0);
}

void COGLTexture2D::Clear()
{
	CheckNotBound("COGLTexture2D::Clear()");
	
	try{
		float* pData = new float[4 * sizeof(float) * m_Width * m_Height];

		memset(pData, 0, 4 * sizeof(float) * m_Width * m_Height);
	
		COGLBindLock lock(this, COGL_TEXTURE0_SLOT);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, pData);

		delete [] pData;
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at COGLTexture2D.Clear()" << std::endl;
		return;
	}
}
