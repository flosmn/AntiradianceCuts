#ifndef _C_GL_TEXTURE_2D_H_
#define _C_GL_TEXTURE_2D_H_

typedef unsigned int uint;

#include "CGLResource.h"

class CGLTexture2D : public CGLResource
{
public:
	CGLTexture2D(std::string debugName);
	~CGLTexture2D();

	virtual bool Init(uint width, uint height, GLenum internalFormat,
		GLenum format, GLenum type, uint nMipMaps, bool genMipMaps);
	virtual void Release();

	void GenerateMipMaps();

	uint GetWidth() { CheckInitialized("CGLTexture2D.GetWidth()"); return m_Width; }
	uint GetHeight() { CheckInitialized("CGLTexture2D.GetHeight()"); return m_Height; }

	GLenum GetInternalFormat() { CheckInitialized("CGLTexture2D.GetInternalFormat()"); return m_InternalFormat; }
	GLenum GetFormat() { CheckInitialized("CGLTexture2D.GetFormat()"); return m_Format; }
	GLenum GetType() { CheckInitialized("CGLTexture2D.GetType()"); return m_Type; }

	void GetPixelData(void* pData);

	void CopyData(CGLTexture2D* pTexture);
	
private:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();

	uint m_Width;
	uint m_Height;
	uint m_nMipMaps;
	GLenum m_InternalFormat;
	GLenum m_Format;
	GLenum m_Type;
	bool m_GenMipMaps;
};

#endif // _C_GL_TEXTURE_2D_H_