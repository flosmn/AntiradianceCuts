#ifndef _C_GL_TEXTURE_2D_H_
#define _C_GL_TEXTURE_2D_H_

typedef unsigned int uint;

#include "COGLResource.h"

class COGLTexture2D : public COGLResource
{
public:
	COGLTexture2D(std::string debugName);
	~COGLTexture2D();

	virtual bool Init(uint width, uint height, GLenum internalFormat,
		GLenum format, GLenum type, uint nMipMaps, bool genMipMaps);
	virtual void Release();

	void GenerateMipMaps();

	uint GetWidth() { CheckInitialized("COGLTexture2D.GetWidth()"); return m_Width; }
	uint GetHeight() { CheckInitialized("COGLTexture2D.GetHeight()"); return m_Height; }

	GLenum GetInternalFormat() { CheckInitialized("COGLTexture2D.GetInternalFormat()"); return m_InternalFormat; }
	GLenum GetFormat() { CheckInitialized("COGLTexture2D.GetFormat()"); return m_Format; }
	GLenum GetType() { CheckInitialized("COGLTexture2D.GetType()"); return m_Type; }

	void GetPixelData(void* pData);
	void SetPixelData(void* pData);

	void CopyData(COGLTexture2D* pTexture);	
	
private:
	virtual void Bind(COGLBindSlot slot);
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