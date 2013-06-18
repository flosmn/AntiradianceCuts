#ifndef _C_GL_TEXTURE_2D_H_
#define _C_GL_TEXTURE_2D_H_

typedef unsigned int uint;

#include "COGLResource.h"

class COGLTexture2D : public COGLResource
{
public:
	COGLTexture2D(uint width, uint height, GLenum internalFormat, GLenum format, 
		GLenum type, uint nMipMaps, bool genMipMaps, std::string const& debugName = "");
	~COGLTexture2D();

	void GenerateMipMaps();

	uint GetWidth() { return m_Width; }
	uint GetHeight() { return m_Height; }

	GLenum GetInternalFormat() { return m_InternalFormat; }
	GLenum GetFormat() { return m_Format; }
	GLenum GetType() { return m_Type; }

	void GetPixelData(void* pData);
	void SetPixelData(void* pData);

	void CopyData(COGLTexture2D* pTexture);

	void Clear();
	
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