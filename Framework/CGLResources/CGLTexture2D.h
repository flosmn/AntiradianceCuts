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