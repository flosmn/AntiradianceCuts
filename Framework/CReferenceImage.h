#ifndef _C_REFERENCE_IMAGE_H_
#define _C_REFERENCE_IMAGE_H_

#include <glm/glm.hpp>

class CImage;
class COGLTexture2D;

typedef unsigned int uint;

class CReferenceImage
{
public:
	CReferenceImage(uint width, uint height);
	~CReferenceImage();

	void Release();

	void LoadFromFile(const char* path, bool flipImage);
	CImage* GetImage() { return m_pImage; }
	COGLTexture2D* GetOGLTexture() { return m_pOGLTexture; }
	
	float GetError(COGLTexture2D* comp);
	
private:
	CImage* m_pImage;
	COGLTexture2D* m_pOGLTexture;
	
	uint m_Width;
	uint m_Height;
};

#endif _C_REFERENCE_IMAGE_H_