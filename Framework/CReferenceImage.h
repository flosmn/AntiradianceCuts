#ifndef _C_REFERENCE_IMAGE_H_
#define _C_REFERENCE_IMAGE_H_

#include <glm/glm.hpp>

#include <memory>

class CImage;
class COGLTexture2D;

typedef unsigned int uint;

class CReferenceImage
{
public:
	CReferenceImage(uint width, uint height);
	~CReferenceImage();
	
	void LoadFromFile(const char* path, bool flipImage);
	CImage* GetImage() { return m_image.get(); }
	COGLTexture2D* GetOGLTexture() { return m_texture.get(); }
	
	float GetError(COGLTexture2D* comp);
	
private:
	std::unique_ptr<CImage> m_image;
	std::unique_ptr<COGLTexture2D> m_texture;
	
	uint m_Width;
	uint m_Height;
};

#endif _C_REFERENCE_IMAGE_H_