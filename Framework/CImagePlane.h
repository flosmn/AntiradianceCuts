#ifndef _C_IMAGE_PLANE_H_
#define _C_IMAGE_PLANE_H_

#include <glm/glm.hpp>

typedef unsigned int uint;

class CCamera;
class COGLTexture2D;

class CImagePlane
{
public:
	CImagePlane();
	~CImagePlane();

	bool Init(CCamera* pCamera);
	void Release();

	void Clear();
	glm::vec2 GetPixelSample();
	void AddSample(glm::vec2 pixelSample, glm::vec4 sample);

	COGLTexture2D* GetOGLTexture(bool blur);

private:
	uint PixelToIndex(glm::vec2 pixel);
	glm::vec2 IndexToPixel(uint index);
	
	uint m_Width;
	uint m_Height;
	glm::vec2 m_CurrentPixelSample;
	COGLTexture2D* m_pOGLTexture;
	uint* m_pNumSamples;
	glm::vec4* m_pData;
};

#endif _C_IMAGE_PLANE_H_