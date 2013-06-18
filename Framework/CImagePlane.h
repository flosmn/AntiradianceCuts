#ifndef _C_IMAGE_PLANE_H_
#define _C_IMAGE_PLANE_H_

#include <glm/glm.hpp>

#include <memory>
#include <vector>

typedef unsigned int uint;

class CCamera;

#include "OGLResources\COGLTexture2D.h"

class CImagePlane
{
public:
	CImagePlane(CCamera* pCamera);
	virtual ~CImagePlane() {}

	void Clear();
	glm::vec2 GetPixelSample();
	void AddSample(glm::vec2 pixelSample, glm::vec4 sample);

	COGLTexture2D* GetOGLTexture();

private:
	uint PixelToIndex(glm::vec2 pixel);
	glm::vec2 IndexToPixel(uint index);
	
	uint m_Width;
	uint m_Height;
	glm::vec2 m_CurrentPixelSample;

	std::vector<uint> m_numSamples;
	std::vector<glm::vec4> m_data;

	std::unique_ptr<COGLTexture2D> m_texture;
};

#endif _C_IMAGE_PLANE_H_
