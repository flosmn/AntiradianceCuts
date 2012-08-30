#ifndef _C_IMAGE_H_
#define _C_IMAGE_H_

#include <glm/glm.hpp>

class CImage
{
public:
	CImage(int width, int height);
	~CImage();

	void LoadFromFile(const char* path, bool flipImage);
	
	void SetData(glm::vec4* pData);
	glm::vec4* GetData();

	void GaussianBlur(int iterations);

private:
	int m_Width;
	int m_Height;

	glm::vec4* m_pData;
};

#endif _C_IMAGE_H_