#include "CReferenceImage.h"

#include "OGLResources\COGLTexture2D.h"

#include "CImage.h"

#include "Defines.h"
#include "Utils\Util.h"

CReferenceImage::CReferenceImage(uint width, uint height)
	: m_Width(width), m_Height(height)
{
	m_image.reset(new CImage(m_Width, m_Height));
	m_texture.reset(new COGLTexture2D(m_Width, m_Height, GL_RGBA32F, 
		GL_RGBA, GL_FLOAT, 1, false, "CReferenceImage.m_pOGLTexture"));
}

CReferenceImage::~CReferenceImage()
{	
}

void CReferenceImage::LoadFromFile(const char* path, bool flipImage)
{
	m_image->LoadFromFile(path, flipImage);
	m_texture->SetPixelData(m_image->GetData());
}

float CReferenceImage::GetError(COGLTexture2D* comp)
{
	uint w = comp->GetWidth();
	uint h = comp->GetHeight();
	
	glm::vec4* pRefData = m_image->GetData();
	glm::vec4* pCompData = new glm::vec4[w * h];
	comp->GetPixelData(pCompData);

	glm::vec3 error = glm::vec3(0.f);
	for(uint i = 0; i < w*h; i++)
	{
		glm::vec3 v = glm::vec3(pRefData[i]); 
		glm::vec3 u = glm::vec3(pCompData[i]);
		glm::vec3 d = v - u;
		error += glm::min(glm::vec3(100.f), glm::vec3(d.x * d.x, d.y * d.y, d.z * d.z));
	}
	error = 1.f/float(w*h) * error;
	error = glm::vec3(sqrt(error.x), sqrt(error.y), sqrt(error.z));

	delete [] pCompData;

	return 1.f/3.f * (error.x * error.y + error.z);
}
