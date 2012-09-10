#include "CReferenceImage.h"

#include "OGLResources\COGLTexture2D.h"

#include "CImage.h"

#include "Macros.h"
#include "Defines.h"
#include "Utils\Util.h"

CReferenceImage::CReferenceImage(uint width, uint height)
	: m_Width(width), m_Height(height)
{
	m_pImage = new CImage(m_Width, m_Height);
	m_pOGLTexture = new COGLTexture2D("CReferenceImage.m_pOGLTexture");
}

CReferenceImage::~CReferenceImage()
{
	SAFE_DELETE(m_pImage);
	SAFE_DELETE(m_pOGLTexture);
	
}

void CReferenceImage::LoadFromFile(const char* path, bool flipImage)
{
	m_pImage->LoadFromFile(path, flipImage);
	
	glm::vec4* pData = m_pImage->GetData();
	for(int i = 0; i < m_pImage->GetWidth() * m_pImage->GetHeight(); ++i)
	{
		pData[i] = glm::min(glm::vec4(100.f), pData[i]);
	}

	m_pOGLTexture->Init(m_Width, m_Height, GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false);
	m_pOGLTexture->SetPixelData(m_pImage->GetData());
}

void CReferenceImage::Release()
{
	m_pOGLTexture->Release();
}

float CReferenceImage::GetError(COGLTexture2D* comp)
{
	uint w = comp->GetWidth();
	uint h = comp->GetHeight();
	
	glm::vec4* pRefData = m_pImage->GetData();
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
