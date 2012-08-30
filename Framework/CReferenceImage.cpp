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

	float error = 0.f;
	for(int i = 0; i < w; i++)
	{
		for(int j = 0; j < h; j++)
		{
			const float diff = Luminance(pRefData[j * w + i]) - Luminance(pCompData[j * w + i]);
			error += (diff * diff);
		}
	}
	error = sqrtf(error/float(w*h));

	delete [] pCompData;

	return error;
}
