#include "CImagePlane.h"

#include "Defines.h"
#include "Macros.h"

#include "CCamera.h"

#include "OGLResources\COGLTexture2D.h"
#include "CImage.h"
#include "Utils\Util.h"

#include <algorithm>

CImagePlane::CImagePlane()
{
	m_pData = 0;
	m_pNumSamples = 0;
	m_pOGLTexture = new COGLTexture2D("CImagePlane.m_pOGLTexture");
}

CImagePlane::~CImagePlane()
{
	SAFE_DELETE(m_pOGLTexture);
	SAFE_DELETE_ARRAY(m_pData);
	SAFE_DELETE_ARRAY(m_pNumSamples);
}

bool CImagePlane::Init(CCamera* pCamera)
{
	m_Width = pCamera->GetWidth();
	m_Height = pCamera->GetHeight();

	V_RET_FOF(m_pOGLTexture->Init(m_Width, m_Height, GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));

	m_pNumSamples = new uint[m_Width * m_Height];
	memset(m_pNumSamples, 0, m_Width * m_Height * sizeof(uint));

	m_pData = new glm::vec4[m_Width * m_Height];
	memset(m_pData, 0, m_Width * m_Height * sizeof(glm::vec4));

	return true;
}

void CImagePlane::Release()
{
	m_pOGLTexture->Release();
}

void CImagePlane::Clear()
{
	m_pOGLTexture->Clear();
	memset(m_pNumSamples, 0, m_Width * m_Height * sizeof(uint));
	memset(m_pData, 0, m_Width * m_Height * sizeof(glm::vec4));
	m_CurrentPixelSample = glm::vec2(0.f, 0.f);
}

glm::vec2 CImagePlane::GetPixelSample()
{
	uint i = (PixelToIndex(m_CurrentPixelSample) + 1) % (m_Width * m_Height);
	m_CurrentPixelSample = IndexToPixel(i);
	return m_CurrentPixelSample;
}

void CImagePlane::AddSample(glm::vec2 pixelSample, glm::vec4 sample)
{
	if(is_nan(sample.x) || is_nan(sample.y) || is_nan(sample.z))
	{
		std::cout << "CImagePlane::AddSample: sample has NAN component" << std::endl;
		return;
	}

	if(is_inf(sample.x) || is_inf(sample.y) || is_inf(sample.z))
	{
		std::cout << "CImagePlane::AddSample: sample has INF component" << std::endl;
		return;
	}

	int index = PixelToIndex(pixelSample);
	int n = m_pNumSamples[index];
	m_pData[index] = 1.f/float(n+1) * (float(n) * m_pData[index] + sample);
	m_pNumSamples[index]++;
}

COGLTexture2D* CImagePlane::GetOGLTexture(bool blur)
{
	if(blur)
	{
		CImage image(m_Width, m_Height);
		image.SetData(m_pData);
		image.GaussianBlur(1);
		m_pOGLTexture->SetPixelData(image.GetData());
	}
	else
	{
		m_pOGLTexture->SetPixelData(m_pData);
	}
	
	return m_pOGLTexture;
}

uint CImagePlane::PixelToIndex(glm::vec2 pixel)
{
	pixel.x = floor(pixel.x);
	pixel.y = floor(pixel.y);

	if(pixel.x < 0 || pixel.x >= m_Width)
		std::cout << "pixel.x out of bounds" << std::endl;

	if(pixel.y < 0 || pixel.y >= m_Height)
		std::cout << "pixel.y out of bounds" << std::endl;

	pixel.x = std::min(float(m_Width-1),  pixel.x);
	pixel.y = std::min(float(m_Height-1), pixel.y);

	return uint(pixel.y) * m_Width + (m_Width-1) - uint(pixel.x);
}

glm::vec2 CImagePlane::IndexToPixel(uint index)
{
	if(index < 0 || index >= m_Width * m_Height)
		std::cout << "index out of bounds" << std::endl;

	return glm::vec2((m_Width-1) - index % m_Width, index / m_Width);
}