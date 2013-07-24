#include "CImagePlane.h"

#include "Defines.h"

#include "CCamera.h"

#include "OGLResources\COGLTexture2D.h"
#include "CImage.h"
#include "Utils\Util.h"

#include <algorithm>

CImagePlane::CImagePlane(CCamera* pCamera)
{
	m_Width = pCamera->GetWidth();
	m_Height = pCamera->GetHeight();

	m_texture.reset(new COGLTexture2D(m_Width, m_Height, GL_RGBA32F, GL_RGBA, 
		GL_FLOAT, 1, false, "CImagePlane.m_pOGLTexture"));

	m_data.resize(m_Width * m_Height);
	m_numSamples.resize(m_Width * m_Height);

	for (size_t i = 0; i < m_data.size(); ++i) {
		m_data[i] = glm::vec4(0);
	}
	for (size_t i = 0; i < m_numSamples.size(); ++i) {
		m_numSamples[i] = 0;
	}

	m_CurrentPixelSample = glm::vec2(0.f, 0.f);
}

void CImagePlane::Clear()
{
	m_data.clear();
	m_numSamples.clear();
	m_data.resize(m_Width * m_Height);
	m_numSamples.resize(m_Width * m_Height);

	for (size_t i = 0; i < m_data.size(); ++i) {
		m_data[i] = glm::vec4(0);
	}
	for (size_t i = 0; i < m_numSamples.size(); ++i) {
		m_numSamples[i] = 0;
	}

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
	int n = m_numSamples[index];
	m_data[index] = 1.f/float(n+1) * (float(n) * m_data[index] + sample);
	m_numSamples[index]++;
}

COGLTexture2D* CImagePlane::GetOGLTexture()
{
	m_texture->SetPixelData(m_data.data());
	return m_texture.get();
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
