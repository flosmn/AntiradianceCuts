#include "COctahedronAtlas.h"

#include <glm/glm.hpp>

#include "Defines.h"
#include "Macros.h"

#include "AVPL.h"

#include "OctahedronUtil.h"

#include "OGLResources\COGLTexture2D.h"

#include <assert.h>

COctahedronAtlas::COctahedronAtlas()
{
	m_pAtlas = new COGLTexture2D("COctahedronAtlas.m_pAtlas");
}

COctahedronAtlas::~COctahedronAtlas()
{
	SAFE_DELETE(m_pAtlas);
}

bool COctahedronAtlas::Init(uint atlasDim, uint tileDim)
{
	m_AtlasDim = atlasDim;
	m_TileDim = tileDim;

	V_RET_FOF(m_pAtlas->Init(atlasDim, atlasDim, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, false));

	return true;
}

void COctahedronAtlas::Release()
{
	m_pAtlas->Release();
}

COGLTexture2D* COctahedronAtlas::GetTexture()
{
	m_pAtlas->CheckInitialized("COctahedronAtlas.GetTexture()");

	return m_pAtlas;
}

void COctahedronAtlas::FillAtlas(std::vector<AVPL*> avpls, const int sqrt_num_ss_samples, const float& N, bool border)
{
	uint maxIndex = avpls.size();
	if(avpls.size() * m_TileDim * m_TileDim > m_AtlasDim * m_AtlasDim)
	{
		maxIndex = (m_AtlasDim * m_AtlasDim) / (m_TileDim * m_TileDim);
		std::cout << "Warning: AVPL info does not fit into atlas!!!" << std::endl;
	}
	
	glm::vec4* pData = new glm::vec4[m_AtlasDim * m_AtlasDim];
	memset(pData, 0, m_AtlasDim * m_AtlasDim * sizeof(glm::vec4));
		
	uint columns = m_AtlasDim / m_TileDim;

	uint b = border ? 1 : 0;

	for(uint index = 0; index < maxIndex; ++index)
	{
		uint tile_x = index % columns;
		uint tile_y = index / columns;		
				
		for(uint x = b; x < m_TileDim-b; ++x)
		{
			for(uint y = b; y < m_TileDim-b; ++y)
			{
				*AccessAtlas(x, y, tile_x, tile_y, pData) = SampleTexel(x, y, sqrt_num_ss_samples, N, avpls[index]);
			}
		}
				
		if(border)
		{
			// top border
			for(uint x = 1; x < m_TileDim - 1; ++x)
			{
				*AccessAtlas(x, m_TileDim-1, tile_x, tile_y, pData) = *AccessAtlas(m_TileDim-1 - x, m_TileDim-2, tile_x, tile_y, pData);
			}

			// bottom border
			for(uint x = 1; x < m_TileDim - 1; ++x)
			{
				*AccessAtlas(x, 0, tile_x, tile_y, pData) = *AccessAtlas(m_TileDim-1 - x, 1, tile_x, tile_y, pData);
			}

			// left border
			for(uint y = 1; y < m_TileDim - 1; ++y)
			{
				*AccessAtlas(0, y, tile_x, tile_y, pData) = *AccessAtlas(1, m_TileDim-1 - y, tile_x, tile_y, pData);
			}

			// right border
			for(uint y = 1; y < m_TileDim - 1; ++y)
			{
				*AccessAtlas(m_TileDim - 1,	y, tile_x, tile_y, pData) = *AccessAtlas(m_TileDim - 2, m_TileDim-1 - y, tile_x, tile_y, pData);
			}

			// corners
			*AccessAtlas(0,				0,				tile_x, tile_y, pData) = *AccessAtlas(m_TileDim-2,	m_TileDim-2,	tile_x, tile_y, pData);
			*AccessAtlas(m_TileDim - 1, m_TileDim - 1,	tile_x, tile_y, pData) = *AccessAtlas(1,			1,				tile_x, tile_y, pData);
			*AccessAtlas(0,				m_TileDim - 1,	tile_x, tile_y, pData) = *AccessAtlas(m_TileDim-2,	1,				tile_x, tile_y, pData);
			*AccessAtlas(m_TileDim - 1, 0,				tile_x, tile_y, pData) = *AccessAtlas(1,			m_TileDim-2,	tile_x, tile_y, pData);
		}
	}
	
	m_pAtlas->SetPixelData(pData);

	delete [] pData;
}

glm::vec4 COctahedronAtlas::SampleTexel(uint x, uint y, const int sqrt_num_ss_samples, const float& N, AVPL* avpl)
{
	const int num_ss_samples = sqrt_num_ss_samples * sqrt_num_ss_samples;

	const float texel_size = 1.f/float(m_TileDim);
	const float delta = 1.f / float(sqrt_num_ss_samples + 2);

	glm::vec2* texCoords = new glm::vec2[num_ss_samples];
	for(int i = 0; i < sqrt_num_ss_samples; ++i)
	{
		for(int j = 0; j < sqrt_num_ss_samples; ++j)
		{
			texCoords[j + i * sqrt_num_ss_samples] = texel_size * glm::vec2(float(x + (i+1) * delta), float(y + (j+1) * delta));
		}
	}
		
	glm::vec3* directions = new glm::vec3[num_ss_samples];
	for(int i = 0; i < num_ss_samples; ++i)
	{
		directions[i] = glm::normalize(GetDirectionForTexCoord(texCoords[i]));
	}
	delete [] texCoords;

	glm::vec3 A = glm::vec3(0.f);
	glm::vec3 I = glm::vec3(0.f);
	
	for(int i = 0; i < num_ss_samples; ++i)
	{
		A += avpl->GetAntiintensity(directions[i], N);
		I += avpl->GetIntensity(directions[i]);
	}
	delete [] directions;
	
	A *= 1.f/float(num_ss_samples);
	I *= 1.f/float(num_ss_samples);
	
	return glm::vec4(I - A, 1.f);
}

glm::vec4* COctahedronAtlas::AccessAtlas(uint x, uint y, uint tile_x, uint tile_y, glm::vec4* pAtlas)
{
	const uint tile_offset = tile_y * (m_AtlasDim * m_TileDim) + tile_x * m_TileDim;
	const uint offset = tile_offset + y * m_AtlasDim + x;
	return &(pAtlas[offset]);
}

