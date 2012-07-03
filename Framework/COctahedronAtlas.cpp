#include "COctahedronAtlas.h"

#include <glm/glm.hpp>

#include "Defines.h"
#include "Macros.h"

#include "AVPL.h"

#include "OctahedronUtil.h"

#include "OGLResources\COGLTexture2D.h"

#include "OCLResources\COCLContext.h"
#include "OCLResources\COCLProgram.h"
#include "OCLResources\COCLKernel.h"
#include "OCLResources\COCLTexture2D.h"
#include "OCLResources\COCLBuffer.h"

#include "Utils\GLErrorUtil.h"

#include <algorithm>
#include <assert.h>

COctahedronAtlas::COctahedronAtlas(COCLContext* pOCLContext)
	: m_pOCLContext(pOCLContext)
{
	m_pOCLProgram = new COCLProgram(pOCLContext, "COctahedronAtlas.m_pOCLProgram");
	m_pOCLKernel = new COCLKernel(m_pOCLContext, m_pOCLProgram, "COctahedronAtlas.m_pOCLKernel");
	m_pOCLKernelClear = new COCLKernel(m_pOCLContext, m_pOCLProgram, "COctahedronAtlas.m_pOCLKernelClear");

	m_pAvplBuffer = new COCLBuffer(m_pOCLContext, "COctahedronAtlas.m_pAvplBuffer");
	m_pLocalBuffer = new COCLBuffer(m_pOCLContext, "COctahedronAtlas.m_pLocalBuffer");

	m_pOCLAtlas = new COCLTexture2D(m_pOCLContext, "COctahedronAtlas.m_pOCLAtlas");
	m_pOGLAtlas = new COGLTexture2D("COctahedronAtlas.m_pOGLAtlas");
	m_pOGLAtlasRef = new COGLTexture2D("COctahedronAtlas.m_pOGLAtlasRef");
}

COctahedronAtlas::~COctahedronAtlas()
{
	SAFE_DELETE(m_pOCLAtlas);
	SAFE_DELETE(m_pOCLKernel);
	SAFE_DELETE(m_pOCLKernelClear);
	SAFE_DELETE(m_pOCLProgram);

	SAFE_DELETE(m_pAvplBuffer);
	SAFE_DELETE(m_pOGLAtlas);
	SAFE_DELETE(m_pOGLAtlasRef);
}

bool COctahedronAtlas::Init(uint atlasDim, uint tileDim, uint maxNumAVPLs)
{
	m_AtlasDim = atlasDim;
	m_TileDim = tileDim;
		
	V_RET_FOF(m_pOCLProgram->Init("Kernels\\FillAVPLAtlas.cl"));
	V_RET_FOF(m_pOCLKernel->Init("Fill"));
	V_RET_FOF(m_pOCLKernelClear->Init("Clear"));

	V_RET_FOF(m_pOGLAtlas->Init(atlasDim, atlasDim, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, false));
	V_RET_FOF(m_pOGLAtlasRef->Init(atlasDim, atlasDim, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, false));

	V_RET_FOF(m_pOCLAtlas->Init(m_pOGLAtlas));

	V_RET_FOF(m_pAvplBuffer->Init(sizeof(AVPL_BUFFER) * maxNumAVPLs, CL_MEM_READ_WRITE));
	
	m_pOCLKernel->SetKernelArg(0, sizeof(cl_mem), m_pOCLAtlas->GetCLTexture());
	m_pOCLKernel->SetKernelArg(1, sizeof(int), &m_TileDim);
	m_pOCLKernel->SetKernelArg(2, sizeof(int), &m_AtlasDim);
	m_pOCLKernel->SetKernelArg(8, sizeof(cl_mem), m_pAvplBuffer->GetCLBuffer());
		
	m_pOCLKernelClear->SetKernelArg(0, sizeof(cl_mem), m_pOCLAtlas->GetCLTexture());

	return true;
}

void COctahedronAtlas::Release()
{
	m_pOCLAtlas->Release();
	m_pOCLKernel->Release();
	m_pOCLKernelClear->Release();
	m_pOCLProgram->Release();
	m_pAvplBuffer->Release();
	m_pOGLAtlas->Release();
	m_pOGLAtlasRef->Release();
}

COGLTexture2D* COctahedronAtlas::GetTexture()
{
	m_pOGLAtlas->CheckInitialized("COctahedronAtlas.GetTexture()");

	return m_pOGLAtlas;
}

COGLTexture2D* COctahedronAtlas::GetRefTexture()
{
	m_pOGLAtlasRef->CheckInitialized("COctahedronAtlas.GetRefTexture()");

	return m_pOGLAtlasRef;
}

void COctahedronAtlas::FillAtlasGPU(AVPL_BUFFER* pBufferData, uint numAVPLs, const int sqrt_num_ss_samples, const float& N, bool border)
{
	size_t local_work_size[2];
	local_work_size[0] = m_TileDim < 16 ? m_TileDim : 16;
	local_work_size[1] = m_TileDim < 16 ? m_TileDim : 16;
	
	int numColumns = m_AtlasDim / m_TileDim;
	int numRows = numAVPLs / numColumns + 1;

	size_t global_work_size[2];
	global_work_size[0] = local_work_size[0] * numColumns;
	global_work_size[1] = local_work_size[1] * numRows;
		
	m_pAvplBuffer->SetBufferData(pBufferData, sizeof(AVPL_BUFFER) * numAVPLs, true);
	
	int n = int(N);
	int b = int(border);

	m_pOCLKernel->SetKernelArg(3, sizeof(int), &numAVPLs);
	m_pOCLKernel->SetKernelArg(4, sizeof(int), &sqrt_num_ss_samples);
	m_pOCLKernel->SetKernelArg(5, sizeof(int), &n);
	m_pOCLKernel->SetKernelArg(6, sizeof(int), &b);
	m_pOCLKernel->SetKernelArg(7, 4 * sizeof(float) * m_TileDim * m_TileDim, NULL);

	m_pOCLAtlas->Lock();
	
	m_pOCLKernel->CallKernel(2, 0, global_work_size, local_work_size);
		
	m_pOCLAtlas->Unlock();
}

void COctahedronAtlas::FillAtlas(std::vector<AVPL*> avpls, const int sqrt_num_ss_samples, const float& N, bool border)
{
	uint maxIndex = (uint)avpls.size();
	if(avpls.size() * m_TileDim * m_TileDim > m_AtlasDim * m_AtlasDim)
	{
		maxIndex = (m_AtlasDim * m_AtlasDim) / (m_TileDim * m_TileDim);
		std::cout << "Warning: AVPL info does not fit into atlas!!!" << std::endl;
	}
	
	try
	{
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
					*AccessAtlas(x, y, tile_x, tile_y, pData) = SampleTexel(x-b, y-b, sqrt_num_ss_samples, N, avpls[index], border);
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
		
		m_pOGLAtlasRef->SetPixelData(pData);

		delete [] pData;
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at COctahedronAtlas::FillAtlas()" << std::endl;
	}
}

glm::vec4 COctahedronAtlas::SampleTexel(uint x, uint y, const int sqrt_num_ss_samples, const float& N, AVPL* avpl, bool border)
{
	uint b = border ? 1 : 0;
	const int num_ss_samples = sqrt_num_ss_samples * sqrt_num_ss_samples;

	const float texel_size = 1.f/float(m_TileDim - 2 * b);
	const float delta = 1.f / float(sqrt_num_ss_samples + 1);

	glm::vec3 A = glm::vec3(0.f);
	glm::vec3 I = glm::vec3(0.f);

	for(int i = 0; i < sqrt_num_ss_samples; ++i)
	{
		for(int j = 0; j < sqrt_num_ss_samples; ++j)
		{
			glm::vec2 texCoord = texel_size * glm::vec2(float(x + (i+1) * delta), float(y + (j+1) * delta));
			glm::vec3 direction = glm::normalize(GetDirForTexCoord(texCoord));
			
			A += avpl->GetAntiintensity(direction, N);
			I += avpl->GetIntensity(direction);
		}
	}		
		
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

void COctahedronAtlas::Clear()
{
	size_t local_work_size[2];
	local_work_size[0] = m_TileDim < m_pOCLContext->GetMaxWorkGroupDimensions2DSquare()[0] ? m_TileDim : m_pOCLContext->GetMaxWorkGroupDimensions2DSquare()[0];
	local_work_size[1] = m_TileDim < m_pOCLContext->GetMaxWorkGroupDimensions2DSquare()[1] ? m_TileDim : m_pOCLContext->GetMaxWorkGroupDimensions2DSquare()[1];
		
	size_t global_work_size[2];
	global_work_size[0] = m_AtlasDim;
	global_work_size[1] = m_AtlasDim;
		
	m_pOCLAtlas->Lock();
		
	m_pOCLKernelClear->CallKernel(2, 0, global_work_size, local_work_size);
	
	m_pOCLAtlas->Unlock();
}