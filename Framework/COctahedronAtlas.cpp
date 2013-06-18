#include "COctahedronAtlas.h"

#include <glm/glm.hpp>

#include "Defines.h"
#include "Macros.h"

#include "AVPL.h"
#include "LightTreeTypes.h"
#include "CMaterialBuffer.h"

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

COctahedronAtlas::COctahedronAtlas(COCLContext* pOCLContext, uint atlasDim, uint tileDim, uint maxNumAVPLs, CMaterialBuffer* pMaterialBuffer)
	: m_pOCLContext(pOCLContext), m_AtlasDim(atlasDim), m_TileDim(tileDim), m_pMaterialBuffer(pMaterialBuffer)
{
	m_pOCLProgram = new COCLProgram(pOCLContext, "COctahedronAtlas.m_pOCLProgram");
	m_pOCLCopyToImageKernel = new COCLKernel(m_pOCLContext, m_pOCLProgram, "COctahedronAtlas.m_pOCLCopyToImageKernel");
	m_pOCLCalcAvplAtlasKernel = new COCLKernel(m_pOCLContext, m_pOCLProgram, "COctahedronAtlas.m_pOCLCalcAvplClusterAtlasKernel");
	m_pOCLCalcAvplClusterAtlasKernel = new COCLKernel(m_pOCLContext, m_pOCLProgram, "COctahedronAtlas.m_pOCLCalcAvplAtlasKernel");
	m_pOCLKernelClear = new COCLKernel(m_pOCLContext, m_pOCLProgram, "COctahedronAtlas.m_pOCLKernelClear");

	m_pAvplBuffer = new COCLBuffer(m_pOCLContext, "COctahedronAtlas.m_pAvplBuffer");
	m_pClusteringBuffer = new COCLBuffer(m_pOCLContext, "COctahedronAtlas.m_pClusteringBuffer");

	m_pAtlasBuffer = new COCLBuffer(m_pOCLContext, "COctahedronAtlas.m_pAtlasBuffer");
	m_pAtlasClusterBuffer = new COCLBuffer(m_pOCLContext, "COctahedronAtlas.m_pAtlasClusterBuffer");

	m_pIndexBuffer = new COCLBuffer(m_pOCLContext, "COctahedronAtlas.m_pIndexBuffer");

	m_pOCLAtlas = new COCLTexture2D(m_pOCLContext, "COctahedronAtlas.m_pOCLAtlas");
	m_pOCLClusterAtlas = new COCLTexture2D(m_pOCLContext, "COctahedronAtlas.m_pOCLClusterAtlas");
	
	m_oglAtlas			.reset(new COGLTexture2D(atlasDim, atlasDim, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, false, "COctahedronAtlas.m_pOGLAtlas"));
	m_oglAtlasDebug		.reset(new COGLTexture2D(atlasDim, atlasDim, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, false, "COctahedronAtlas.m_pOGLAtlasDebug"));
	m_oglAtlasCPU		.reset(new COGLTexture2D(atlasDim, atlasDim, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, false, "COctahedronAtlas.m_pOGLAtlasCPU"));
	m_oglClusterAtlas	.reset(new COGLTexture2D(atlasDim, atlasDim, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, false, "COctahedronAtlas.m_pOGLClusterAtlas"));
	m_oglClusterAtlasCPU.reset(new COGLTexture2D(atlasDim, atlasDim, GL_RGBA32F, GL_RGBA, GL_FLOAT, 0, false, "COctahedronAtlas.m_pOGLClusterAtlasCPU"));

	m_pOCLProgram->Init("Kernels\\FillAVPLAtlas.cl");
	m_pOCLCalcAvplAtlasKernel->Init("CalcAvplAtlas");
	m_pOCLCalcAvplClusterAtlasKernel->Init("CalcAvplClusterAtlas");
	m_pOCLCopyToImageKernel->Init("CopyToImage");
	m_pOCLKernelClear->Init("Clear");

	m_pAtlasBuffer->Init(atlasDim * atlasDim * 4 * sizeof(float), CL_MEM_READ_WRITE);
	m_pAtlasClusterBuffer->Init(atlasDim * atlasDim * 4 * sizeof(float), CL_MEM_READ_WRITE);
	
	m_pIndexBuffer->Init(sizeof(int), CL_MEM_READ_WRITE);

	m_pOCLAtlas->Init(m_oglAtlas.get());
	m_pOCLClusterAtlas->Init(m_oglClusterAtlas.get());

	m_pAvplBuffer->Init(sizeof(AVPL_BUFFER) * maxNumAVPLs, CL_MEM_READ_WRITE);
	m_pClusteringBuffer->Init(sizeof(CLUSTERING) * (2 * maxNumAVPLs) - 1, CL_MEM_READ_WRITE);
	
	m_pOCLCalcAvplAtlasKernel->SetKernelArg(0, sizeof(cl_mem), m_pAtlasBuffer->GetCLBuffer());
	m_pOCLCalcAvplAtlasKernel->SetKernelArg(1, sizeof(int), &m_TileDim);
	m_pOCLCalcAvplAtlasKernel->SetKernelArg(2, sizeof(int), &m_AtlasDim);
	m_pOCLCalcAvplAtlasKernel->SetKernelArg(7, sizeof(cl_mem), m_pAvplBuffer->GetCLBuffer());
	m_pOCLCalcAvplAtlasKernel->SetKernelArg(8, sizeof(cl_mem), m_pMaterialBuffer->GetOCLMaterialBuffer()->GetCLBuffer());

	m_pOCLCalcAvplClusterAtlasKernel->SetKernelArg(0, sizeof(cl_mem), m_pClusteringBuffer->GetCLBuffer());
	m_pOCLCalcAvplClusterAtlasKernel->SetKernelArg(1, sizeof(cl_mem), m_pAtlasBuffer->GetCLBuffer());
	m_pOCLCalcAvplClusterAtlasKernel->SetKernelArg(2, sizeof(cl_mem), m_pAtlasClusterBuffer->GetCLBuffer());
	m_pOCLCalcAvplClusterAtlasKernel->SetKernelArg(3, sizeof(int), &m_TileDim);
	m_pOCLCalcAvplClusterAtlasKernel->SetKernelArg(4, sizeof(int), &m_AtlasDim);
	m_pOCLCalcAvplClusterAtlasKernel->SetKernelArg(7, sizeof(cl_mem), m_pIndexBuffer->GetCLBuffer());

	m_pOCLCopyToImageKernel->SetKernelArg(1, sizeof(int), &m_TileDim);
	m_pOCLCopyToImageKernel->SetKernelArg(2, sizeof(int), &m_AtlasDim);

	m_pData = new glm::vec4[m_AtlasDim * m_AtlasDim];
	m_pClusterData = new glm::vec4[m_AtlasDim * m_AtlasDim];
	memset(m_pData, 0, m_AtlasDim * m_AtlasDim * sizeof(glm::vec4));
	memset(m_pClusterData, 0, m_AtlasDim * m_AtlasDim * sizeof(glm::vec4));
}

COctahedronAtlas::~COctahedronAtlas()
{
	m_pAtlasBuffer->Release();
	m_pAtlasClusterBuffer->Release();
	m_pOCLAtlas->Release();
	m_pOCLClusterAtlas->Release();
	m_pOCLKernelClear->Release();
	m_pOCLCalcAvplAtlasKernel->Release();
	m_pOCLCalcAvplClusterAtlasKernel->Release();
	m_pOCLCopyToImageKernel->Release();
	m_pOCLProgram->Release();
	m_pAvplBuffer->Release();
	m_pClusteringBuffer->Release();
	m_pIndexBuffer->Release();
	
	SAFE_DELETE(m_pOCLAtlas);
	SAFE_DELETE(m_pOCLClusterAtlas);
	SAFE_DELETE(m_pOCLCopyToImageKernel);
	SAFE_DELETE(m_pOCLCalcAvplClusterAtlasKernel);
	SAFE_DELETE(m_pOCLKernelClear);
	SAFE_DELETE(m_pOCLProgram);
	SAFE_DELETE(m_pAtlasBuffer);
	SAFE_DELETE(m_pAtlasClusterBuffer);
	SAFE_DELETE(m_pAvplBuffer);
	SAFE_DELETE(m_pClusteringBuffer);
	SAFE_DELETE(m_pIndexBuffer);
}

void COctahedronAtlas::FillAtlasGPU(const std::vector<AVPL>& avpls, const int sqrt_num_ss_samples, const float& N, bool border)
{
	uint numAVPLs = (uint)avpls.size();
	AVPL_BUFFER* avplBuffer = new AVPL_BUFFER[numAVPLs];
	memset(avplBuffer, 0, sizeof(AVPL_BUFFER) * numAVPLs);
	for(uint i = 0; i < numAVPLs; ++i)
	{
		AVPL_BUFFER buffer;
		avpls[i].Fill(buffer);
		avplBuffer[i] = buffer;
	}
	
	size_t local_work_size[2];
	local_work_size[0] = m_TileDim < 16 ? m_TileDim : 16;
	local_work_size[1] = m_TileDim < 16 ? m_TileDim : 16;
	
	int numColumns = m_AtlasDim / m_TileDim;
	int numRows = numAVPLs / numColumns + 1;
	
	size_t global_work_size[2];
	global_work_size[0] = local_work_size[0] * numColumns;
	global_work_size[1] = local_work_size[1] * numRows;
	
	m_pAvplBuffer->SetBufferData(avplBuffer, sizeof(AVPL_BUFFER) * numAVPLs, true);
	
	int n = int(N);
	int b = int(border);
	
	m_pOCLCalcAvplAtlasKernel->SetKernelArg(3, sizeof(int), &numAVPLs);
	m_pOCLCalcAvplAtlasKernel->SetKernelArg(4, sizeof(int), &sqrt_num_ss_samples);
	m_pOCLCalcAvplAtlasKernel->SetKernelArg(5, sizeof(int), &n);
	m_pOCLCalcAvplAtlasKernel->SetKernelArg(6, sizeof(int), &b);
	m_pOCLCalcAvplClusterAtlasKernel->SetKernelArg(5, sizeof(int), &numAVPLs);
	m_pOCLCopyToImageKernel->SetKernelArg(3, sizeof(int), &numAVPLs);	
			
	m_pOCLCalcAvplAtlasKernel->CallKernel(2, 0, global_work_size, local_work_size);
		
	clFinish(*(m_pOCLContext->GetCLCommandQueue()));
	
	m_pOCLCopyToImageKernel->SetKernelArg(0, sizeof(cl_mem), m_pOCLAtlas->GetCLTexture());
	m_pOCLCopyToImageKernel->SetKernelArg(4, sizeof(cl_mem), m_pAtlasBuffer->GetCLBuffer());

	m_pOCLAtlas->Lock();
	m_pOCLCopyToImageKernel->CallKernel(2, 0, global_work_size, local_work_size);
	m_pOCLAtlas->Unlock();

	delete [] avplBuffer;
}

void COctahedronAtlas::FillClusterAtlasGPU(CLUSTER* pClustering, uint clusteringSize, uint numAVPLs)
{
	int numInnerNodes = int(clusteringSize - numAVPLs);
	
	size_t local_work_size[2];
	local_work_size[0] = m_TileDim < 16 ? m_TileDim : 16;
	local_work_size[1] = m_TileDim < 16 ? m_TileDim : 16;
	
	int numColumns = m_AtlasDim / m_TileDim;
	int numRowsClustering = numInnerNodes / numColumns + 1;

	size_t global_work_size[2];
	global_work_size[0] = local_work_size[0] * numColumns;
	global_work_size[1] = local_work_size[1] * numRowsClustering;
		
	CLUSTERING* pClusteringData = new CLUSTERING[clusteringSize];
	for(uint i = 0; i < clusteringSize; ++i)
	{
		CLUSTERING c;
		if(i < numAVPLs)
		{
			c.isAlreadyCalculated = 1;
			c.isLeaf = 1;
			c.leftChildId = -1;
			c.rightChildId = -1;
		}
		else
		{
			c.isAlreadyCalculated = 0;
			c.isLeaf = 0;
			c.leftChildId = pClustering[i].left->id;
			c.rightChildId = pClustering[i].right->id;
		}

		pClusteringData[i] = c;
	}

	m_pClusteringBuffer->SetBufferData(pClusteringData, sizeof(CLUSTERING) * clusteringSize, true);
	
	delete [] pClusteringData;

	int index = 0;
	m_pIndexBuffer->SetBufferData(&index, sizeof(int), false);
		
	m_pOCLCalcAvplClusterAtlasKernel->SetKernelArg(5, sizeof(int), &numAVPLs);
	m_pOCLCalcAvplClusterAtlasKernel->SetKernelArg(6, sizeof(int), &numInnerNodes);
	m_pOCLCopyToImageKernel->SetKernelArg(3, sizeof(int), &numAVPLs);	
	
	clFinish(*(m_pOCLContext->GetCLCommandQueue()));

	m_pOCLCalcAvplClusterAtlasKernel->CallKernel(2, 0, global_work_size, local_work_size);
		
	m_pOCLCopyToImageKernel->SetKernelArg(0, sizeof(cl_mem), m_pOCLClusterAtlas->GetCLTexture());
	m_pOCLCopyToImageKernel->SetKernelArg(4, sizeof(cl_mem), m_pAtlasClusterBuffer->GetCLBuffer());

	m_pOCLClusterAtlas->Lock();
	m_pOCLCopyToImageKernel->CallKernel(2, 0, global_work_size, local_work_size);
	m_pOCLClusterAtlas->Unlock();
}

void COctahedronAtlas::FillAtlas(const std::vector<AVPL>& avpls, const int sqrt_num_ss_samples, const float& N, bool border)
{
	uint maxIndex = (uint)avpls.size();
	if(avpls.size() * m_TileDim * m_TileDim > m_AtlasDim * m_AtlasDim)
	{
		maxIndex = (m_AtlasDim * m_AtlasDim) / (m_TileDim * m_TileDim);
		std::cout << "Warning: AVPL info does not fit into atlas!!!" << std::endl;
	}
	
	memset(m_pData, 0, m_AtlasDim * m_AtlasDim * sizeof(glm::vec4));

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
				*AccessAtlas(x, y, tile_x, tile_y, m_pData) = SampleTexel(x-b, y-b, sqrt_num_ss_samples, N, avpls[index], border);
			}
		}
				
		if(border)
		{
			// top border
			for(uint x = 1; x < m_TileDim - 1; ++x)
			{
				*AccessAtlas(x, m_TileDim-1, tile_x, tile_y, m_pData) = *AccessAtlas(m_TileDim-1 - x, m_TileDim-2, tile_x, tile_y, m_pData);
			}

			// bottom border
			for(uint x = 1; x < m_TileDim - 1; ++x)
			{
				*AccessAtlas(x, 0, tile_x, tile_y, m_pData) = *AccessAtlas(m_TileDim-1 - x, 1, tile_x, tile_y, m_pData);
			}

			// left border
			for(uint y = 1; y < m_TileDim - 1; ++y)
			{
				*AccessAtlas(0, y, tile_x, tile_y, m_pData) = *AccessAtlas(1, m_TileDim-1 - y, tile_x, tile_y, m_pData);
			}

			// right border
			for(uint y = 1; y < m_TileDim - 1; ++y)
			{
				*AccessAtlas(m_TileDim - 1,	y, tile_x, tile_y, m_pData) = *AccessAtlas(m_TileDim - 2, m_TileDim-1 - y, tile_x, tile_y, m_pData);
			}

			// corners
			*AccessAtlas(0,				0,				tile_x, tile_y, m_pData) = *AccessAtlas(m_TileDim-2,	m_TileDim-2,	tile_x, tile_y, m_pData);
			*AccessAtlas(m_TileDim - 1, m_TileDim - 1,	tile_x, tile_y, m_pData) = *AccessAtlas(1,			1,				tile_x, tile_y, m_pData);
			*AccessAtlas(0,				m_TileDim - 1,	tile_x, tile_y, m_pData) = *AccessAtlas(m_TileDim-2,	1,				tile_x, tile_y, m_pData);
			*AccessAtlas(m_TileDim - 1, 0,				tile_x, tile_y, m_pData) = *AccessAtlas(1,			m_TileDim-2,	tile_x, tile_y, m_pData);
		}
		
	}
	
	m_oglAtlasCPU->SetPixelData(m_pData);
}

void COctahedronAtlas::FillClusterAtlas(const std::vector<AVPL>& avpls, CLUSTER* pClustering, int clusteringSize)
{
	memset(m_pClusterData, 0, m_AtlasDim * m_AtlasDim * sizeof(glm::vec4));
	
	uint columns = m_AtlasDim / m_TileDim;

	// compute cluster information
	int numClusterNodes = int(clusteringSize - avpls.size());
	int offsetClusterNodes = int(avpls.size());

	for(int i = 0; i < numClusterNodes; ++i)
	{
		int innerNodeIndex = pClustering[i + offsetClusterNodes].id - int(avpls.size());
		int leftChildIndex = pClustering[i + offsetClusterNodes].left->id;
		int rightChildIndex = pClustering[i + offsetClusterNodes].right->id;

		bool leftChildInnerNode = false;
		bool rightChildInnerNode = false;
		if(leftChildIndex >= avpls.size())
		{
			leftChildIndex -= int(avpls.size());
			leftChildInnerNode = true;
		}
		if(rightChildIndex >= avpls.size())
		{
			rightChildIndex -= int(avpls.size());
			rightChildInnerNode = true;
		}
		
		uint columns = m_AtlasDim / m_TileDim;
					
		uint left_child_tile_x = leftChildIndex % columns;
		uint left_child_tile_y = leftChildIndex / columns;

		uint right_child_tile_x = rightChildIndex % columns;
		uint right_child_tile_y = rightChildIndex / columns;

		uint inner_node_tile_x = innerNodeIndex % columns;
		uint inner_node_tile_y = innerNodeIndex / columns;
					
		for(uint x = 0; x < m_TileDim; ++x)
		{
			for(uint y = 0; y < m_TileDim; ++y)
			{
				glm::vec4 leftChildTexel;
				glm::vec4 rightChildTexel;

				if(leftChildInnerNode)
					leftChildTexel = *AccessAtlas(x, y, left_child_tile_x, left_child_tile_y, m_pClusterData);
				else
					leftChildTexel = *AccessAtlas(x, y, left_child_tile_x, left_child_tile_y, m_pData);
				
				if(rightChildInnerNode)
					rightChildTexel = *AccessAtlas(x, y, right_child_tile_x, right_child_tile_y, m_pClusterData);
				else
					rightChildTexel = *AccessAtlas(x, y, right_child_tile_x, right_child_tile_y, m_pData);
				
				*AccessAtlas(x, y, inner_node_tile_x, inner_node_tile_y, m_pClusterData) = rightChildTexel + leftChildTexel;
			}
		}
	}

	m_oglClusterAtlasCPU->SetPixelData(m_pClusterData);
}

glm::vec4 COctahedronAtlas::SampleTexel(uint x, uint y, const int sqrt_num_ss_samples, const float& N, const AVPL& avpl, bool border)
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
			
			A += avpl.GetAntiradiance(direction);
			I += avpl.GetRadiance(direction);
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
	local_work_size[0] = 16; //m_TileDim < m_pOCLContext->GetMaxWorkGroupDimensions2DSquare()[0] ? m_TileDim : m_pOCLContext->GetMaxWorkGroupDimensions2DSquare()[0];
	local_work_size[1] = 16; //m_TileDim < m_pOCLContext->GetMaxWorkGroupDimensions2DSquare()[1] ? m_TileDim : m_pOCLContext->GetMaxWorkGroupDimensions2DSquare()[1];
		
	size_t global_work_size[2];
	global_work_size[0] = m_AtlasDim;
	global_work_size[1] = m_AtlasDim;
	
	m_pOCLAtlas->Lock();
	m_pOCLKernelClear->SetKernelArg(0, sizeof(cl_mem), m_pOCLAtlas->GetCLTexture());	
	m_pOCLKernelClear->CallKernel(2, 0, global_work_size, local_work_size);
	m_pOCLAtlas->Unlock();	

	m_pOCLClusterAtlas->Lock();
	m_pOCLKernelClear->SetKernelArg(0, sizeof(cl_mem), m_pOCLClusterAtlas->GetCLTexture());	
	m_pOCLKernelClear->CallKernel(2, 0, global_work_size, local_work_size);
	m_pOCLClusterAtlas->Unlock();
}