#ifndef _C_OCTAHEDRON_ATLAS_H_
#define _C_OCTAHEDRON_ATLAS_H_

#include <glm/glm.hpp>

typedef unsigned int uint;

class COGLTexture2D;

class COCLTexture2D;
class COCLContext;
class COCLProgram;
class COCLKernel;
class COCLBuffer;

struct CLUSTER;

class AVPL;

#include "Structs.h"

#include <vector>

class COctahedronAtlas
{
public:
	COctahedronAtlas(COCLContext* pOCLContext);
	~COctahedronAtlas();

	bool Init(uint atlasDim, uint tileDim, uint maxNumAVPLs);
	void Release();
	
	void FillClusterAtlas(const std::vector<AVPL*>& avpls, CLUSTER* pClustering, int clusteringSize);
	void FillClusterAtlasGPU(CLUSTER* pClustering, uint clusteringSize, uint numAVPLs);

	void FillAtlas(const std::vector<AVPL*>& avpls, const int sqrt_num_ss_samples, const float& N, bool border);
	void FillAtlasGPU(AVPL_BUFFER* pBufferData, uint numAVPLs, const int sqrt_num_ss_samples, const float& N, bool border);

	COGLTexture2D* GetAVPLAtlas();
	COGLTexture2D* GetAVPLAtlasDebug() { return m_pOGLAtlasDebug; };
	COGLTexture2D* GetAVPLAtlasCPU();
		
	COGLTexture2D* GetAVPLClusterAtlas();
	COGLTexture2D* GetAVPLClusterAtlasCPU();
	
	void Clear();

private:
	glm::vec4* AccessAtlas(uint x, uint y, uint tile_x, uint tile_y, glm::vec4* pAtlas);
	glm::vec4 SampleTexel(uint x, uint y, const int sqrt_num_ss_samples, const float& N, AVPL* avpl, bool border);

	COGLTexture2D* m_pOGLAtlasCPU;
	COGLTexture2D* m_pOGLAtlas;
	COGLTexture2D* m_pOGLAtlasDebug;
	COGLTexture2D* m_pOGLClusterAtlasCPU;
	COGLTexture2D* m_pOGLClusterAtlas;

	COCLTexture2D* m_pOCLAtlas;
	COCLTexture2D* m_pOCLClusterAtlas;

	uint m_AtlasDim;
	uint m_TileDim;

	COCLContext* m_pOCLContext;
	COCLProgram* m_pOCLProgram;
	COCLKernel* m_pOCLKernelClear;
	COCLKernel* m_pOCLCalcAvplAtlasKernel;
	COCLKernel* m_pOCLCalcAvplClusterAtlasKernel;
	COCLKernel* m_pOCLCopyToImageKernel;
	COCLBuffer* m_pAvplBuffer;
	COCLBuffer* m_pClusteringBuffer;
	COCLBuffer* m_pAtlasBuffer;
	COCLBuffer* m_pAtlasClusterBuffer;
	COCLBuffer* m_pIndexBuffer;

	glm::vec4* m_pData;
	glm::vec4* m_pClusterData;
};

#endif _C_OCTAHEDRON_ATLAS_H_