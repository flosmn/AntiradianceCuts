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

class CMaterialBuffer;

struct CLUSTER;

class AVPL;

#include "Structs.h"

#include <vector>
#include <memory>

class COctahedronAtlas
{
public:
	COctahedronAtlas(COCLContext* pOCLContext, uint atlasDim, uint tileDim, uint maxNumAVPLs, CMaterialBuffer* pMaterialBuffer);
	~COctahedronAtlas();
	
	void FillClusterAtlas(const std::vector<AVPL>& avpls, CLUSTER* pClustering, int clusteringSize);
	void FillClusterAtlasGPU(CLUSTER* pClustering, uint clusteringSize, uint numAVPLs);

	void FillAtlas(const std::vector<AVPL>& avpls, const int sqrt_num_ss_samples, const float& N, bool border);
	void FillAtlasGPU(const std::vector<AVPL>& avpls, const int sqrt_num_ss_samples, const float& N, bool border);

	COGLTexture2D* GetAVPLAtlas() { return m_oglAtlas.get(); };
	COGLTexture2D* GetAVPLAtlasDebug() { return m_oglAtlasDebug.get(); }
	COGLTexture2D* GetAVPLAtlasCPU() { return m_oglAtlasDebug.get(); };
		
	COGLTexture2D* GetAVPLClusterAtlas() { return m_oglClusterAtlas.get(); };
	COGLTexture2D* GetAVPLClusterAtlasCPU() { return m_oglClusterAtlasCPU.get(); };
	
	void Clear();

private:
	glm::vec4* AccessAtlas(uint x, uint y, uint tile_x, uint tile_y, glm::vec4* pAtlas);
	glm::vec4 SampleTexel(uint x, uint y, const int sqrt_num_ss_samples, const float& N, const AVPL& avpl, bool border);

	std::unique_ptr<COGLTexture2D> m_oglAtlasCPU;
	std::unique_ptr<COGLTexture2D> m_oglAtlas;
	std::unique_ptr<COGLTexture2D> m_oglAtlasDebug;
	std::unique_ptr<COGLTexture2D> m_oglClusterAtlasCPU;
	std::unique_ptr<COGLTexture2D> m_oglClusterAtlas;

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

	CMaterialBuffer* m_pMaterialBuffer;
};

#endif _C_OCTAHEDRON_ATLAS_H_