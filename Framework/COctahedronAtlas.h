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
	
	void FillAtlas(std::vector<AVPL*> avpls, const int sqrt_num_ss_samples, const float& N, bool border);
	void FillAtlasGPU(AVPL_BUFFER* pBufferData, uint numAVPLs, const int sqrt_num_ss_samples, const float& N, bool border);
	
	COGLTexture2D* GetTexture();
	COGLTexture2D* GetRefTexture();

	void Clear();

private:
	glm::vec4* AccessAtlas(uint x, uint y, uint tile_x, uint tile_y, glm::vec4* pAtlas);
	glm::vec4 SampleTexel(uint x, uint y, const int sqrt_num_ss_samples, const float& N, AVPL* avpl, bool border);

	COGLTexture2D* m_pOGLAtlasRef;
	COGLTexture2D* m_pOGLAtlas;
	COCLTexture2D* m_pOCLAtlas;
	uint m_AtlasDim;
	uint m_TileDim;

	COCLContext* m_pOCLContext;
	COCLProgram* m_pOCLProgram;
	COCLKernel* m_pOCLKernel;
	COCLKernel* m_pOCLKernelClear;
	COCLBuffer* m_pAvplBuffer;
	COCLBuffer* m_pLocalBuffer;
};

#endif _C_OCTAHEDRON_ATLAS_H_