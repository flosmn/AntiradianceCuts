#ifndef _C_OCTAHEDRON_ATLAS_H_
#define _C_OCTAHEDRON_ATLAS_H_

#include <glm/glm.hpp>

typedef unsigned int uint;

class COGLTexture2D;
class AVPL;

#include <vector>

class COctahedronAtlas
{
public:
	COctahedronAtlas();
	~COctahedronAtlas();

	bool Init(uint atlasDim, uint tileDim);
	void Release();
	
	void FillAtlas(std::vector<AVPL*> avpls, const int sqrt_num_ss_samples, const float& N, bool border);
	
	COGLTexture2D* GetTexture();

private:
	glm::vec4* AccessAtlas(uint x, uint y, uint tile_x, uint tile_y, glm::vec4* pAtlas);

	/*
		samples texel with ss^2 samples
	*/
	glm::vec4 SampleTexel(uint x, uint y, const int sqrt_num_ss_samples, const float& N, AVPL* avpl);

	COGLTexture2D* m_pAtlas;
	uint m_AtlasDim;
	uint m_TileDim;
};

#endif _C_OCTAHEDRON_ATLAS_H_