#ifndef _C_OCTAHEDRON_MAP_H_
#define _C_OCTAHEDRON_MAP_H_

#include <glm/glm.hpp>

class COGLTexture2D;

typedef unsigned int uint;

class COctahedronMap
{
public:
	COctahedronMap();
	~COctahedronMap();

	bool Init(uint dimension);
	void Release();

	void FillWithDebugData();

	COGLTexture2D* GetTexture();

private:
	COGLTexture2D* m_pTexture;

	glm::vec4* AccessMap(uint x, uint y, glm::vec4* pMap);

	uint m_Dimension;
};

#endif _C_OCTAHEDRON_MAP_H_