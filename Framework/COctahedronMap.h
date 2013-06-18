#ifndef _C_OCTAHEDRON_MAP_H_
#define _C_OCTAHEDRON_MAP_H_

#include <glm/glm.hpp>

#include <memory>

class COGLTexture2D;

typedef unsigned int uint;

class COctahedronMap
{
public:
	COctahedronMap(uint dimension);
	~COctahedronMap();

	void FillWithDebugData();

	COGLTexture2D* GetTexture() { return m_texture.get(); }

private:
	std::unique_ptr<COGLTexture2D> m_texture;

	glm::vec4* AccessMap(uint x, uint y, glm::vec4* pMap);

	uint m_Dimension;
};

#endif _C_OCTAHEDRON_MAP_H_