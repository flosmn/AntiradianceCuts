#ifndef _C_POINT_CLOUD_H_
#define _C_POINT_CLOUD_H_

#include "glm/glm.hpp"

typedef unsigned int uint;

class COGLVertexArray;

class CPointCloud
{
public:
	CPointCloud();
	~CPointCloud();

	bool Init();
	void Release();
	
	void Draw(glm::vec4* positionData, glm::vec4* colorData, uint nPoints);
private:
	COGLVertexArray* m_pVertexArray;
	uint m_nPoints;
};

#endif // _C_POINT_CLOUD_H_