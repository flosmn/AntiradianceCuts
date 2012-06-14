#ifndef _C_FULL_SCREEN_QUAD_H_
#define _C_FULL_SCREEN_QUAD_H_

class COGLVertexArray;
class CMesh;

class CFullScreenQuad
{
public:
	CFullScreenQuad();
	~CFullScreenQuad();

	bool Init();
	void Release();

	void Draw();

private:
	COGLVertexArray* m_pGLVARenderData;
	CMesh* m_pFullScreenQuadMesh;
};

#endif // _C_FULL_SCREEN_QUAD_H_