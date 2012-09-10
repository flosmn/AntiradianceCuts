#ifndef _C_CUBE_SHADOW_MAP
#define _C_CUBE_SHADOW_MAP

class CProgram;
class COGLUniformBuffer;
class COGLTexture2D;
class Scene;

#include <gl\glew.h>
#include <glm\glm.hpp>

#include "AVPL.h"

class CCubeShadowMap
{
public:
	CCubeShadowMap();
	~CCubeShadowMap();

	bool Init(COGLUniformBuffer* m_pUBTransform);
	void Release();

	void Create(Scene* pScene, const AVPL& avpl, COGLUniformBuffer* pUBTransform);

	COGLTexture2D* GetCubeMapFace(int i);

private:
	GLuint m_CubeMap;
	GLuint m_FrameBuffer;

	CProgram* m_pCreateProgram;
	COGLTexture2D* m_pCubeMapFace;
	COGLUniformBuffer* m_pUBTranforms;

	glm::mat4 m_Proj;
	glm::vec3 m_Directions[6];
	glm::vec3 m_Ups[6];
};

#endif _C_CUBE_SHADOW_MAP