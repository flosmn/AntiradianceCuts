#ifndef _C_CUBE_SHADOW_MAP
#define _C_CUBE_SHADOW_MAP

class CProgram;
class COGLUniformBuffer;
class COGLTexture2D;
class Scene;

#include <gl\glew.h>
#include <glm\glm.hpp>

#include "AVPL.h"

#include <memory>

class CCubeShadowMap
{
public:
	CCubeShadowMap(COGLUniformBuffer* pUBTransform);
	~CCubeShadowMap();

	void Create(Scene* pScene, AVPL const& avpl, COGLUniformBuffer* pUBTransform);

	COGLTexture2D* GetCubeMapFace(int i);

private:
	GLuint m_CubeMap;
	GLuint m_FrameBuffer;

	std::unique_ptr<CProgram> m_program;
	std::unique_ptr<COGLTexture2D> m_cubeMapFace;

	glm::mat4 m_Proj;
	glm::vec3 m_Directions[6];
	glm::vec3 m_Ups[6];
};

#endif _C_CUBE_SHADOW_MAP