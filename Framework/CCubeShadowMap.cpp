#include "CCubeShadowMap.h"

#include "OGLResources\COGLResource.h"
#include "OGLResources\COGLBindLock.h"
#include "OGLResources\COGLBindSlot.h"
#include "OGLResources\COGLProgram.h"
#include "OGLResources\COGLTexture2D.h"
#include "OGLResources\COGLUniformBuffer.h"

#include "MeshResources\CFullScreenQuad.h"

#include "Scene.h"
#include "CProgram.h"
#include "Defines.h"
#include "Macros.h"
#include "CCamera.h"

#include <glm/gtx/transform.hpp>

#include <iostream>

CCubeShadowMap::CCubeShadowMap(COGLUniformBuffer* pUBTransform)
{
	m_program.reset(new CProgram( 
		"Shaders/CreateCubeShadowMap.vert", 
		"Shaders/CreateCubeShadowMap.frag"));
	m_cubeMapFace.reset(new COGLTexture2D(512, 512, GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));

	m_Proj = glm::perspective(45.f, 1.f, 0.01f, 20000.f);
	m_Directions[0] = glm::vec3(1.f, 0.f, 0.f);
	m_Directions[1] = glm::vec3(-1.f, 0.f, 0.f);
	m_Directions[2] = glm::vec3(0.f, 1.f, 0.f);
	m_Directions[3] = glm::vec3(0.f, -1.f, 0.f);
	m_Directions[4] = glm::vec3(0.f, 0.f, 1.f);
	m_Directions[5] = glm::vec3(0.f, 0.f, -1.f);

	m_Ups[0] = glm::vec3(0.f, 1.f, 0.f);
	m_Ups[1] = glm::vec3(0.f, 1.f, 0.f);
	m_Ups[2] = glm::vec3(1.f, 0.f, 0.f);
	m_Ups[3] = glm::vec3(1.f, 0.f, 0.f);
	m_Ups[4] = glm::vec3(0.f, 1.f, 0.f);
	m_Ups[5] = glm::vec3(0.f, 1.f, 0.f);

	int face;
	GLenum status;
	
	glActiveTexture(GL_TEXTURE1);
	glGenTextures(1, &m_CubeMap);
	glBindTexture(GL_TEXTURE_CUBE_MAP, m_CubeMap);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	
	for (face = 0; face < 6; face++) {
	    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA32F, 512, 512, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	
	GLuint rboId;
	glGenRenderbuffers(1, &rboId);
	glBindRenderbuffer(GL_RENDERBUFFER, rboId);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 512, 512);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	
	// framebuffer object
	glGenFramebuffers(1, &m_FrameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBuffer);
	
	for (face = 0; face < 6; face++) {
	    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + face, GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, m_CubeMap, 0);
	}
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboId);

	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	
	if(status != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete" << std::endl;
	
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
	
	m_program->BindUniformBuffer(pUBTransform, "transform");
}

CCubeShadowMap::~CCubeShadowMap()
{
	glDeleteTextures(1, &m_CubeMap);
	glDeleteFramebuffers(1, &m_FrameBuffer);
}

void CCubeShadowMap::Create(Scene* pScene, const AVPL& avpl, COGLUniformBuffer* pUBTransform)
{
	glEnable(GL_DEPTH_TEST);
	
	// prevent surface acne
	//glEnable(GL_POLYGON_OFFSET_FILL);
	//glPolygonOffset(1.1f, 4.0f);
	
	COGLBindLock lockProgram(m_program->GetGLProgram(), COGL_PROGRAM_SLOT);
	glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBuffer);
	
	glViewport(0, 0, 512, 512);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	
	for(int face = 0; face < 6; ++face)
	{
		GLenum buffer[1] = {GL_COLOR_ATTACHMENT0 + face};
		
		glDrawBuffers(1, buffer);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		glm::vec3 pos = pScene->GetCamera()->GetPosition();
		glm::mat4 viewMatrix = glm::lookAt(pos, pos + m_Directions[face], m_Ups[face]);
		pScene->DrawScene(viewMatrix, m_Proj, pUBTransform);
	}

	glViewport(0, 0, pScene->GetCamera()->GetWidth(), pScene->GetCamera()->GetHeight());
	//glDisable(GL_POLYGON_OFFSET_FILL);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDrawBuffer(GL_BACK);
}

COGLTexture2D* CCubeShadowMap::GetCubeMapFace(int i)
{
	glm::vec4* pData = new glm::vec4[512 * 512];
	memset(pData, 0, 512 * 512 * sizeof(glm::vec4));

	glBindTexture(GL_TEXTURE_CUBE_MAP, m_CubeMap);
	glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA, GL_FLOAT, pData);
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

	m_cubeMapFace->SetPixelData(pData);
	delete [] pData;
	return m_cubeMapFace.get();
}