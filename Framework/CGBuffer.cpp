#include "CGBuffer.h"

#include <glm/gtc/type_ptr.hpp>

#include "Macros.h"

#include "Scene.h"
#include "CProgram.h"
#include "CShadowMap.h"

#include "OGLResources\COGLFrameBuffer.h"
#include "OGLResources\COGLTexture2D.h"
#include "OGLResources\COGLBindLock.h"
#include "OGLResources\COGLProgram.h"
#include "OGLResources\COGLSampler.h"

#include "Utils\ShaderUtil.h"

#include "MeshResources\CFullScreenQuad.h"


CGBuffer::CGBuffer(uint width, uint height, COGLTexture2D* pDepthBuffer)
	: m_Width(width), m_Height(height)
{
	m_positionWS.reset(new COGLTexture2D(m_Width, m_Height, 
		GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));

	m_normalWS.reset(new COGLTexture2D(m_Width, m_Height, 
		GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));

	m_materials.reset(new COGLTexture2D(m_Width, m_Height, 
		GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));

	m_renderTarget.reset(new COGLFrameBuffer());

	m_renderTarget->AttachTexture2D(pDepthBuffer, GL_DEPTH_ATTACHMENT);
	m_renderTarget->AttachTexture2D(m_positionWS.get(), GL_COLOR_ATTACHMENT0);
	m_renderTarget->AttachTexture2D(m_normalWS.get(), GL_COLOR_ATTACHMENT1);
	m_renderTarget->AttachTexture2D(m_materials.get(), GL_COLOR_ATTACHMENT2);

	m_renderTarget->CheckFrameBufferComplete();

	m_program.reset(new CProgram("Shaders\\CreateGBuffer.vert", "Shaders\\CreateGBuffer.frag"));

	m_pointSampler.reset(new COGLSampler(GL_NEAREST, GL_NEAREST, GL_REPEAT, GL_REPEAT));

	m_program->BindSampler(0, m_pointSampler.get());
	m_program->BindSampler(1, m_pointSampler.get());
	m_program->BindSampler(2, m_pointSampler.get());
	m_program->BindSampler(3, m_pointSampler.get());
}

CGBuffer::~CGBuffer()
{
}
