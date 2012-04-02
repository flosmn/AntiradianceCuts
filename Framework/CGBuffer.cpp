#include "CGBuffer.h"

#include <glm/gtc/type_ptr.hpp>

#include "Macros.h"
#include "ShaderUtil.h"

#include "Scene.h"
#include "Light.h"
#include "Camera.h"

#include "CGLFrameBuffer.h"
#include "CGLTexture2D.h"
#include "CGLRenderBuffer.h"
#include "CGLBindLock.h"
#include "CGLProgram.h"
#include "CGLSampler.h"

#include "CProgram.h"
#include "CFullScreenQuad.h"
#include "CShadowMap.h"


CGBuffer::CGBuffer()
	: m_Width(0), m_Height(0), m_pGLFBRenderTarget(0), m_pGLRBDepthRenderBuffer(0),
	  m_pGLTTexturePositionWS(0), m_pGLTTextureNormalWS(0), m_pGLTTextureMaterials(0)
{
	m_pGLFBRenderTarget = new CGLFrameBuffer("CGBuffer.m_pGLFBRenderTarget");
	m_pGLTTexturePositionWS = new CGLTexture2D("CGBuffer.m_pGLTTexturePositionWS");
	m_pGLTTextureNormalWS = new CGLTexture2D("CGBuffer.m_pGLTTextureNormalWS");
	m_pGLTTextureMaterials = new CGLTexture2D("CGBuffer.m_pGLTTextureMaterials");
	m_pGLRBDepthRenderBuffer = new CGLRenderBuffer("CGBuffer.m_pGLRBDepthRenderBuffer");
	m_pGLPointSampler = new CGLSampler("CGBuffer.m_pGLPointSampler");

	m_pCreateGBufferProgram = new CProgram("GBuffer.m_pCreateGBufferProgram", 
		"CreateGBuffer.vert", "CreateGBuffer.frag");
}

CGBuffer::~CGBuffer()
{
	SAFE_DELETE(m_pGLRBDepthRenderBuffer);

	SAFE_DELETE(m_pGLTTexturePositionWS);
	SAFE_DELETE(m_pGLTTextureNormalWS);
	SAFE_DELETE(m_pGLTTextureMaterials);

	SAFE_DELETE(m_pGLFBRenderTarget);

	SAFE_DELETE(m_pGLPointSampler);

	SAFE_DELETE(m_pCreateGBufferProgram);
}

bool CGBuffer::Init(uint width, uint height)
{
	m_Width = width;
	m_Height = height;

	V_RET_FOF(m_pGLTTexturePositionWS->Init(m_Width, m_Height, 
		GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));

	V_RET_FOF(m_pGLTTextureNormalWS->Init(m_Width, m_Height, 
		GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));

	V_RET_FOF(m_pGLTTextureMaterials->Init(m_Width, m_Height, 
		GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));

	V_RET_FOF(m_pGLRBDepthRenderBuffer->Init(m_Width, m_Height, GL_DEPTH_COMPONENT));

	V_RET_FOF(m_pGLFBRenderTarget->Init());

	m_pGLFBRenderTarget->AttachRenderBuffer(m_pGLRBDepthRenderBuffer, GL_DEPTH_ATTACHMENT);
	m_pGLFBRenderTarget->AttachTexture2D(m_pGLTTexturePositionWS, GL_COLOR_ATTACHMENT0);
	m_pGLFBRenderTarget->AttachTexture2D(m_pGLTTextureNormalWS, GL_COLOR_ATTACHMENT1);
	m_pGLFBRenderTarget->AttachTexture2D(m_pGLTTextureMaterials, GL_COLOR_ATTACHMENT2);

	V_RET_FOF(m_pGLFBRenderTarget->CheckFrameBufferComplete());

	V_RET_FOF(m_pCreateGBufferProgram->Init());

	V_RET_FOF(m_pGLPointSampler->Init(GL_NEAREST, GL_NEAREST));

	m_pCreateGBufferProgram->BindSampler(0, m_pGLPointSampler);
	m_pCreateGBufferProgram->BindSampler(1, m_pGLPointSampler);
	m_pCreateGBufferProgram->BindSampler(2, m_pGLPointSampler);
	m_pCreateGBufferProgram->BindSampler(3, m_pGLPointSampler);

	return true;
}

void CGBuffer::Release()
{
	m_pGLFBRenderTarget->Release();

	m_pGLRBDepthRenderBuffer->Release();

	m_pGLTTexturePositionWS->Release();
	m_pGLTTextureNormalWS->Release();
	m_pGLTTextureMaterials->Release();

	m_pGLPointSampler->Release();

	m_pCreateGBufferProgram->Release();
}