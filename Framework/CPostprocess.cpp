#include "CPostprocess.h"

#include "Macros.h"

#include "CProgram.h"
#include "CRenderTarget.h"
#include "CRenderTargetLock.h"

#include "CUtils\ShaderUtil.h"

#include "CGLResources\CGLProgram.h"
#include "CGLResources\CGLTexture2D.h"
#include "CGLResources\CGLSampler.h"
#include "CGLResources\CGLUniformBuffer.h"

#include "CMeshResources\CFullScreenQuad.h"

struct POST_PROCESS
{
	float one_over_gamma;
	float exposure;
};

CPostprocess::CPostprocess() 
{
	m_Gamma = 2.2f;
	m_Exposure = 1.0f;

	m_pFullScreenQuad = new CFullScreenQuad();
	m_pPostProcessProgram = new CProgram("CPostProcess", "Shaders\\PostProcess.vert", "Shaders\\PostProcess.frag");
	m_pPointSampler = new CGLSampler("CPostProcess.m_pPointSampler");
	m_pUniformBuffer = new CGLUniformBuffer("CPostProcess.m_pUniformBuffer");
}

CPostprocess::~CPostprocess() 
{
	SAFE_DELETE(m_pFullScreenQuad);
	SAFE_DELETE(m_pPostProcessProgram);
	SAFE_DELETE(m_pPointSampler);
	SAFE_DELETE(m_pUniformBuffer);
}

bool CPostprocess::Init()
{
	POST_PROCESS pp;
	pp.one_over_gamma = 1.f / m_Gamma;
	pp.exposure = m_Exposure;
	
	V_RET_FOF(m_pFullScreenQuad->Init());	
	V_RET_FOF(m_pPostProcessProgram->Init());
	V_RET_FOF(m_pUniformBuffer->Init(sizeof(POST_PROCESS), &pp, GL_STATIC_DRAW));
	V_RET_FOF(m_pPointSampler->Init(GL_NEAREST, GL_NEAREST, GL_REPEAT, GL_REPEAT));
	
	m_pPostProcessProgram->BindSampler(0, m_pPointSampler);
	m_pPostProcessProgram->BindUniformBuffer(m_pUniformBuffer, "postprocess");

	return true;
}

void CPostprocess::Release()
{
	m_pFullScreenQuad->Release();
	m_pPostProcessProgram->Release();
	m_pUniformBuffer->Release();
	m_pPointSampler->Release();
}

void CPostprocess::Postprocess(CGLTexture2D* pTexture, CRenderTarget* pTarget)
{
	CRenderTargetLock lock(pTarget);

	CGLBindLock lockProgram(m_pPostProcessProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	CGLBindLock lockTexture(pTexture, CGL_TEXTURE0_SLOT);
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	m_pFullScreenQuad->Draw();
}

void CPostprocess::UpdateUniformBuffer()
{
	POST_PROCESS pp;
	pp.one_over_gamma = 1.f / m_Gamma;
	pp.exposure = m_Exposure;
	m_pUniformBuffer->UpdateData(&pp);
}
