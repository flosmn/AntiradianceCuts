#include "CPostprocess.h"

#include "Macros.h"

#include "CProgram.h"
#include "CRenderTarget.h"
#include "CRenderTargetLock.h"

#include "CConfigManager.h"

#include "Utils\ShaderUtil.h"

#include "OGLResources\COGLProgram.h"
#include "OGLResources\COGLTexture2D.h"
#include "OGLResources\COGLSampler.h"
#include "OGLResources\COGLUniformBuffer.h"

#include "MeshResources\CFullScreenQuad.h"

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
	m_pPointSampler = new COGLSampler("CPostProcess.m_pPointSampler");
	m_pUniformBuffer = new COGLUniformBuffer("CPostProcess.m_pUniformBuffer");
}

CPostprocess::~CPostprocess() 
{
	SAFE_DELETE(m_pFullScreenQuad);
	SAFE_DELETE(m_pPostProcessProgram);
	SAFE_DELETE(m_pPointSampler);
	SAFE_DELETE(m_pUniformBuffer);
}

bool CPostprocess::Init(CConfigManager* pConfigManager)
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

	m_pConfigManager = pConfigManager;

	return true;
}

void CPostprocess::Release()
{
	m_pFullScreenQuad->Release();
	m_pPostProcessProgram->Release();
	m_pUniformBuffer->Release();
	m_pPointSampler->Release();
}

void CPostprocess::Postprocess(COGLTexture2D* pTexture, CRenderTarget* pTarget)
{
	UpdateUniformBuffer();
	
	CRenderTargetLock lock(pTarget);

	COGLBindLock lockProgram(m_pPostProcessProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lockTexture(pTexture, COGL_TEXTURE0_SLOT);
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	m_pFullScreenQuad->Draw();
}

void CPostprocess::UpdateUniformBuffer()
{
	float gamma = 1.f;
	if(m_pConfigManager->GetConfVars()->UseGammaCorrection)
		gamma = m_pConfigManager->GetConfVars()->Gamma;

	float exposure = 1.f;
	if(m_pConfigManager->GetConfVars()->UseToneMapping)
		exposure = m_pConfigManager->GetConfVars()->Exposure;
	
	POST_PROCESS pp;
	pp.one_over_gamma = 1.f / gamma;
	pp.exposure = exposure;
	m_pUniformBuffer->UpdateData(&pp);
}
