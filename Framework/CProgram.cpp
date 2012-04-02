#include "CProgram.h"

#include "Macros.h"

#include "CGLProgram.h"
#include "CGLUniformBuffer.h"
#include "CGLSampler.h"


CProgram::CProgram(const std::string debugName, const std::string VS, const std::string FS)
{
	m_pGLProgram = new CGLProgram(debugName, VS, FS);
}

CProgram::~CProgram()
{
	SAFE_DELETE(m_pGLProgram);
}

bool CProgram::Init()
{
	V_RET_FOF(m_pGLProgram->Init());

	return true;
}

void CProgram::Release()
{
	m_pGLProgram->Release();
}

void CProgram::BindUniformBuffer(CGLUniformBuffer* pGLUniformBuffer, std::string strUniformBlockName)
{
	m_pGLProgram->BindUniformBuffer(pGLUniformBuffer, strUniformBlockName);
}

void CProgram::BindSampler(uint samplerSlot, CGLSampler* pSampler)
{
	m_pGLProgram->BindSampler(samplerSlot, pSampler);
}

CGLProgram* CProgram::GetGLProgram()
{
	return m_pGLProgram;
}
