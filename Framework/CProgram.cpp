#include "CProgram.h"

#include "Macros.h"

#include "OGLResources\COGLProgram.h"
#include "OGLResources\COGLUniformBuffer.h"
#include "OGLResources\COGLSampler.h"

CProgram::CProgram(const std::string debugName, const std::string VS, const std::string FS)
{
	m_pGLProgram = new COGLProgram(debugName, VS, FS);
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

void CProgram::BindUniformBuffer(COGLUniformBuffer* pGLUniformBuffer, std::string strUniformBlockName)
{
	m_pGLProgram->BindUniformBuffer(pGLUniformBuffer, strUniformBlockName);
}

void CProgram::BindSampler(uint samplerSlot, COGLSampler* pSampler)
{
	m_pGLProgram->BindSampler(samplerSlot, pSampler);
}

COGLProgram* CProgram::GetGLProgram()
{
	return m_pGLProgram;
}
