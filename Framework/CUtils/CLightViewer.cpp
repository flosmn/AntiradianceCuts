#include "CLightViewer.h"

#include "ShaderUtil.h"

#include "..\Macros.h"

#include "..\Light.h"
#include "..\Camera.h"

#include "..\CMeshResources\CMesh.h"
#include "..\CMeshResources\CModel.h"

#include "..\CGLResources\CGLProgram.h"
#include "..\CGLResources\CGLUniformBuffer.h"

CLightViewer::CLightViewer()
	: CProgram("CLightViewer", "Shaders\\DrawLight.vert", "Shaders\\DrawLight.frag")
{
	m_pLightModel = new CModel();
}

CLightViewer::~CLightViewer()
{
	CProgram::~CProgram();

	SAFE_DELETE(m_pLightModel);
}

bool CLightViewer::Init()
{
	V_RET_FOF(CProgram::Init());
	
	uniformLightColor = glGetUniformLocation(GetGLProgram()->GetResourceIdentifier(), 
		"lightColor");
	
	V_RET_FOF(m_pLightModel->Init(new CCubeMesh()));

	return true;
}

void CLightViewer::Release()
{
	CProgram::Release();

	m_pLightModel->Release();
}

void CLightViewer::DrawLight(Light* light, Camera* camera, CGLUniformBuffer* pUBTransform) 
{
	
	glm::mat4 scale = glm::scale(0.01f, 0.01f, 0.01f);
	glm::mat4 translate = glm::translate(light->GetPosition());

	m_pLightModel->SetWorldTransform(translate * scale);

	CGLBindLock lockProgram(GetGLProgram(), CGL_PROGRAM_SLOT);

	glUniform3fv(uniformLightColor, 1, glm::value_ptr(light->GetDebugColor()));

	m_pLightModel->Draw(camera->GetViewMatrix(), camera->GetProjectionMatrix(), pUBTransform);
}