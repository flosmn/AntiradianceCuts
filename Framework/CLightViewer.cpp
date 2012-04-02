#include "CLightViewer.h"

#include "Macros.h"
#include "ShaderUtil.h"

#include "Light.h"
#include "Camera.h"

#include "CMesh.h"
#include "CModel.h"

#include "CGLProgram.h"
#include "CGLUniformBuffer.h"

CLightViewer::CLightViewer()
	: CProgram("CLightViewer", "DrawLight.vert", "DrawLight.frag")
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
	
	glm::mat4 scale = glm::scale(0.05f, 0.05f, 0.05f);
	glm::mat4 translate = glm::translate(light->GetPosition());

	m_pLightModel->SetWorldTransform(translate * scale);

	CGLBindLock lockProgram(GetGLProgram(), CGL_PROGRAM_SLOT);

	glUniform3fv(uniformLightColor, 1, glm::value_ptr(light->GetDebugColor()));

	m_pLightModel->Draw(camera->GetViewMatrix(), camera->GetProjectionMatrix(), pUBTransform);
}