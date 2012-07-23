#include "CLightViewer.h"

#include "ShaderUtil.h"

#include "..\Macros.h"

#include "..\AVPL.h"
#include "..\CCamera.h"

#include "..\MeshResources\CMesh.h"
#include "..\MeshResources\CModel.h"

#include "..\OGLResources\COGLProgram.h"
#include "..\OGLResources\COGLUniformBuffer.h"

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

void CLightViewer::DrawLight(AVPL* avpl, CCamera* camera, COGLUniformBuffer* pUBTransform) 
{	
	glm::mat4 scale = glm::scale(0.025f, 0.025f, 0.025f);
	glm::mat4 translate = glm::translate(avpl->GetPosition());

	m_pLightModel->SetWorldTransform(translate * scale);

	COGLBindLock lockProgram(GetGLProgram(), COGL_PROGRAM_SLOT);

	glm::vec3 color;
	switch(avpl->GetBounce()){
		case 0: color = glm::vec3(0.8f, 0.8f, 0.8f); break;
		case 1: color = glm::vec3(0.8f, 0.0f, 0.0f); break;
		case 2: color = glm::vec3(0.0f, 0.8f, 0.0f); break;
		case 3: color = glm::vec3(0.0f, 0.0f, 0.8f); break;
		case 4: color = glm::vec3(0.8f, 0.8f, 0.0f); break;
		case 5: color = glm::vec3(0.8f, 0.0f, 0.8f); break;
		case 6: color = glm::vec3(0.0f, 0.8f, 0.8f); break;
		default: color = glm::vec3(0.2f, 0.2f, 0.2f); break;
	}

	glUniform3fv(uniformLightColor, 1, glm::value_ptr(color));

	m_pLightModel->Draw(camera->GetViewMatrix(), camera->GetProjectionMatrix(), pUBTransform);
}