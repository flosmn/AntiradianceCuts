#ifndef _C_LIGHT_VIEWER_H_
#define _C_LIGHT_VIEWER_H_

#include "GL/glew.h"

#include "ShaderUtil.h"

#include "..\AVPL.h"
#include "..\CCamera.h"

#include "..\MeshResources\CMesh.h"
#include "..\MeshResources\CModel.h"

#include "..\OGLResources\COGLProgram.h"
#include "..\OGLResources\COGLUniformBuffer.h"

#include "..\CProgram.h"

#include <memory>

class CCamera;
class AVPL;

class CModel;

class COGLUniformBuffer;

class CLightViewer : public CProgram
{
public:
	CLightViewer() : CProgram("CLightViewer", "Shaders\\DrawLight.vert", "Shaders\\DrawLight.frag")
	{
		m_lightModel.reset(new CModel(new CCubeMesh));

		uniformLightColor = glGetUniformLocation(GetGLProgram()->GetResourceIdentifier(), 
			"lightColor");
	}

	~CLightViewer() {}

	void DrawLight(AVPL* avpl, CCamera* camera, COGLUniformBuffer* pUBTransform) 
	{	
		glm::mat4 scale = glm::scale(0.025f, 0.025f, 0.025f);
		glm::mat4 translate = glm::translate(avpl->GetPosition());

		m_lightModel->SetWorldTransform(translate * scale);

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

		m_lightModel->Draw(camera->GetViewMatrix(), camera->GetProjectionMatrix(), pUBTransform);
	}

private:
	std::unique_ptr<CModel> m_lightModel;
	GLuint uniformLightColor;
};

#endif // _C_LIGHT_VIEWER_H_
