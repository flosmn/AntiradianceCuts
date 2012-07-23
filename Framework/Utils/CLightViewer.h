#ifndef _C_LIGHT_VIEWER_H_
#define _C_LIGHT_VIEWER_H_

#include "GL/glew.h"

#include "..\CProgram.h"

class CCamera;
class AVPL;

class CModel;

class COGLUniformBuffer;

class CLightViewer : public CProgram
{
public:
	CLightViewer();
	~CLightViewer();

	bool Init();
	void Release();
	void DrawLight(AVPL* avpl, CCamera* camera, COGLUniformBuffer* pUBTransform);

private:
	CModel* m_pLightModel;

	GLuint uniformLightColor;
};

#endif // _C_LIGHT_VIEWER_H_