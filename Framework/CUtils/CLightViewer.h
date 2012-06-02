#ifndef _C_LIGHT_VIEWER_H_
#define _C_LIGHT_VIEWER_H_

#include "GL/glew.h"

#include "..\CProgram.h"

class Camera;
class AVPL;

class CModel;

class CGLUniformBuffer;

class CLightViewer : public CProgram
{
public:
	CLightViewer();
	~CLightViewer();

	bool Init();
	void Release();
	void DrawLight(AVPL* avpl, Camera* camera, CGLUniformBuffer* pUBTransform);

private:
	CModel* m_pLightModel;

	GLuint uniformLightColor;
};

#endif // _C_LIGHT_VIEWER_H_