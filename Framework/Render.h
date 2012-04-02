#ifndef RENDER_H
#define RENDER_H

#include "GL/glew.h"

#include <vector>

class Scene;
class Camera;
class CShadowMap;
class Postprocessing;
class Light;

class CAccumulationBuffer;
class CGBuffer;
class CTimer;
class CProgram;
class CLightViewer;
class CTextureViewer;
class CFullScreenQuad;

class CGLUniformBuffer;
class CGLSampler;

class Renderer {
public:
	Renderer(Camera* _camera);
	~Renderer();
	
	bool Init();
	void Release();

	void Render();

	void WindowChanged();

	void SetDrawTexture(bool b) { drawTexture = b; }
	bool GetDrawTexture() { return drawTexture; } 

	void SetDrawLight(bool b) { drawLight = b; }
	bool GetDrawLight() { return drawLight; } 

	void SetPrintTimes(bool b) { m_PrintTimes = b; }

	void ClearAccumulationBuffer();
	void ClearLighting();

	void PrintCameraConfig();

	void SetDrawLightNumber(int i) { drawLightNumber = i; }

	void DrawOnlyDirectLight(bool b) { m_DrawOnlyDirectLight = b; }
	void DrawOnlyIndirectLight(bool b) { m_DrawOnlyIndirectLight = b; }
	bool GetDrawOnlyDirectLight() { return m_DrawOnlyDirectLight; }
	bool GetDrawOnlyIndirectLight() { return m_DrawOnlyIndirectLight; }

	void UseAntiradiance(bool b) { if(m_UseAntiradiance == b) return; m_UseAntiradiance = b; ConfigureLighting(); ClearAccumulationBuffer(); }
	bool GetUseAntiradiance() { return m_UseAntiradiance; }
	void DrawAntiradiance(bool b) { if(m_DrawAntiradiance == b) return; m_DrawAntiradiance = b; ConfigureLighting(); ClearAccumulationBuffer();}
	bool GetDrawAntiradiance() { return m_DrawAntiradiance; }

	void SetGeoTermLimit(float f) { if(m_GeoTermLimit == f) return; m_GeoTermLimit = f; ConfigureLighting(); ClearAccumulationBuffer();}
	void SetBlurFactor(float f) { if(m_BlurSigma == f) return; m_BlurSigma = f; ConfigureLighting(); ClearAccumulationBuffer();}
	
	float GetBlurFactor() { return m_BlurSigma; }

	void Stats();

private:
	// functions of the render phase
	void SetUpRender();
	void CreateGBuffer();
	void GatherRadiance();
	void GatherAntiradiance();
	void PostProcess();
	void DebugRender();
	void DrawAreaLight();
	void Normalize();
	void FinalGather();

	void GatherRadianceFromLight(Light* light);
	void GatherAntiradianceFromLight(Light* light);

	void FillShadowMap(Light* light);

	void ConfigureLighting();

	float CalcBlurNormalizationFactor(float sigma);
	
	Camera *camera;
	Scene* scene;
	CShadowMap* m_pShadowMap;
	CGBuffer* m_pGBuffer;
	Postprocessing* postProcessing;
	CAccumulationBuffer* m_pAccumulationRadiance;
	CAccumulationBuffer* m_pNormalizedRadiance;
	CAccumulationBuffer* m_pAccumulationAntiradiance;
	CAccumulationBuffer* m_pNormalizedAntiradiance;
	CTimer* m_Timer;
	CLightViewer* m_pLightViewer;
	CTextureViewer* m_pTextureViewer;

	CGLUniformBuffer* m_pUBTransform;
	CGLUniformBuffer* m_pUBMaterial;
	CGLUniformBuffer* m_pUBLight;
	CGLUniformBuffer* m_pUBConfig;
	CGLUniformBuffer* m_pUBCamera;

	CProgram* m_pCreateGBufferProgram;
	CProgram* m_pCreateSMProgram;
	CProgram* m_pGatherRadianceProgram;
	CProgram* m_pGatherAntiradianceProgram;
	CProgram* m_pFinalGatherProgram;
	CProgram* m_pNormalizeRadianceProgram;

	CGLSampler* m_pGLPointSampler;

	CFullScreenQuad* m_pFullScreenQuad;

	bool drawTexture;
	bool drawLight;
	bool m_PrintTimes;
	bool m_DrawOnlyIndirectLight;
	bool m_DrawOnlyDirectLight;
	bool m_UseAntiradiance;
	bool m_DrawAntiradiance;
	float m_GeoTermLimit;
	float m_CosBlurFactor;

	int m_Frame;
	int m_PathRadiance;
	int m_PathAntiradiance;
	int m_MaxPaths;
	int m_NumPaths;

	bool m_Finished;

	float m_BlurSigma;

	int drawLightNumber;
};

#endif