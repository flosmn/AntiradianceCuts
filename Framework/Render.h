#ifndef RENDER_H
#define RENDER_H

#include "GL/glew.h"

#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string>

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
class CPointCloud;

class CGLUniformBuffer;
class CGLSampler;
class CGLTexture2D;
class CGLTextureBuffer;

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

	void UseAntiradiance(bool b) { if(m_UseAntiradiance == b) return; m_UseAntiradiance = b; ConfigureLighting(); ClearLighting(); }
	bool GetUseAntiradiance() { return m_UseAntiradiance; }
	void DrawAntiradiance(bool b) { if(m_DrawAntiradiance == b) return; m_DrawAntiradiance = b; ConfigureLighting(); ClearLighting();}
	bool GetDrawAntiradiance() { return m_DrawAntiradiance; }

	void SetGeoTermLimit(float f) { if(m_GeoTermLimit == f) return; m_GeoTermLimit = f; ConfigureLighting(); ClearLighting();}
	void SetBlurFactor(float f) { if(m_BlurSigma == f) return; m_BlurSigma = f; ConfigureLighting(); ClearLighting();}
	void SetNumPaths(int i) {if(m_NumPaths == i) return; m_NumPaths = i; ConfigureLighting(); ClearLighting(); }
	void SetNumPathsPerFrame(int i) {if(m_NumPathsPerFrame == i) return; m_NumPathsPerFrame = i; ConfigureLighting(); ClearLighting(); }
	void SetNumOfAdditionalAVPLs(int i) {if(m_NumAdditionalAVPLs == i) return; m_NumAdditionalAVPLs = i; CreatePlaneHammersleySamples(m_NumAdditionalAVPLs); ClearLighting(); }
	void SetRenderBounce(int i) { if(m_RenderBounce == i) return; m_RenderBounce = i; ConfigureLighting(); ClearLighting(); }
	void SetN(int n) { if(m_N == n) return; m_N = n; ConfigureLighting(); ClearLighting(); }
	void SetBias(float b) { if(m_Bias == b) return; m_Bias = b; ConfigureLighting(); ClearAccumulationBuffer(); }
	void SetUseHammersley(bool useHamm) { if(m_UseHammersley == useHamm) return; m_UseHammersley = useHamm; ConfigureLighting(); ClearAccumulationBuffer(); }

	float GetBlurFactor() { return m_BlurSigma; }

	void ConfigureLighting();

	void Stats();
	
private:
	// functions of the render phase
	void SetUpRender();
	void CreateGBuffer();
	void GatherRadiance(std::vector<Light*> path);
	void GatherRadianceWithShadowMap(std::vector<Light*> path);
	void GatherAntiradiance(std::vector<Light*> path);
	void DebugRender();
	void DrawAreaLight();
	void Normalize();
	void FinalGather();

	void CreatePlaneHammersleySamples(int i);

	void Debug(std::vector<Light*> path);
	void Debug(Light* light);
	std::string Debug(glm::vec3 v);
	std::string Debug(glm::vec4 v);

	void GatherRadianceFromLight(Light* light);
	void GatherRadianceFromLights(std::vector<Light*> lights);
	void GatherRadianceFromLightWithShadowMap(Light* light);
	void GatherAntiradianceFromLight(Light* light);
	void GatherAntiradianceFromLights(std::vector<Light*> lights);

	void FillShadowMap(Light* light);

	float CalcBlurNormalizationFactor(float sigma);
	
	glm::vec4 ColorForLight(Light* light);

	Camera *camera;
	Scene* scene;
	CShadowMap* m_pShadowMap;
	CGBuffer* m_pGBuffer;
	Postprocessing* postProcessing;
	CAccumulationBuffer* m_pAccumulationRadiance;
	CAccumulationBuffer* m_pNormalizedRadiance;
	CAccumulationBuffer* m_pAccumulationAntiradiance;
	CAccumulationBuffer* m_pNormalizedAntiradiance;
	CAccumulationBuffer* m_pFinalResult;

	CGLTextureBuffer* m_pLightBuffer;

	CPointCloud* m_pPointCloud;

	CTextureViewer* m_pTextureViewer;

	CGLUniformBuffer* m_pUBTransform;
	CGLUniformBuffer* m_pUBMaterial;
	CGLUniformBuffer* m_pUBLight;
	CGLUniformBuffer* m_pUBConfig;
	CGLUniformBuffer* m_pUBCamera;
	CGLUniformBuffer* m_pUBInfo;

	CGLTexture2D* m_pDepthBuffer;

	CProgram* m_pCreateGBufferProgram;
	CProgram* m_pCreateSMProgram;
	CProgram* m_pGatherRadianceWithSMProgram;
	CProgram* m_pGatherRadianceProgram;
	CProgram* m_pGatherAntiradianceProgram;
	CProgram* m_pFinalGatherProgram;
	CProgram* m_pNormalizeRadianceProgram;
	CProgram* m_pPointCloudProgram;

	CGLSampler* m_pGLPointSampler;
	CGLSampler* m_pGLShadowMapSampler;

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
	int m_RenderBounce;

	int m_Frame;
	int m_NumPaths;
	int m_NumPathsPerFrame;
	int m_NumAdditionalAVPLs;
	int m_CurrentPath;
	int m_N;
	float m_Bias;
	bool m_UseHammersley;

	bool m_Finished;

	float m_BlurSigma;

	int drawLightNumber;
};

#endif