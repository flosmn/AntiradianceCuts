#ifndef RENDER_H
#define RENDER_H

#include "GL/glew.h"

#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string>

class Scene;
class Camera;
class CShadowMap;
class CPostprocess;
class AVPL;

class CAccumulationBuffer;
class CGBuffer;
class CTimer;
class CProgram;
class CLightViewer;
class CTextureViewer;
class CFullScreenQuad;
class CPointCloud;
class CExport;
class CRenderTarget;

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
	
	void ClearAccumulationBuffer();
	void ClearLighting();

	void PrintCameraConfig();
		
	void UseAntiradiance(bool b) { if(m_UseAntiradiance == b) return; m_UseAntiradiance = b; ConfigureLighting(); ClearLighting(); }
	bool GetUseAntiradiance() { return m_UseAntiradiance; }
	
	void SetGeoTermLimit(float f) { if(m_GeoTermLimit == f) return; m_GeoTermLimit = f; ConfigureLighting(); ClearLighting();}
	void SetNumPaths(int i) {if(m_NumPaths == i) return; m_NumPaths = i; ConfigureLighting(); ClearLighting(); }
	void SetNumPathsPerFrame(int i) {if(m_NumPathsPerFrame == i) return; m_NumPathsPerFrame = i; ConfigureLighting(); ClearLighting(); }
	void SetNumOfAdditionalAVPLs(int i) { if(m_NumAdditionalAVPLs == i) return; m_NumAdditionalAVPLs = i; CreatePlaneHammersleySamples(m_NumAdditionalAVPLs); ClearLighting(); }
	void SetRenderBounce(int i) { if(m_RenderBounce == i) return; m_RenderBounce = i; ConfigureLighting(); ClearLighting(); }
	void SetN(int n) { if(m_N == n) return; m_N = n; ConfigureLighting(); ClearLighting(); }
	void SetBias(float b) { if(m_Bias == b) return; m_Bias = b; ConfigureLighting(); ClearAccumulationBuffer(); }
	void SetUseHammersley(bool useHamm) { if(m_UseHammersley == useHamm) return; m_UseHammersley = useHamm; ConfigureLighting(); ClearAccumulationBuffer(); }
	void SetToneMapping(bool b) { m_ToneMapping = b; }

	void SetGamma(float gamma);
	void SetExposure(float exposure);

	void ConfigureLighting();

	void Stats();

	void Export();
	
private:
	// functions of the render phase
	void SetUpRender();
	void CreateGBuffer();
	
	void Gather(std::vector<AVPL*> path);
	void Normalize();
	void Shade();	
	
	void GatherRadianceWithShadowMap(std::vector<AVPL*> path);	
	void DebugRender();
	void DrawAreaLight();
	
	void CreatePlaneHammersleySamples(int i);
		
	void GatherRadianceFromLightWithShadowMap(AVPL* avpl);
	void FillShadowMap(AVPL* avpl);
	void DrawLights(std::vector<AVPL*> avpl);
	
	glm::vec4 ColorForLight(AVPL* light);
	
	void ExportPartialResult();

	Camera *camera;
	Scene* scene;
	CShadowMap* m_pShadowMap;
	CGBuffer* m_pGBuffer;
	
	CRenderTarget* m_pGatherRenderTarget;
	CRenderTarget* m_pNormalizeRenderTarget;
	CRenderTarget* m_pShadeRenderTarget;
		
	CProgram* m_pGatherProgram;
	CProgram* m_pNormalizeProgram;
	CProgram* m_pShadeProgram;
		
	CRenderTarget* m_pLightDebugRenderTarget;
	CRenderTarget* m_pPostProcessRenderTarget;
	CPostprocess* m_pPostProcess;

	CGLTextureBuffer* m_pLightBuffer;

	CPointCloud* m_pPointCloud;
	CExport* m_Export;

	CTimer* m_Timer;
	CTextureViewer* m_pTextureViewer;

	CGLUniformBuffer* m_pUBTransform;
	CGLUniformBuffer* m_pUBMaterial;
	CGLUniformBuffer* m_pUBLight;
	CGLUniformBuffer* m_pUBConfig;
	CGLUniformBuffer* m_pUBCamera;
	CGLUniformBuffer* m_pUBInfo;
	CGLUniformBuffer* m_pUBAreaLight;

	CGLTexture2D* m_pDepthBuffer;

	CProgram* m_pCreateGBufferProgram;
	CProgram* m_pCreateSMProgram;
	CProgram* m_pGatherRadianceWithSMProgram;
	CProgram* m_pPointCloudProgram;
	CProgram* m_pAreaLightProgram;

	CGLSampler* m_pGLPointSampler;
	CGLSampler* m_pGLShadowMapSampler;

	CFullScreenQuad* m_pFullScreenQuad;

	bool drawTexture;
	bool drawLight;
	bool m_UseAntiradiance;
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
	bool m_PartialSum;
	bool m_ToneMapping;

	bool m_Finished;

	time_t m_StartTime;
};

#endif