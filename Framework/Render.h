#ifndef RENDER_H
#define RENDER_H

#include "GL/glew.h"

#include <glm/gtc/type_ptr.hpp>

#include "Avpl.h"

#include <vector>
#include <string>
#include <memory>

#include "CudaResources/CudaContext.hpp"

class Scene;
class CCamera;
class CConfigManager;
class CShadowMap;
class Postprocess;
class CImagePlane;
class CPathTracingIntegrator;
class AvplShooter;

class PointCloud;
class AABBCloud;
class AvplBvh;
class Sphere;
class SceneProbe;

class CudaGather;

class CGBuffer;
class CProgram;
class CLightViewer;
class CExport;
class CRenderTarget;
class CModel;
class CTimer;
class CRenderTarget;
class CExperimentData;

class TextureViewer;
class FullScreenQuad;

class COGLUniformBuffer;
class COGLSampler;
class COGLTexture2D;
class COGLTextureBuffer;
class COGLContext;
class COGLRenderTarget;
class Renderer {
public:
	Renderer(CCamera* camera, COGLContext* glContext, CConfigManager* confManager);
	~Renderer();

	void Render();
	void CancelRender();
	void WindowChanged();

	void IssueClearLighting();
	void IssueClearAccumulationBuffer();
	
	void PrintCameraConfig();

	void UpdateAreaLights();
			
	void UpdateUniformBuffers();
	void BindSamplers();

	void Stats();

	void Export();

	void shootSceneProbe(int x, int y);
	void NewDebugLights();
	
	void ProfileFrame() { m_ProfileFrame = true; }

	void UpdateBvhDebug();
	void RebuildBvh();
private:
	// functions of the render phase
	void SetUpRender();
	void CreateGBuffer();

	void drawScene(glm::mat4 const& view, glm::mat4 const& proj);
	void drawAreaLight(CRenderTarget* pTarget, glm::vec3 const& color);
	
	void updateTransform(glm::mat4 const& world, glm::mat4 const& view, glm::mat4 const& proj);

	void Gather(const std::vector<Avpl>& path, CRenderTarget* pRenderTarget);
	void Normalize(CRenderTarget* pTarget, CRenderTarget* source, int normFactor);
	void Add(CRenderTarget* target, CRenderTarget* source1, CRenderTarget* source2);
	void Add(CRenderTarget* target, CRenderTarget* source);

	void SetTransformToCamera();

	void GetAVPLs(std::vector<Avpl>& avpls_shadowmap, std::vector<Avpl>& avpls_antiradiance);
	void SeparateAVPLs(const std::vector<Avpl> avpls, std::vector<Avpl>& avpls_shadowmap, std::vector<Avpl>& avpls_antiradiance, int numPaths);
	
	void Gather(std::vector<Avpl>& avpls_shadowmap, std::vector<Avpl>& avpls_antiradiance);
	void CalculateError();
	void DrawDebug();
	void CheckExport();

	bool UseAVPL(Avpl& avpl);
	
	void InitDebugLights();

	void ClearAccumulationBuffer();
	void ClearLighting();

	void CreateRandomAVPLs(std::vector<Avpl>& avpls, int numAVPLs);
	
	void DetermineUsedAvpls(const std::vector<Avpl>& avpls, std::vector<Avpl>& used);
	
	void GatherRadianceWithShadowMap(const std::vector<Avpl>& path, CRenderTarget* pRenderTarget);	
	
	void CreatePlaneHammersleySamples(int i);
		
	void GatherRadianceFromLightWithShadowMap(const Avpl& avpl, CRenderTarget* pRenderTarget);
	void FillShadowMap(const Avpl& avpl);
	void DrawLights(const std::vector<Avpl>& avpls, CRenderTarget* target);
	void DrawSceneSamples(CRenderTarget* target);
	void DrawBidirSceneSamples(CRenderTarget* target);
		
	glm::vec4 ColorForLight(const Avpl& light);
	
	void ExportPartialResult();
		
	// pointers to externally managed resources
	CCamera* m_camera;
	COGLContext* m_glContext;
	CConfigManager* m_confManager;

	std::unique_ptr<CCamera> m_clusterRenderCamera;
	std::unique_ptr<Scene> m_scene;
	std::unique_ptr<CShadowMap> m_shadowMap;
	std::unique_ptr<CGBuffer> m_gbuffer;
	std::unique_ptr<Postprocess> m_postProcess;
	std::unique_ptr<CExport> m_export;
	std::unique_ptr<TextureViewer> m_textureViewer;
	std::unique_ptr<FullScreenQuad> m_fullScreenQuad;
	std::unique_ptr<CImagePlane> m_imagePlane;
	std::unique_ptr<CPathTracingIntegrator> m_pathTracingIntegrator;
	std::unique_ptr<CExperimentData> m_experimentData;
	std::unique_ptr<PointCloud> m_pointCloud;
	std::unique_ptr<AABBCloud> m_aabbCloud;
	std::unique_ptr<AvplBvh> m_avplBvh;
		
	std::unique_ptr<CRenderTarget> m_resultRenderTarget;
	std::unique_ptr<CRenderTarget> m_errorRenderTarget;
	std::unique_ptr<CRenderTarget> m_gatherShadowmapRenderTarget;
	std::unique_ptr<CRenderTarget> m_gatherAntiradianceRenderTarget;
	std::unique_ptr<CRenderTarget> m_normalizeShadowmapRenderTarget;
	std::unique_ptr<CRenderTarget> m_normalizeAntiradianceRenderTarget;
	std::unique_ptr<CRenderTarget> m_lightDebugRenderTarget;
	std::unique_ptr<CRenderTarget> m_postProcessRenderTarget;
	std::unique_ptr<CRenderTarget> m_cudaRenderTarget;
	std::unique_ptr<CRenderTarget> m_cudaRenderTargetSum;
		
	std::unique_ptr<CProgram> m_gatherProgram;
	std::unique_ptr<CProgram> m_normalizeProgram;
	std::unique_ptr<CProgram> m_addProgram;
	std::unique_ptr<CProgram> m_createGBufferProgram;
	std::unique_ptr<CProgram> m_createSMProgram;
	std::unique_ptr<CProgram> m_gatherRadianceWithSMProgram;
	std::unique_ptr<CProgram> m_areaLightProgram;
	std::unique_ptr<CProgram> m_errorProgram;
	std::unique_ptr<CProgram> m_skyboxProgram;
	std::unique_ptr<CProgram> m_drawSphere;
	std::unique_ptr<CProgram> m_debugProgram;

	std::unique_ptr<COGLUniformBuffer> m_ubTransform;
	std::unique_ptr<COGLUniformBuffer> m_ubMaterial;
	std::unique_ptr<COGLUniformBuffer> m_ubLight;
	std::unique_ptr<COGLUniformBuffer> m_ubConfig;
	std::unique_ptr<COGLUniformBuffer> m_ubCamera;
	std::unique_ptr<COGLUniformBuffer> m_ubInfo;
	
	std::unique_ptr<COGLTexture2D> m_depthBuffer;
	std::unique_ptr<COGLTexture2D> m_testTexture;
	
	std::unique_ptr<COGLTexture2D> m_cudaTargetTexture;
	std::unique_ptr<cuda::CudaContext> m_cudaContext;
	std::unique_ptr<CudaGather> m_cudaGather;

	std::unique_ptr<AvplShooter> m_avplShooter;

	std::unique_ptr<SceneProbe> m_sceneProbe;

	std::unique_ptr<COGLSampler> m_linearSampler;
	std::unique_ptr<COGLSampler> m_pointSampler;
	std::unique_ptr<COGLSampler> m_shadowMapSampler;
	
	std::vector<Avpl> m_DebugAVPLs;
		
	std::unique_ptr<CTimer> m_clTimer;
	std::unique_ptr<CTimer> m_glTimer;
	std::unique_ptr<CTimer> m_cpuTimer;
	std::unique_ptr<CTimer> m_globalTimer;
	std::unique_ptr<CTimer> m_resultTimer;
	std::unique_ptr<CTimer> m_cpuFrameProfiler;
	std::unique_ptr<CTimer> m_gpuFrameProfiler;

	int m_Frame;
	int m_CurrentPathShadowmap;
	int m_numPathsAntiradiance;
	int m_CurrentPathAntiradiance;
	int m_NumPathsDebug;
	int m_NumAVPLsForNextDataExport;
	int m_NumAVPLsForNextImageExport;
	int m_TimeForNextDataExport;
	int m_TimeForNextImageExport;
	int m_NumAVPLs;
	
	bool m_Finished;
	bool m_FinishedDirectLighting;
	bool m_FinishedIndirectLighting;
	bool m_ProfileFrame;
	bool m_FinishedDebug;
	bool m_ClearLighting;
	bool m_ClearAccumulationBuffer;

	time_t m_StartTime;
};

#endif
