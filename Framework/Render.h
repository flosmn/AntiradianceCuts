#ifndef RENDER_H
#define RENDER_H

#include "GL/glew.h"

#include <glm/gtc/type_ptr.hpp>

#include "AVPL.h"

#include <vector>
#include <string>
#include <memory>

#include "CudaResources/CudaContext.hpp"

class Scene;
class CCamera;
class CConfigManager;
class CShadowMap;
class CPostprocess;
class CImagePlane;
class CPathTracingIntegrator;

class PointCloud;
class AABBCloud;
class AvplBvh;
class Sphere;
class SceneProbe;

class CudaGather;

class CAccumulationBuffer;
class CGBuffer;
class CProgram;
class CLightViewer;
class CTextureViewer;
class CFullScreenQuad;
class CExport;
class CRenderTarget;
class COctahedronMap;
class COctahedronAtlas;
class CModel;
class CClusterTree;
class CTimer;
class CAVPLImportanceSampling;
class CBidirInstantRadiosity;
class CRenderTarget;
class CExperimentData;

class COGLUniformBuffer;
class COGLSampler;
class COGLTexture2D;
class COGLTextureBuffer;
class COGLContext;
class COGLRenderTarget;
class COGLCubeMap;

class COCLContext;
class COCLBuffer;

class Renderer {
public:
	Renderer(CCamera* camera, COGLContext* glContext, CConfigManager* confManager);
	~Renderer();

	void Render();
	void RenderPathTracingReference();

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
	
	void ClusteringTestRender();

	void ProfileFrame() { m_ProfileFrame = true; }

	void TestClusteringSpeed();

	void UpdateBvhDebug();
	void RebuildBvh();
private:
	// functions of the render phase
	void SetUpRender();
	void CreateGBuffer();
	
	void Gather(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget);
	void GatherWithAtlas(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget);
	void GatherWithClustering(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget);
	void Normalize(CRenderTarget* pTarget, CRenderTarget* source, int normFactor);
	void Add(CRenderTarget* target, CRenderTarget* source1, CRenderTarget* source2);
	void Add(CRenderTarget* target, CRenderTarget* source);

	void SetTransformToCamera();

	void GetAVPLs(std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance);
	void SeparateAVPLs(const std::vector<AVPL> avpls, std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance, int numPaths);
	
	void Gather(std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance);
	void CalculateError();
	void DrawDebug();
	void CheckExport();

	void FillLightBuffer(const std::vector<AVPL>& avpls);
	void FillAVPLAtlas(const std::vector<AVPL>& avpls);
	void FillClusterAtlas(const std::vector<AVPL>& avpls);
	void CreateClustering(std::vector<AVPL>& avpls);
	
	bool UseAVPL(AVPL& avpl);

	void DirectEnvMapLighting();

	void InitDebugLights();

	void ClearAccumulationBuffer();
	void ClearLighting();

	void CreateRandomAVPLs(std::vector<AVPL>& avpls, int numAVPLs);
	
	void DetermineUsedAvpls(const std::vector<AVPL>& avpls, std::vector<AVPL>& used);
	
	void GatherRadianceWithShadowMap(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget);	
	void DrawAreaLight(CRenderTarget* pTarget);
	void DrawAreaLight(CRenderTarget* pTarget, glm::vec3 color);
	
	void CreatePlaneHammersleySamples(int i);
		
	void GatherRadianceFromLightWithShadowMap(const AVPL& avpl, CRenderTarget* pRenderTarget);
	void FillShadowMap(const AVPL& avpl);
	void DrawLights(const std::vector<AVPL>& avpls, CRenderTarget* target);
	void DrawSceneSamples(CRenderTarget* target);
	void DrawBidirSceneSamples(CRenderTarget* target);
		
	glm::vec4 ColorForLight(const AVPL& light);
	
	void ExportPartialResult();

	float GetAntiradFilterNormFactor();
	float IntegrateGauss();
	
	// pointers to externally managed resources
	CCamera* m_camera;
	COGLContext* m_glContext;
	CConfigManager* m_confManager;

	std::unique_ptr<CCamera> m_clusterRenderCamera;
	std::unique_ptr<Scene> m_scene;
	std::unique_ptr<CShadowMap> m_shadowMap;
	std::unique_ptr<CGBuffer> m_gbuffer;
	std::unique_ptr<CPostprocess> m_postProcess;
	std::unique_ptr<CExport> m_export;
	std::unique_ptr<CTextureViewer> m_textureViewer;
	std::unique_ptr<CFullScreenQuad> m_fullScreenQuad;
	std::unique_ptr<COCLContext> m_clContext;
	std::unique_ptr<CClusterTree> m_clusterTree;
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
	std::unique_ptr<CProgram> m_gatherWithAtlas;
	std::unique_ptr<CProgram> m_gatherWithClustering;
	std::unique_ptr<CProgram> m_normalizeProgram;
	std::unique_ptr<CProgram> m_addProgram;
	std::unique_ptr<CProgram> m_directEnvmapLighting;
	std::unique_ptr<CProgram> m_createGBufferProgram;
	std::unique_ptr<CProgram> m_createSMProgram;
	std::unique_ptr<CProgram> m_gatherRadianceWithSMProgram;
	std::unique_ptr<CProgram> m_areaLightProgram;
	std::unique_ptr<CProgram> m_drawOctahedronProgram;
	std::unique_ptr<CProgram> m_errorProgram;
	std::unique_ptr<CProgram> m_skyboxProgram;
	std::unique_ptr<CProgram> m_drawSphere;
	std::unique_ptr<CProgram> m_debugProgram;

	std::unique_ptr<COGLTextureBuffer> m_lightBuffer;
	std::unique_ptr<COGLTextureBuffer> m_clusterBuffer;
	std::unique_ptr<COGLTextureBuffer> m_avplPositions;

	std::unique_ptr<COctahedronAtlas> m_octahedronAtlas;
	std::unique_ptr<COctahedronMap> m_octahedronMap;

	std::unique_ptr<COGLUniformBuffer> m_ubTransform;
	std::unique_ptr<COGLUniformBuffer> m_ubMaterial;
	std::unique_ptr<COGLUniformBuffer> m_ubLight;
	std::unique_ptr<COGLUniformBuffer> m_ubConfig;
	std::unique_ptr<COGLUniformBuffer> m_ubCamera;
	std::unique_ptr<COGLUniformBuffer> m_ubInfo;
	std::unique_ptr<COGLUniformBuffer> m_ubAreaLight;
	std::unique_ptr<COGLUniformBuffer> m_ubModel;
	std::unique_ptr<COGLUniformBuffer> m_ubAtlasInfo;
	std::unique_ptr<COGLUniformBuffer> m_ubNormalize;
	
	std::unique_ptr<COGLTexture2D> m_depthBuffer;
	std::unique_ptr<COGLTexture2D> m_testTexture;
	
	std::unique_ptr<COGLTexture2D> m_cudaTargetTexture;
	std::unique_ptr<cuda::CudaContext> m_cudaContext;
	std::unique_ptr<CudaGather> m_cudaGather;

	std::unique_ptr<COGLCubeMap> m_cubeMap;
	std::unique_ptr<SceneProbe> m_sceneProbe;

	std::unique_ptr<COGLSampler> m_linearSampler;
	std::unique_ptr<COGLSampler> m_pointSampler;
	std::unique_ptr<COGLSampler> m_shadowMapSampler;
	
	std::vector<AVPL> m_DebugAVPLs;
	std::vector<AVPL> m_ClusterTestAVPLs;
	std::vector<AVPL> m_CollectedAVPLs;
	std::vector<AVPL> m_CollectedImportanceSampledAVPLs;
	
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
	int m_MaxNumAVPLs;
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
