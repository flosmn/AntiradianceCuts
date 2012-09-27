#ifndef RENDER_H
#define RENDER_H

#include "GL/glew.h"

#include <glm/gtc/type_ptr.hpp>

#include "CSimpleKdTree.h"
#include "CClusterTree.h"
#include "AVPL.h"

#include <vector>
#include <string>

class Scene;
class CCamera;
class CConfigManager;
class CShadowMap;
class CPostprocess;
class CImagePlane;
class CPathTracingIntegrator;

class CAccumulationBuffer;
class CGBuffer;
class CProgram;
class CLightViewer;
class CTextureViewer;
class CFullScreenQuad;
class CPointCloud;
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
	Renderer(CCamera* _camera);
	~Renderer();
	
	bool Init();
	void Release();

	void Render();
	void RenderPathTracingReference();

	void CancelRender();

	void SetOGLContext(COGLContext* pOGLContext) { m_pOGLContext = pOGLContext; }

	void WindowChanged();

	void IssueClearLighting();
	void IssueClearAccumulationBuffer();
	
	void PrintCameraConfig();

	void UpdateAreaLights();
			
	void UpdateUniformBuffers();

	void SetConfigManager(CConfigManager* pConfManager);

	void Stats();

	void Export();

	void NewDebugLights();
	
	void ClusteringTestRender();

	void ProfileFrame() { m_ProfileFrame = true; }

	void StartCollectingAVPLs();
	void EndCollectingAVPLs();
	void StartCollectingISAVPLs();
	void EndCollectingISAVPLs();
	void ClearCollectedAVPLs() { m_CollectedAVPLs.clear(); m_CollectedImportanceSampledAVPLs.clear(); }

	void TestClusteringSpeed();

private:
	// functions of the render phase
	void SetUpRender();
	void CreateGBuffer();
	
	void Gather(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget);
	void GatherWithAtlas(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget);
	void GatherWithClustering(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget);
	void Normalize(CRenderTarget* pTarget, CRenderTarget* source, int normFactor);
	void Shade(CRenderTarget* target, CRenderTarget* source);	
	void Add(CRenderTarget* target, CRenderTarget* source1, CRenderTarget* source2);

	void SetTranformToCamera();

	void GetAVPLs(std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance);
	void SeparateAVPLs(const std::vector<AVPL> avpls, std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance, int numPaths);
	
	void Gather(std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance);
	void Normalize();
	void Shade();
	void Finalize();
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
		
	void DrawPointCloud(const std::vector<POINT_CLOUD_POINT>& pcp, CRenderTarget* target);

	glm::vec4 ColorForLight(const AVPL& light);
	
	void ExportPartialResult();

	float GetAntiradFilterNormFactor();
	float IntegrateGauss();
	
	CCamera *camera;
	CCamera *m_pClusterRenderCamera;
	Scene* scene;
	CShadowMap* m_pShadowMap;
	CGBuffer* m_pGBuffer;
	CConfigManager* m_pConfManager;
		
	CRenderTarget* m_pResultRenderTarget;
	CRenderTarget* m_pErrorRenderTarget;
	CRenderTarget* m_pGatherShadowmapRenderTarget;
	CRenderTarget* m_pGatherAntiradianceRenderTarget;
	CRenderTarget* m_pNormalizeShadowmapRenderTarget;
	CRenderTarget* m_pNormalizeAntiradianceRenderTarget;
	CRenderTarget* m_pShadeShadowmapRenderTarget;
	CRenderTarget* m_pShadeAntiradianceRenderTarget;
		
	CProgram* m_pGatherProgram;
	CProgram* m_pGatherWithAtlas;
	CProgram* m_pGatherWithClustering;
	CProgram* m_pNormalizeProgram;
	CProgram* m_pShadeProgram;
	CProgram* m_pAddProgram;
	CProgram* m_pDirectEnvmapLighting;
		
	CRenderTarget* m_pLightDebugRenderTarget;
	CRenderTarget* m_pPostProcessRenderTarget;
	CPostprocess* m_pPostProcess;

	COGLTextureBuffer* m_pOGLLightBuffer;
	COGLTextureBuffer* m_pOGLClusterBuffer;
	
	COGLTextureBuffer* m_pAVPLPositions;

	CPointCloud* m_pPointCloud;
	CExport* m_Export;
		
	CTextureViewer* m_pTextureViewer;

	COctahedronAtlas* m_pOctahedronAtlas;
	COctahedronMap* m_pOctahedronMap;

	COGLUniformBuffer* m_pUBTransform;
	COGLUniformBuffer* m_pUBMaterial;
	COGLUniformBuffer* m_pUBLight;
	COGLUniformBuffer* m_pUBConfig;
	COGLUniformBuffer* m_pUBCamera;
	COGLUniformBuffer* m_pUBInfo;
	COGLUniformBuffer* m_pUBAreaLight;
	COGLUniformBuffer* m_pUBModel;
	COGLUniformBuffer* m_pUBAtlasInfo;
	COGLUniformBuffer* m_pUBNormalize;
	
	COGLTexture2D* m_pDepthBuffer;
	COGLTexture2D* m_pTestTexture;

	COGLCubeMap* m_pCubeMap;

	COGLContext* m_pOGLContext;

	CProgram* m_pCreateGBufferProgram;
	CProgram* m_pCreateSMProgram;
	CProgram* m_pGatherRadianceWithSMProgram;
	CProgram* m_pPointCloudProgram;
	CProgram* m_pAreaLightProgram;
	CProgram* m_pDrawOctahedronProgram;
	CProgram* m_pErrorProgram;
	CProgram* m_pSkyboxProgram;

	COGLSampler* m_pGLLinearSampler;
	COGLSampler* m_pGLPointSampler;
	COGLSampler* m_pGLShadowMapSampler;

	CAVPLImportanceSampling* m_pAVPLImportanceSampling;
	CBidirInstantRadiosity* m_pBidirInstantRadiosity;

	CFullScreenQuad* m_pFullScreenQuad;

	COCLContext* m_pCLContext;
	
	CModel* m_pOctahedron;

	int m_Frame;
	int m_CurrentPathShadowmap;
	int m_CurrentPathAntiradiance;
	
	bool m_Finished;
	bool m_FinishedDirectLighting;
	bool m_FinishedIndirectLighting;
	bool m_ProfileFrame;
	bool m_FinishedDebug;

	time_t m_StartTime;

	int m_MaxNumAVPLs;
	int m_NumPathsDebug;
		
	std::vector<AVPL> m_DebugAVPLs;
	std::vector<AVPL> m_ClusterTestAVPLs;

	std::vector<AVPL> m_CollectedAVPLs;
	std::vector<AVPL> m_CollectedImportanceSampledAVPLs;
	
	CClusterTree* m_pClusterTree;

	CTimer* m_pOCLTimer;
	CTimer* m_pOGLTimer;
	CTimer* m_pCPUTimer;
	CTimer* m_pGlobalTimer;
	CTimer* m_pResultTimer;
	CTimer* m_pCPUFrameProfiler;
	CTimer* m_pGPUFrameProfiler;

	CImagePlane* m_pImagePlane;
	CPathTracingIntegrator* m_pPathTracingIntegrator;
	CExperimentData* m_pExperimentData;

	int m_NumAVPLsForNextDataExport;
	int m_NumAVPLsForNextImageExport;
	int m_TimeForNextDataExport;
	int m_TimeForNextImageExport;
	int m_NumAVPLs;

	bool m_ClearLighting;
	bool m_ClearAccumulationBuffer;
};

#endif