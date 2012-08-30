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
class CLightTree;
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
	void RenderDirectIndirectLight();
	void RenderPathTracingReference();

	void CancelRender();

	void SetOGLContext(COGLContext* pOGLContext) { m_pOGLContext = pOGLContext; }

	void WindowChanged();

	void ClearAccumulationBuffer();
	void ClearLighting();

	void PrintCameraConfig();

	void UpdateAreaLights();
			
	void ConfigureLighting();

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

	void DirectEnvMapLighting();

	void InitDebugLights();
	
	void DetermineUsedAvpls(const std::vector<AVPL>& avpls, std::vector<AVPL>& used);
	
	void GatherRadianceWithShadowMap(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget);	
	void DebugRender();
	void DrawAreaLight();
	
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

	void UpdateTransform();

	CCamera *camera;
	CCamera *m_pClusterRenderCamera;
	Scene* scene;
	CShadowMap* m_pShadowMap;
	CGBuffer* m_pGBuffer;
	CConfigManager* m_pConfManager;
	
	CRenderTarget* m_pAVPLRenderTarget;
	CRenderTarget* m_pGatherRenderTarget;
	CRenderTarget* m_pNormalizeRenderTarget;
	CRenderTarget* m_pShadeRenderTarget;
	CRenderTarget* m_pGatherDirectLightRenderTarget;
	CRenderTarget* m_pGatherIndirectLightRenderTarget;
	CRenderTarget* m_pNormalizeDirectLightRenderTarget;
	CRenderTarget* m_pNormalizeIndirectLightRenderTarget;
	CRenderTarget* m_pShadeDirectLightRenderTarget;
	CRenderTarget* m_pShadeIndirectLightRenderTarget;
	
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
	int m_CurrentPath;
	int m_CurrentVPLDirect;
	
	bool m_Finished;
	bool m_FinishedDirectLighting;
	bool m_FinishedIndirectLighting;
	bool m_ProfileFrame;

	time_t m_StartTime;

	int m_MaxNumAVPLs;
		
	std::vector<AVPL> m_DebugAVPLs;
	std::vector<AVPL> m_ClusterTestAVPLs;

	std::vector<AVPL> m_CollectedAVPLs;
	std::vector<AVPL> m_CollectedImportanceSampledAVPLs;
	
	CLightTree* m_pLightTree;
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
};

#endif