#ifndef RENDER_H
#define RENDER_H

#include "GL/glew.h"

#include <glm/gtc/type_ptr.hpp>

#include "CSimpleKdTree.h"
#include "CClusterTree.h"

#include <vector>
#include <string>

class Scene;
class CCamera;
class CConfigManager;
class CShadowMap;
class CPostprocess;
class AVPL;

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

class COGLUniformBuffer;
class COGLSampler;
class COGLTexture2D;
class COGLTextureBuffer;
class COGLContext;

class COCLContext;
class COCLBuffer;

class Renderer {
public:
	Renderer(CCamera* _camera);
	~Renderer();
	
	bool Init();
	void Release();

	void Render();

	void SetOGLContext(COGLContext* pOGLContext) { m_pOGLContext = pOGLContext; }

	void WindowChanged();

	void ClearAccumulationBuffer();
	void ClearLighting();

	void PrintCameraConfig();
			
	void ConfigureLighting();

	void SetConfigManager(CConfigManager* pConfManager);

	void Stats();

	void Export();

	void NewDebugLights();
	
	void ClusteringTestRender();

private:
	// functions of the render phase
	void SetUpRender();
	void CreateGBuffer();
	
	void Gather(std::vector<AVPL*> path);
	void GatherWithAtlas(std::vector<AVPL*> path);
	void GatherWithClustering(std::vector<AVPL*> path);
	void Normalize();
	void Shade();	

	void InitDebugLights();

	std::vector<AVPL*> DetermineUsedAvpls(std::vector<AVPL*> path);
	
	void GatherRadianceWithShadowMap(std::vector<AVPL*> path);	
	void DebugRender();
	void DrawAreaLight();
	
	void CreatePlaneHammersleySamples(int i);
		
	void GatherRadianceFromLightWithShadowMap(AVPL* avpl);
	void FillShadowMap(AVPL* avpl);
	void DrawLights(std::vector<AVPL*> avpl);
	
	glm::vec4 ColorForLight(AVPL* light);
	
	void ExportPartialResult();

	CCamera *camera;
	CCamera *m_pClusterRenderCamera;
	Scene* scene;
	CShadowMap* m_pShadowMap;
	CGBuffer* m_pGBuffer;
	CConfigManager* m_pConfManager;
	
	CRenderTarget* m_pGatherRenderTarget;
	CRenderTarget* m_pNormalizeRenderTarget;
	CRenderTarget* m_pShadeRenderTarget;
		
	CProgram* m_pGatherProgram;
	CProgram* m_pGatherWithAtlas;
	CProgram* m_pGatherWithClustering;
	CProgram* m_pNormalizeProgram;
	CProgram* m_pShadeProgram;
		
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
	
	COGLContext* m_pOGLContext;

	CProgram* m_pCreateGBufferProgram;
	CProgram* m_pCreateSMProgram;
	CProgram* m_pGatherRadianceWithSMProgram;
	CProgram* m_pPointCloudProgram;
	CProgram* m_pAreaLightProgram;
	CProgram* m_pDrawOctahedronProgram;

	COGLSampler* m_pGLLinearSampler;
	COGLSampler* m_pGLPointSampler;
	COGLSampler* m_pGLShadowMapSampler;

	CFullScreenQuad* m_pFullScreenQuad;

	COCLContext* m_pCLContext;
	
	CModel* m_pOctahedron;

	int m_Frame;
	int m_CurrentPath;
		
	bool m_Finished;

	time_t m_StartTime;

	int m_MaxNumAVPLs;

	std::vector<AVPL*> m_DebugAVPLs;
	std::vector<AVPL*> m_ClusterTestAVPLs;
		
	CLightTree* m_pLightTree;
	CClusterTree* m_pClusterTree;

	CTimer* m_pOCLTimer;
	CTimer* m_pOGLTimer;
	CTimer* m_pCPUTimer;
};

#endif