#include "Render.h"

#include "CL/cl.h"

#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Macros.h"
#include "Structs.h"

#include "CConfigManager.h"
#include "CGBuffer.h"
#include "CTimer.h"
#include "CProgram.h"
#include "CPointCloud.h"
#include "COctahedronMap.h"
#include "COctahedronAtlas.h"

#include "AVPL.h"
#include "Scene.h"
#include "CCamera.h"
#include "CShadowMap.h"
#include "CPostprocess.h"
#include "CRenderTarget.h"
#include "CClusterTree.h"
#include "CLightTree.h"
#include "CPriorityQueue.h"
#include "CAVPLImportanceSampling.h"

#include "LightTreeTypes.h"

#include "Utils\Util.h"
#include "Utils\GLErrorUtil.h"
#include "Utils\ShaderUtil.h"
#include "Utils\CTextureViewer.h"
#include "Utils\CExport.h"
#include "Utils\Rand.h"

#include "MeshResources\CFullScreenQuad.h"
#include "MeshResources\CModel.h"

#include "OGLResources\COGLUniformBuffer.h"
#include "OGLResources\COGLTexture2D.h"
#include "OGLResources\COGLFrameBuffer.h"
#include "OGLResources\COGLProgram.h"
#include "OGLResources\COGLSampler.h"
#include "OGLResources\COGLTextureBuffer.h"

#include "OCLResources\COCLContext.h"
#include "OCLResources\COCLProgram.h"
#include "OCLResources\COCLKernel.h"
#include "OCLResources\COCLBuffer.h"
#include "OCLResources\COCLTexture2D.h"

#include <memory>
#include <string>
#include <sstream>
#include <time.h>

std::vector<Light*> initialLights;

Renderer::Renderer(CCamera* _camera) {
	camera = _camera;
	
	scene = 0;

	m_pCLContext = new COCLContext();
		
	m_pShadowMap = new CShadowMap();
	
	m_pOctahedronAtlas = new COctahedronAtlas(m_pCLContext);
	m_pOctahedronMap = new COctahedronMap();

	m_Export = new CExport();
	
	m_pDepthBuffer = new COGLTexture2D("Renderer.m_pDepthBuffer");
	m_pTestTexture = new COGLTexture2D("Renderer.m_pTestTexture");
	
	m_pGBuffer = new CGBuffer();
		
	m_pLightDebugRenderTarget = new CRenderTarget();
	m_pPostProcessRenderTarget = new CRenderTarget();
	m_pPostProcess = new CPostprocess();

	m_pLightTree = new CLightTree();
	m_pClusterTree = new CClusterTree();
	
	m_pTextureViewer = new CTextureViewer();

	m_pUBTransform = new COGLUniformBuffer("Renderer.m_pUBTransform");
	m_pUBMaterial = new COGLUniformBuffer("Renderer.m_pUBMaterial");
	m_pUBLight = new COGLUniformBuffer("Renderer.m_pUBLight");
	m_pUBConfig = new COGLUniformBuffer("Renderer.m_pUBConfig");
	m_pUBCamera = new COGLUniformBuffer("Renderer.m_pUBCamera");
	m_pUBInfo = new COGLUniformBuffer("Renderer.m_pUBInfo");
	m_pUBAreaLight = new COGLUniformBuffer("Renderer.m_pUBAreaLight");
	m_pUBModel = new COGLUniformBuffer("Renderer.m_pUBModel");
	m_pUBAtlasInfo = new COGLUniformBuffer("Renderer.m_pUBAtlasInfo");

	m_pGLLinearSampler = new COGLSampler("Renderer.m_pGLLinearSampler");
	m_pGLPointSampler = new COGLSampler("Renderer.m_pGLPointSampler");
	m_pGLShadowMapSampler = new COGLSampler("Renderer.m_pGLShadowMapSampler");

	m_pGatherRenderTarget = new CRenderTarget();
	m_pNormalizeRenderTarget = new CRenderTarget();
	m_pShadeRenderTarget = new CRenderTarget();
	m_pGatherDirectLightRenderTarget = new CRenderTarget();
	m_pGatherIndirectLightRenderTarget = new CRenderTarget();
	m_pNormalizeDirectLightRenderTarget = new CRenderTarget();
	m_pNormalizeIndirectLightRenderTarget = new CRenderTarget();
	m_pShadeDirectLightRenderTarget = new CRenderTarget();
	m_pShadeIndirectLightRenderTarget = new CRenderTarget();
	m_pAVPLRenderTarget = new CRenderTarget();

	m_pGatherProgram = new CProgram("Renderer.m_pGatherProgram", "Shaders\\Gather.vert", "Shaders\\Gather.frag");
	m_pGatherWithAtlas = new CProgram("Renderer.m_pGatherProgram", "Shaders\\Gather.vert", "Shaders\\GatherWithAtlas.frag");
	m_pGatherWithClustering = new CProgram("Renderer.m_pGatherProgram", "Shaders\\Gather.vert", "Shaders\\GatherWithClustering.frag");
	m_pNormalizeProgram = new CProgram("Renderer.m_pNormalizeProgram", "Shaders\\Gather.vert", "Shaders\\Normalize.frag");
	m_pShadeProgram = new CProgram("Renderer.m_pShadeProgram", "Shaders\\Gather.vert", "Shaders\\Shade.frag");
	m_pAddProgram = new CProgram("Renderer.m_pAddProgram", "Shaders\\Gather.vert", "Shaders\\Add.frag");

	m_pCreateGBufferProgram = new CProgram("Renderer.m_pCreateGBufferProgram", "Shaders\\CreateGBuffer.vert", "Shaders\\CreateGBuffer.frag");
	m_pCreateSMProgram = new CProgram("Renderer.m_pCreateSMProgram", "Shaders\\CreateSM.vert", "Shaders\\CreateSM.frag");
	m_pGatherRadianceWithSMProgram = new CProgram("Renderer.m_pGatherRadianceWithSMProgram", "Shaders\\Gather.vert", "Shaders\\GatherRadianceWithSM.frag");
	m_pPointCloudProgram = new CProgram("Renderer.m_pPointCloudProgram", "Shaders\\PointCloud.vert", "Shaders\\PointCloud.frag");
	m_pAreaLightProgram = new CProgram("Renderer.m_pAreaLightProgram", "Shaders\\DrawAreaLight.vert", "Shaders\\DrawAreaLight.frag");
	m_pDrawOctahedronProgram = new CProgram("Renderer.m_pDrawOctahedronProgram", "Shaders\\DrawOctahedron.vert", "Shaders\\DrawOctahedron.frag");

	m_pFullScreenQuad = new CFullScreenQuad();
	m_pOctahedron = new CModel();

	m_pPointCloud = new CPointCloud();

	m_pOGLLightBuffer = new COGLTextureBuffer("Renderer.m_pOGLLightBuffer");
	m_pOGLClusterBuffer = new COGLTextureBuffer("Renderer.m_pOGLClusterBuffer");
	
	m_pAVPLPositions = new COGLTextureBuffer("Renderer.m_pAVPLPositions");
	
	m_pCPUTimer = new CTimer(CTimer::CPU);
	m_pOGLTimer = new CTimer(CTimer::OGL);
	m_pOCLTimer = new CTimer(CTimer::OCL, m_pCLContext);
	m_pGlobalTimer = new CTimer(CTimer::CPU);
	m_pCPUFrameProfiler = new CTimer(CTimer::CPU);
	m_pGPUFrameProfiler = new CTimer(CTimer::OGL);

	m_Finished = false;
	m_FinishedDirectLighting = false;
	m_FinishedIndirectLighting = false;

	m_ProfileFrame = false;
}

Renderer::~Renderer() {
	SAFE_DELETE(scene);
	SAFE_DELETE(m_Export);
	SAFE_DELETE(m_pShadowMap);
	SAFE_DELETE(m_pGBuffer);
	SAFE_DELETE(m_pLightTree);
	SAFE_DELETE(m_pClusterTree);

	SAFE_DELETE(m_pGatherRenderTarget);
	SAFE_DELETE(m_pNormalizeRenderTarget);
	SAFE_DELETE(m_pShadeProgram);
	SAFE_DELETE(m_pGatherDirectLightRenderTarget);
	SAFE_DELETE(m_pGatherIndirectLightRenderTarget);
	SAFE_DELETE(m_pNormalizeDirectLightRenderTarget);
	SAFE_DELETE(m_pNormalizeIndirectLightRenderTarget);
	SAFE_DELETE(m_pShadeDirectLightRenderTarget);
	SAFE_DELETE(m_pShadeIndirectLightRenderTarget);
	SAFE_DELETE(m_pAVPLRenderTarget);

	SAFE_DELETE(m_pGatherProgram);
	SAFE_DELETE(m_pGatherWithAtlas);
	SAFE_DELETE(m_pGatherWithClustering);
	SAFE_DELETE(m_pNormalizeProgram);
	SAFE_DELETE(m_pShadeProgram);
	SAFE_DELETE(m_pAddProgram);
	
	SAFE_DELETE(m_pLightDebugRenderTarget);
	SAFE_DELETE(m_pPostProcessRenderTarget);
	SAFE_DELETE(m_pPostProcess);
	
	SAFE_DELETE(m_pTextureViewer);
	SAFE_DELETE(m_pFullScreenQuad);
	SAFE_DELETE(m_pOctahedron);
	SAFE_DELETE(m_pPointCloud);

	SAFE_DELETE(m_pUBTransform);
	SAFE_DELETE(m_pUBMaterial);
	SAFE_DELETE(m_pUBLight);
	SAFE_DELETE(m_pUBConfig);
	SAFE_DELETE(m_pUBCamera);
	SAFE_DELETE(m_pUBInfo);
	SAFE_DELETE(m_pUBAreaLight);
	SAFE_DELETE(m_pUBModel);
	SAFE_DELETE(m_pUBAtlasInfo);

	SAFE_DELETE(m_pCreateGBufferProgram);
	SAFE_DELETE(m_pCreateSMProgram);
	SAFE_DELETE(m_pGatherRadianceWithSMProgram);
	SAFE_DELETE(m_pPointCloudProgram);
	SAFE_DELETE(m_pAreaLightProgram);
	SAFE_DELETE(m_pDrawOctahedronProgram);

	SAFE_DELETE(m_pGLPointSampler);
	SAFE_DELETE(m_pGLLinearSampler);
	SAFE_DELETE(m_pGLShadowMapSampler);

	SAFE_DELETE(m_pDepthBuffer);
	SAFE_DELETE(m_pTestTexture);
	
	SAFE_DELETE(m_pAVPLPositions);

	SAFE_DELETE(m_pOGLLightBuffer);
	SAFE_DELETE(m_pOGLClusterBuffer);
	
	SAFE_DELETE(m_pOctahedronAtlas);
	SAFE_DELETE(m_pOctahedronMap);

	SAFE_DELETE(m_pCPUTimer);
	SAFE_DELETE(m_pOCLTimer);
	SAFE_DELETE(m_pOGLTimer);
	SAFE_DELETE(m_pGlobalTimer);

	SAFE_DELETE(m_pCLContext);

	SAFE_DELETE(m_pAVPLImportanceSampling);
}

bool Renderer::Init() 
{	
	V_RET_FOF(m_pCLContext->Init(m_pOGLContext));
		
	V_RET_FOF(m_pDepthBuffer->Init(camera->GetWidth(), camera->GetHeight(), GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, false));
	V_RET_FOF(m_pTestTexture->Init(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));
	
	V_RET_FOF(m_pUBTransform->Init(sizeof(TRANSFORM), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBMaterial->Init(sizeof(MATERIAL), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBLight->Init(sizeof(AVPL), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBConfig->Init(sizeof(CONFIG), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBCamera->Init(sizeof(CAMERA), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBInfo->Init(sizeof(INFO), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBAreaLight->Init(sizeof(AREA_LIGHT), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBModel->Init(sizeof(MODEL), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBAtlasInfo->Init(sizeof(ATLAS_INFO), 0, GL_DYNAMIC_DRAW));

	V_RET_FOF(m_pTextureViewer->Init());
	V_RET_FOF(m_pPointCloud->Init());
	
	V_RET_FOF(m_pCreateGBufferProgram->Init());
	V_RET_FOF(m_pCreateSMProgram->Init());
	V_RET_FOF(m_pGatherRadianceWithSMProgram->Init());
	V_RET_FOF(m_pPointCloudProgram->Init());

	V_RET_FOF(m_pGatherProgram->Init());
	V_RET_FOF(m_pGatherWithAtlas->Init());
	V_RET_FOF(m_pGatherWithClustering->Init());
	V_RET_FOF(m_pNormalizeProgram->Init());
	V_RET_FOF(m_pShadeProgram->Init());
	V_RET_FOF(m_pAreaLightProgram->Init());
	V_RET_FOF(m_pAddProgram->Init());
	
	V_RET_FOF(m_pGatherRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 4, m_pDepthBuffer));
	V_RET_FOF(m_pNormalizeRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 3, 0));
	V_RET_FOF(m_pShadeRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));
	V_RET_FOF(m_pGatherDirectLightRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 4, 0));
	V_RET_FOF(m_pGatherIndirectLightRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 4, 0));
	V_RET_FOF(m_pNormalizeDirectLightRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 3, 0));
	V_RET_FOF(m_pNormalizeIndirectLightRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 3, 0));
	V_RET_FOF(m_pShadeDirectLightRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));
	V_RET_FOF(m_pShadeIndirectLightRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));
	V_RET_FOF(m_pAVPLRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));

	V_RET_FOF(m_pPostProcess->Init());
	V_RET_FOF(m_pPostProcessRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, 0));
	V_RET_FOF(m_pLightDebugRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));

	m_pAreaLightProgram->BindUniformBuffer(m_pUBTransform, "transform");
	m_pAreaLightProgram->BindUniformBuffer(m_pUBAreaLight, "arealight");

	m_pCreateGBufferProgram->BindUniformBuffer(m_pUBTransform, "transform");
	m_pCreateGBufferProgram->BindUniformBuffer(m_pUBMaterial, "material");
		
	m_pGatherRadianceWithSMProgram->BindUniformBuffer(m_pUBLight, "light");
	m_pGatherRadianceWithSMProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherRadianceWithSMProgram->BindUniformBuffer(m_pUBCamera, "camera");

	V_RET_FOF(m_pGLLinearSampler->Init(GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP));
	V_RET_FOF(m_pGLPointSampler->Init(GL_NEAREST, GL_NEAREST, GL_REPEAT, GL_REPEAT));
	V_RET_FOF(m_pGLShadowMapSampler->Init(GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER));
	
	m_pGatherRadianceWithSMProgram->BindSampler(0, m_pGLShadowMapSampler);
	m_pGatherRadianceWithSMProgram->BindSampler(1, m_pGLPointSampler);
	m_pGatherRadianceWithSMProgram->BindSampler(2, m_pGLPointSampler);

	m_pGatherProgram->BindSampler(0, m_pGLPointSampler);
	m_pGatherProgram->BindSampler(1, m_pGLPointSampler);
	m_pGatherProgram->BindSampler(2, m_pGLPointSampler);

	m_pGatherWithAtlas->BindSampler(0, m_pGLPointSampler);
	m_pGatherWithAtlas->BindSampler(1, m_pGLPointSampler);
	m_pGatherWithAtlas->BindSampler(2, m_pGLPointSampler);
	m_pGatherWithAtlas->BindSampler(3, m_pGLLinearSampler);

	m_pGatherWithAtlas->BindUniformBuffer(m_pUBInfo, "info_block");
	m_pGatherWithAtlas->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherWithAtlas->BindUniformBuffer(m_pUBCamera, "camera");
	m_pGatherWithAtlas->BindUniformBuffer(m_pUBAtlasInfo, "atlas_info");

	m_pGatherWithClustering->BindSampler(0, m_pGLPointSampler);
	m_pGatherWithClustering->BindSampler(1, m_pGLPointSampler);
	m_pGatherWithClustering->BindSampler(2, m_pGLPointSampler);
	m_pGatherWithClustering->BindSampler(3, m_pGLPointSampler);
	m_pGatherWithClustering->BindSampler(4, m_pGLPointSampler);
	m_pGatherWithClustering->BindSampler(5, m_pGLPointSampler);

	m_pGatherWithClustering->BindUniformBuffer(m_pUBInfo, "info_block");
	m_pGatherWithClustering->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherWithClustering->BindUniformBuffer(m_pUBCamera, "camera");
	m_pGatherWithClustering->BindUniformBuffer(m_pUBAtlasInfo, "atlas_info");

	m_pGatherProgram->BindUniformBuffer(m_pUBInfo, "info_block");
	m_pGatherProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pNormalizeProgram->BindSampler(0, m_pGLPointSampler);
	m_pNormalizeProgram->BindSampler(1, m_pGLPointSampler);
	m_pNormalizeProgram->BindSampler(2, m_pGLPointSampler);

	m_pNormalizeProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pNormalizeProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pShadeProgram->BindSampler(0, m_pGLPointSampler);
	m_pShadeProgram->BindSampler(1, m_pGLPointSampler);
	m_pShadeProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pAddProgram->BindSampler(0, m_pGLPointSampler);
	m_pAddProgram->BindSampler(1, m_pGLPointSampler);
	m_pAddProgram->BindUniformBuffer(m_pUBCamera, "camera");
		
	V_RET_FOF(m_pDrawOctahedronProgram->Init());
	m_pDrawOctahedronProgram->BindSampler(0, m_pGLPointSampler);
	m_pDrawOctahedronProgram->BindUniformBuffer(m_pUBTransform, "transform");
	m_pDrawOctahedronProgram->BindUniformBuffer(m_pUBModel, "model");
	
	V_RET_FOF(m_pFullScreenQuad->Init());
	
	V_RET_FOF(m_pShadowMap->Init(2048));
	
	V_RET_FOF(m_pGBuffer->Init(camera->GetWidth(), camera->GetHeight(), m_pDepthBuffer));
		
	scene = new Scene(camera, m_pConfManager);
	scene->Init();
	scene->LoadCornellBoxSmall();
		
	ConfigureLighting();

	time(&m_StartTime);

	int dim_atlas = 4096;
	int dim_tile = 16;
		
	ATLAS_INFO atlas_info;
	atlas_info.dim_atlas = dim_atlas;
	atlas_info.dim_tile = dim_tile;
	m_pUBAtlasInfo->UpdateData(&atlas_info);
	
	m_pOctahedronMap->Init(32);
	m_pOctahedronMap->FillWithDebugData();

	m_MaxNumAVPLs = int(std::pow(float(dim_atlas) / float(dim_tile), 2.f));

	V_RET_FOF(m_pAVPLPositions->Init(sizeof(AVPL_POSITION) * m_MaxNumAVPLs, GL_STATIC_DRAW));

	V_RET_FOF(m_pOGLLightBuffer->Init(sizeof(AVPL_BUFFER) * m_MaxNumAVPLs, GL_STATIC_DRAW));
	V_RET_FOF(m_pOGLClusterBuffer->Init(sizeof(CLUSTER_BUFFER) * (2 * m_MaxNumAVPLs - 1), GL_STATIC_DRAW));

	m_pOctahedronAtlas->Init(dim_atlas, dim_tile, m_MaxNumAVPLs);
	
	V_RET_FOF(m_pOctahedron->Init("octahedron"));
		
	glm::mat4 scale = glm::scale(IdentityMatrix(), glm::vec3(70.f, 70.f, 70.f));
	glm::mat4 trans = glm::translate(IdentityMatrix(), glm::vec3(278.f, 273.f, 270.f));
	m_pOctahedron->SetWorldTransform(trans * scale);

	m_pAVPLImportanceSampling = new CAVPLImportanceSampling(scene, m_pConfManager);
	scene->SetAVPLImportanceSampling(m_pAVPLImportanceSampling);
	
	InitDebugLights();
		
	ClearAccumulationBuffer();
	
	return true;
}

void Renderer::Release()
{
	CheckGLError("CDSRenderer", "CDSRenderer::Release()");

	m_pDepthBuffer->Release();
	m_pTestTexture->Release();

	m_pGBuffer->Release();
	m_pTextureViewer->Release();
	m_pFullScreenQuad->Release();
	m_pPointCloud->Release();
	m_pAVPLPositions->Release();

	m_pOGLLightBuffer->Release();	
	m_pOGLClusterBuffer->Release();

	m_pLightDebugRenderTarget->Release();
	m_pPostProcessRenderTarget->Release();
	m_pPostProcess->Release();

	scene->Release();

	m_pShadowMap->Release();

	m_pCreateGBufferProgram->Release();
	m_pCreateSMProgram->Release();
	m_pGatherRadianceWithSMProgram->Release();
	m_pPointCloudProgram->Release();
	m_pAreaLightProgram->Release();
	m_pDrawOctahedronProgram->Release();
	m_pAddProgram->Release();

	m_pGatherRenderTarget->Release();
	m_pNormalizeRenderTarget->Release();
	m_pShadeRenderTarget->Release();
	m_pGatherDirectLightRenderTarget->Release();
	m_pGatherIndirectLightRenderTarget->Release();
	m_pNormalizeDirectLightRenderTarget->Release();
	m_pNormalizeIndirectLightRenderTarget->Release();
	m_pShadeDirectLightRenderTarget->Release();
	m_pShadeIndirectLightRenderTarget->Release();
	m_pAVPLRenderTarget->Release();

	m_pNormalizeProgram->Release();
	m_pGatherProgram->Release();
	m_pGatherWithAtlas->Release();
	m_pGatherWithClustering->Release();
	m_pShadeProgram->Release();
	
	m_pGLPointSampler->Release();
	m_pGLLinearSampler->Release();
	m_pGLShadowMapSampler->Release();

	m_pUBTransform->Release();
	m_pUBMaterial->Release();
	m_pUBLight->Release();
	m_pUBConfig->Release();
	m_pUBCamera->Release();
	m_pUBInfo->Release();
	m_pUBAreaLight->Release();
	m_pUBModel->Release();
	m_pUBAtlasInfo->Release();

	m_pOctahedron->Release();
	m_pOctahedronAtlas->Release();
	m_pOctahedronMap->Release();
		
	m_pCLContext->Release();

	m_pLightTree->Release();
	m_pClusterTree->Release();
		
	m_ClusterTestAVPLs.clear();
}

void Renderer::ClusteringTestRender()
{	
	SetUpRender();
	
	{
		CRenderTargetLock lock(m_pShadeRenderTarget);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	}

	DrawLights(m_ClusterTestAVPLs, m_pShadeRenderTarget);

	m_pTextureViewer->DrawTexture(m_pShadeRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
}

void Renderer::Render() 
{	
	CTimer cpuTimer(CTimer::CPU);
	CTimer gpuTimer(CTimer::OGL);

	cpuTimer.Start();
	gpuTimer.Start();

	/*
		OpenCL test
	*/

	if(m_ProfileFrame) std::cout << std::endl;
	if(m_ProfileFrame) std::cout << "Profile frame --------------- " << std::endl;
	if(m_ProfileFrame) std::cout << std::endl;

	SetUpRender();

	uint width = camera->GetWidth();
	uint height = camera->GetHeight();
	
	if(m_pConfManager->GetConfVars()->UseAVPLImportanceSampling && !m_pConfManager->GetConfVars()->UseDebugMode)
	{
		m_pAVPLImportanceSampling->UpdateCurrentIrradiance(m_pNormalizeRenderTarget->GetBuffer(1));
		m_pAVPLImportanceSampling->UpdateCurrentAntiirradiance(m_pNormalizeRenderTarget->GetBuffer(2));
		m_pAVPLImportanceSampling->SetNumberOfSceneSamples(m_pConfManager->GetConfVars()->NumSceneSamples);
		m_pAVPLImportanceSampling->CreateSceneSamples();
	}

	if(m_CurrentPath == 0)
	{
		m_pGlobalTimer->Start();
		m_pOGLTimer->Start();
		CreateGBuffer();
		m_pOGLTimer->Stop("CreateGBuffer");
	}
	
	std::vector<AVPL> avpls;

	if(m_pConfManager->GetConfVars()->UseAntiradiance)
	{
		if(m_pConfManager->GetConfVars()->UseDebugMode && !m_Finished)
		{
			if(!m_Finished)
			{
				if(m_pConfManager->GetConfVars()->GatherWithAVPLAtlas)
				{
					if(m_pConfManager->GetConfVars()->GatherWithAVPLClustering)
					{
						if(m_pConfManager->GetConfVars()->UseLightTree)
						{
							m_pCPUTimer->Start();
							m_pLightTree->Release();		
							m_pLightTree->BuildTreeTweakNN(m_DebugAVPLs, 0.f);
							m_pLightTree->Color(m_DebugAVPLs, m_pConfManager->GetConfVars()->ClusterDepth);
							m_pCPUTimer->Stop("Clustering (LightTree)");
						}
						else
						{
							m_pCPUTimer->Start();
							m_pClusterTree->Release();		
							m_pClusterTree->BuildTree(m_DebugAVPLs);
							m_pClusterTree->Color(m_DebugAVPLs, m_pConfManager->GetConfVars()->ClusterDepth);
							m_pCPUTimer->Stop("Clustering (ClusterTree)");
						}

						m_pOGLTimer->Start();
						GatherWithClustering(m_DebugAVPLs, m_pGatherRenderTarget);
						m_pOGLTimer->Stop("GatherWithClustering");
					}
					else	
					{
						m_pOGLTimer->Start();
						GatherWithAtlas(m_DebugAVPLs, m_pGatherRenderTarget);
						m_pOGLTimer->Stop("GatherWithAtlas");						
					}
				}
				else
				{
					m_pOGLTimer->Start();
					Gather(m_DebugAVPLs, m_pGatherRenderTarget);
					m_pOGLTimer->Stop("Gather");	
				}

				if(m_pConfManager->GetConfVars()->DrawLights)
					DrawLights(m_DebugAVPLs, m_pGatherRenderTarget);

				m_CurrentPath += m_pConfManager->GetConfVars()->NumPaths;

				m_Finished = true;

				cpuTimer.Stop("Render (CPU timer)");
				gpuTimer.Stop("Render (GPU timer)");
			}
		}
		else
		{
			if(m_CurrentPath < m_pConfManager->GetConfVars()->NumPaths)
			{	
				if(m_ProfileFrame) cpuTimer.Start();
				int remaining = m_pConfManager->GetConfVars()->NumPaths - m_CurrentPath;
				if(remaining >= m_pConfManager->GetConfVars()->NumPathsPerFrame)
				{
					scene->CreatePaths(avpls, m_pConfManager->GetConfVars()->NumPathsPerFrame, 
						m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
					m_CurrentPath += m_pConfManager->GetConfVars()->NumPathsPerFrame;
				}
				else
				{
					 scene->CreatePaths(avpls, remaining, m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
					 m_CurrentPath += remaining;
				}
				if(m_ProfileFrame) std::cout << "num avpls: " << avpls.size() << std::endl;
				if(m_ProfileFrame) cpuTimer.Stop("Create AVPLs");
				
				if(m_pConfManager->GetConfVars()->GatherWithAVPLAtlas)
				{
					if(m_pConfManager->GetConfVars()->GatherWithAVPLClustering)
					{
						if(m_pConfManager->GetConfVars()->UseLightTree)
						{
							m_pLightTree->Release();		
							m_pLightTree->BuildTreeTweakNN(avpls, 0.f);
							m_pLightTree->Color(avpls, m_pConfManager->GetConfVars()->ClusterDepth);
						}
						else
						{
							m_pClusterTree->Release();		
							m_pClusterTree->BuildTree(avpls);
							m_pClusterTree->Color(avpls, m_pConfManager->GetConfVars()->ClusterDepth);
						}

						GatherWithClustering(avpls, m_pGatherRenderTarget);
					}
					else	
					{
						GatherWithAtlas(avpls, m_pGatherRenderTarget);
					}
				}
				else
				{	
					if(m_ProfileFrame) gpuTimer.Start();
					Gather(avpls, m_pGatherRenderTarget);
					if(m_ProfileFrame) gpuTimer.Stop("Gather");
				}
				
				if(m_CurrentPath >= m_pConfManager->GetConfVars()->NumPaths && !m_Finished)
				{
					std::cout << "Finished." << std::endl;
					m_Finished = true;
				}
			}
		}
	}	
	else
	{
		if(m_CurrentPath < m_pConfManager->GetConfVars()->NumPaths)
		{				
			int remaining = m_pConfManager->GetConfVars()->NumPaths - m_CurrentPath;
			if(remaining >= m_pConfManager->GetConfVars()->NumPathsPerFrame)
			{
				scene->CreatePaths(avpls, m_pConfManager->GetConfVars()->NumPathsPerFrame, 
					m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
				m_CurrentPath += m_pConfManager->GetConfVars()->NumPathsPerFrame;
			}
			else
			{
				 scene->CreatePaths(avpls, remaining, m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
				 m_CurrentPath += remaining;
			}

			{
				GatherRadianceWithShadowMap(avpls, m_pGatherRenderTarget);
			}
		}
		else{
			if(!m_Finished)
				std::cout << "Finished." << std::endl;

			m_Finished = true;
		}
	}
	
	if(m_ProfileFrame) gpuTimer.Start();
	int normFactor = std::min(m_CurrentPath, m_pConfManager->GetConfVars()->NumPaths * m_pConfManager->GetConfVars()->NumPathsPerFrame);
	Normalize(m_pNormalizeRenderTarget, m_pGatherRenderTarget, normFactor);

	Shade(m_pShadeRenderTarget, m_pNormalizeRenderTarget);
	if(m_ProfileFrame) gpuTimer.Stop("normalize and shade");

	if(m_pConfManager->GetConfVars()->DrawLights)
		DrawLights(avpls, m_pAVPLRenderTarget);

	if(m_ProfileFrame) cpuTimer.Start();
	avpls.clear();
	if(m_ProfileFrame) cpuTimer.Stop("delete avpls");

	if(m_pConfManager->GetConfVars()->DrawSceneSamples)
	{
		DrawSceneSamples(m_pShadeRenderTarget);
	}

	if(m_pConfManager->GetConfVars()->UseToneMapping)
	{
		m_pPostProcess->SetGamma(m_pConfManager->GetConfVars()->Gamma);
		m_pPostProcess->SetExposure(m_pConfManager->GetConfVars()->Exposure);
		m_pPostProcess->Postprocess(m_pShadeRenderTarget->GetBuffer(0), m_pPostProcessRenderTarget);
		m_pTextureViewer->DrawTexture(m_pPostProcessRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	else
	{
		m_pTextureViewer->DrawTexture(m_pShadeRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	
	m_Frame++;
	
	if(m_pConfManager->GetConfVars()->DrawCutSizes)
		m_pTextureViewer->DrawTexture(m_pGatherRenderTarget->GetBuffer(3), 0, 0, camera->GetWidth(), camera->GetHeight());

	if(m_pConfManager->GetConfVars()->DrawAVPLAtlas)
	{
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU)
			m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLAtlas(), 0, 0, camera->GetWidth(), camera->GetHeight());
		else
			m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLAtlasCPU(), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	
	if(m_pConfManager->GetConfVars()->DrawAVPLClusterAtlas)
	{
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU)
			m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLClusterAtlas(), 0, 0, camera->GetWidth(), camera->GetHeight());
		else
			m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLClusterAtlasCPU(), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	
	if(m_CurrentPath % 50000 == 0 && m_CurrentPath > 0 && !m_Finished)
	{
		double time = m_pGlobalTimer->GetTime();
		std::stringstream ss;
		ss << "result-" << m_CurrentPath << "paths-" << scene->GetNumCreatedAVPLs() << "c-avpls-" << scene->GetNumAVPLsAfterIS() << "u-avpls-" << time << "ms" << ".pfm";
		m_Export->ExportPFM(m_pShadeRenderTarget->GetBuffer(0), ss.str());
		
		std::cout << "# paths: " << m_CurrentPath << std::endl;
	}	

	if(m_pConfManager->GetConfVars()->DrawLights)
	{
		m_pTextureViewer->DrawTexture(m_pAVPLRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	}

	m_ProfileFrame = false;
}

void Renderer::RenderDirectIndirectLight() 
{	
	CTimer cpuTimer(CTimer::CPU);
	CTimer gpuTimer(CTimer::OGL);

	cpuTimer.Start();
	gpuTimer.Start();

	/*
		OpenCL test
	*/

	SetUpRender();
	
	if(m_CurrentPath == 0)
	{
		m_pGlobalTimer->Start();
		m_pOGLTimer->Start();
		CreateGBuffer();
		m_pOGLTimer->Stop("CreateGBuffer");
	}
	
	if(m_CurrentPath < m_pConfManager->GetConfVars()->NumPaths)
	{
		if(m_ProfileFrame) m_pCPUFrameProfiler->Start();
		
		std::vector<AVPL> avpls;
				
		int remaining = m_pConfManager->GetConfVars()->NumPaths - m_CurrentPath;
		if(remaining >= m_pConfManager->GetConfVars()->NumPathsPerFrame)
		{
			scene->CreatePaths(avpls, m_pConfManager->GetConfVars()->NumPathsPerFrame, 
				m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
			m_CurrentPath += m_pConfManager->GetConfVars()->NumPathsPerFrame;
		}
		else
		{
			scene->CreatePaths(avpls, remaining, m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
			m_CurrentPath += remaining;
		}
			
		std::vector<AVPL> indirect_avpls;
		indirect_avpls.reserve(avpls.size());
		
		for(int i = 0; i < avpls.size(); ++i)
		{
			AVPL avpl = avpls[i];
			
			if(avpl.GetBounce() != 0)
			{
				if(avpl.GetBounce() == 1)
					avpl.SetAntiintensity(glm::vec3(0.f));
				indirect_avpls.push_back(avpl);
			}
		}

		if(m_ProfileFrame) std::cout << "Profile Frame - Indirect Lighting - NumLights: " << indirect_avpls.size() << std::endl;
		if(m_ProfileFrame) m_pCPUFrameProfiler->Stop("Profile Frame - Indirect Lighting - Light Creation");
		
		if(indirect_avpls.size() > 0)
		{			
			if(m_pConfManager->GetConfVars()->GatherWithAVPLAtlas)
			{
				if(m_pConfManager->GetConfVars()->GatherWithAVPLClustering)
				{
					if(m_ProfileFrame) m_pCPUFrameProfiler->Start();
					if(m_pConfManager->GetConfVars()->UseLightTree)
					{
						m_pLightTree->Release();		
						m_pLightTree->BuildTreeTweakNN(indirect_avpls, 0.f);
						m_pLightTree->Color(indirect_avpls, m_pConfManager->GetConfVars()->ClusterDepth);
					}
					else
					{
						m_pClusterTree->Release();		
						m_pClusterTree->BuildTree(indirect_avpls);
						m_pClusterTree->Color(indirect_avpls, m_pConfManager->GetConfVars()->ClusterDepth);
					}
					if(m_ProfileFrame) m_pCPUFrameProfiler->Stop("Profile Frame - Indirect Lighting - Light Clustering");

					if(m_ProfileFrame) m_pGPUFrameProfiler->Start();
					GatherWithClustering(indirect_avpls, m_pGatherIndirectLightRenderTarget);
				}
				else	
				{
					if(m_ProfileFrame) m_pGPUFrameProfiler->Start();
					GatherWithAtlas(indirect_avpls, m_pGatherIndirectLightRenderTarget);
				}
			}
			else
			{
				if(m_ProfileFrame) m_pGPUFrameProfiler->Start();
				Gather(indirect_avpls, m_pGatherIndirectLightRenderTarget);
			}
		}
		
		if(m_pConfManager->GetConfVars()->DrawLights)
			DrawLights(avpls, m_pShadeRenderTarget);
						
		if(m_ProfileFrame) m_pGPUFrameProfiler->Stop("Profile Frame - Indirect Lighting - Gather");
				
		indirect_avpls.clear();	
		avpls.clear();

		if(m_CurrentPath >= m_pConfManager->GetConfVars()->NumPaths && !m_FinishedIndirectLighting)
		{
			std::cout << "Finished indirect lighting." << std::endl;
			m_FinishedIndirectLighting = true;
		}
	}	

	/* direct lighting */
	if(m_CurrentVPLDirect < m_pConfManager->GetConfVars()->NumVPLsDirectLight)
	{
		if(m_ProfileFrame) m_pCPUFrameProfiler->Start();
		
		std::vector<AVPL> direct_avpls;
		int remaining = m_pConfManager->GetConfVars()->NumVPLsDirectLight - m_CurrentVPLDirect;
		if(remaining >= m_pConfManager->GetConfVars()->NumVPLsDirectLightPerFrame)
		{
			scene->CreatePrimaryVpls(direct_avpls, m_pConfManager->GetConfVars()->NumVPLsDirectLightPerFrame);
			m_CurrentVPLDirect += m_pConfManager->GetConfVars()->NumVPLsDirectLightPerFrame;
		}
		else
		{
			 scene->CreatePrimaryVpls(direct_avpls, remaining);
			 m_CurrentVPLDirect += remaining;
		}
		if(m_ProfileFrame) std::cout << "Profile Frame - Direct Lighting - NumLights: " << direct_avpls.size() << std::endl;
		if(m_ProfileFrame) m_pCPUFrameProfiler->Stop("Profile Frame - Direct Lighting - Light Creation");

		if(m_ProfileFrame) m_pGPUFrameProfiler->Start();
		GatherRadianceWithShadowMap(direct_avpls, m_pGatherDirectLightRenderTarget);
		if(m_ProfileFrame) m_pGPUFrameProfiler->Stop("Profile Frame - Direct Lighting - Gather");

		direct_avpls.clear();
		
		if(m_CurrentVPLDirect >= m_pConfManager->GetConfVars()->NumVPLsDirectLight && !m_FinishedDirectLighting)
		{
			std::cout << "Finished direct lighting." << std::endl;
			m_FinishedDirectLighting = true;
		}
	}	
	
	int normFactorDirect = std::min(m_CurrentVPLDirect, m_pConfManager->GetConfVars()->NumVPLsDirectLight * m_pConfManager->GetConfVars()->NumVPLsDirectLightPerFrame);
	int normFactorIndirect = std::min(m_CurrentPath, m_pConfManager->GetConfVars()->NumPaths * m_pConfManager->GetConfVars()->NumPathsPerFrame);
	
	Normalize(m_pNormalizeDirectLightRenderTarget, m_pGatherDirectLightRenderTarget, normFactorDirect);
	Normalize(m_pNormalizeIndirectLightRenderTarget, m_pGatherIndirectLightRenderTarget, normFactorIndirect);
	
	Shade(m_pShadeDirectLightRenderTarget, m_pNormalizeDirectLightRenderTarget);
	Shade(m_pShadeIndirectLightRenderTarget, m_pNormalizeIndirectLightRenderTarget);
	
	Add(m_pShadeRenderTarget, m_pShadeDirectLightRenderTarget, m_pShadeIndirectLightRenderTarget);
	
	m_pTextureViewer->DrawTexture(m_pShadeRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	
	if(m_pConfManager->GetConfVars()->DrawDirectLight)
	{
		m_pTextureViewer->DrawTexture(m_pShadeDirectLightRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	else if(m_pConfManager->GetConfVars()->DrawIndirectLight)
	{
		m_pTextureViewer->DrawTexture(m_pShadeIndirectLightRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	}

	if(m_pConfManager->GetConfVars()->DrawAVPLAtlas)
	{
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU)
			m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLAtlas(), 0, 0, camera->GetWidth(), camera->GetHeight());
		else
			m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLAtlasCPU(), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	
	if(m_pConfManager->GetConfVars()->DrawAVPLClusterAtlas)
	{
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU)
			m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLClusterAtlas(), 0, 0, camera->GetWidth(), camera->GetHeight());
		else
			m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLClusterAtlasCPU(), 0, 0, camera->GetWidth(), camera->GetHeight());
	}

	if(m_CurrentPath % 100000 == 0 && m_CurrentPath > 0 && !m_FinishedIndirectLighting)
	{
		double time = m_pGlobalTimer->GetTime();
		std::stringstream ss1, ss2, ss3;
		ss1 << "result-" << m_CurrentPath << "paths-" << time << "ms" << ".pfm";
		ss2 << "result-direct-" << m_CurrentPath << "paths-" << time << "ms" << ".pfm";
		ss3 << "result-indirect-" << m_CurrentPath << "paths-" << time << "ms" << ".pfm";
		m_Export->ExportPFM(m_pShadeRenderTarget->GetBuffer(0), ss1.str());
		m_Export->ExportPFM(m_pShadeDirectLightRenderTarget->GetBuffer(0), ss2.str());
		m_Export->ExportPFM(m_pShadeIndirectLightRenderTarget->GetBuffer(0), ss3.str());

		std::cout << "# paths: " << m_CurrentPath << ", time: " << time << "ms" << std::endl;
	}

	m_ProfileFrame = false;
}

void Renderer::SetUpRender()
{
	glViewport(0, 0, (GLsizei)scene->GetCamera()->GetWidth(), (GLsizei)scene->GetCamera()->GetHeight());

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);

	CAMERA camera;
	camera.positionWS = scene->GetCamera()->GetPosition();
	camera.width = (int)scene->GetCamera()->GetWidth();
	camera.height = (int)scene->GetCamera()->GetHeight();
	m_pUBCamera->UpdateData(&camera);
}

void Renderer::CreateGBuffer()
{
	GLenum buffers [3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
	COGLRenderTargetLock lockRenderTarget(m_pGBuffer->GetRenderTarget(), 3, buffers);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	COGLBindLock lockProgram(m_pCreateGBufferProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	scene->DrawScene(m_pUBTransform, m_pUBMaterial);
}

void Renderer::GatherRadianceWithShadowMap(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget)
{
	for(int i = 0; i < path.size(); ++i)
	{
		GatherRadianceFromLightWithShadowMap(path[i], pRenderTarget);
	}
}

void Renderer::Normalize(CRenderTarget* pTarget, CRenderTarget* source, int normFactor)
{
	CONFIG conf;
	conf.GeoTermLimit = m_pConfManager->GetConfVars()->GeoTermLimit;
	conf.ClampGeoTerm = m_pConfManager->GetConfVars()->ClampGeoTerm;
	conf.nPaths = normFactor;
	conf.N = m_pConfManager->GetConfVars()->ConeFactor;
	conf.AntiradFilterK = GetAntiradFilterNormFactor();
	conf.AntiradFilterMode = m_pConfManager->GetConfVars()->AntiradFilterMode;
	conf.AntiradFilterGaussFactor = m_pConfManager->GetConfVars()->AntiradFilterGaussFactor;
	m_pUBConfig->UpdateData(&conf);
	
	{
		CRenderTargetLock lock(pTarget);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		COGLBindLock lockProgram(m_pNormalizeProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(source->GetBuffer(0), COGL_TEXTURE0_SLOT);
		COGLBindLock lock1(source->GetBuffer(1), COGL_TEXTURE1_SLOT);
		COGLBindLock lock2(source->GetBuffer(2), COGL_TEXTURE2_SLOT);

		m_pFullScreenQuad->Draw();
	}
}

void Renderer::GatherRadianceFromLightWithShadowMap(const AVPL& avpl, CRenderTarget* pRenderTarget)
{
	FillShadowMap(avpl);
	
	if(avpl.GetIntensity(avpl.GetOrientation()).length() == 0.f)
		return;
	
	if(avpl.GetBounce() != m_pConfManager->GetConfVars()->RenderBounce && m_pConfManager->GetConfVars()->RenderBounce != -1)
		return;
	
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
		
	COGLBindLock lockProgram(m_pGatherRadianceWithSMProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	AVPL_STRUCT light_info;
	avpl.Fill(light_info);
	m_pUBLight->UpdateData(&light_info);

	CRenderTargetLock lock(pRenderTarget);

	COGLBindLock lock0(m_pShadowMap->GetShadowMapTexture(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_pGBuffer->GetPositionTextureWS(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_pGBuffer->GetNormalTexture(), COGL_TEXTURE2_SLOT);
	
	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::DetermineUsedAvpls(const std::vector<AVPL>& avpls, std::vector<AVPL>& used)
{
	// determine avpls used for lighting with m_RenderBounce config
	if(m_pConfManager->GetConfVars()->RenderBounce == -1)
	{
		// use all avpls
		used = avpls;
	}
	else
	{
		// collect all avpls contributing to the illumination
		for(uint i = 0; i < avpls.size(); ++i)
		{
			AVPL avpl = avpls[i];
			if(avpl.GetBounce() != m_pConfManager->GetConfVars()->RenderBounce + 1) avpl.SetAntiintensity(glm::vec3(0.f));
			if(avpl.GetBounce() != m_pConfManager->GetConfVars()->RenderBounce) avpl.SetIntensity(glm::vec3(0.f));
			used.push_back(avpl);
		}
	}
}

void Renderer::Gather(const std::vector<AVPL>& avpls, CRenderTarget* pRenderTarget)
{
	try
	{
		AVPL_BUFFER* avplBuffer = new AVPL_BUFFER[avpls.size()];
		memset(avplBuffer, 0, sizeof(AVPL_BUFFER) * avpls.size());
		for(uint i = 0; i < avpls.size(); ++i)
		{
			AVPL_BUFFER buffer;
			avpls[i].Fill(buffer);
			avplBuffer[i] = buffer;
		}
		m_pOGLLightBuffer->SetContent(avplBuffer, sizeof(AVPL_BUFFER) * avpls.size());
		delete [] avplBuffer;
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at Renderer::Gather()" << std::endl;
		return;
	}
		
	INFO info;
	info.numLights = (int)avpls.size();
	info.filterAVPLAtlas = m_pConfManager->GetConfVars()->FilterAvplAtlasLinear;
	info.drawLightingOfLight = m_pConfManager->GetConfVars()->DrawLightingOfLight;
	m_pUBInfo->UpdateData(&info);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	
	COGLBindLock lockProgram(m_pGatherProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	CRenderTargetLock lock(pRenderTarget);

	COGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_pGBuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_pOGLLightBuffer, COGL_TEXTURE2_SLOT);

	m_pFullScreenQuad->Draw();

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::GatherWithAtlas(const std::vector<AVPL>& avpls, CRenderTarget* pRenderTarget)
{	
	if((int)avpls.size() > m_MaxNumAVPLs)
		std::cout << "To many avpls. Some are not considered" << std::endl;

	try
	{
		AVPL_BUFFER* avplBuffer = new AVPL_BUFFER[avpls.size()];
		memset(avplBuffer, 0, sizeof(AVPL_BUFFER) * avpls.size());
		for(uint i = 0; i < avpls.size(); ++i)
		{
			AVPL_BUFFER buffer;
			avpls[i].Fill(buffer);
			avplBuffer[i] = buffer;
		}
		m_pOGLLightBuffer->SetContent(avplBuffer, sizeof(AVPL_BUFFER) * avpls.size());
	
		m_pOctahedronAtlas->Clear();

		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU)
		{
			m_pOCLTimer->Start();
			m_pOctahedronAtlas->FillAtlasGPU(avplBuffer, (int)avpls.size(), m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
				float(m_pConfManager->GetConfVars()->ConeFactor), m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
			m_pOCLTimer->Stop("FillAtlasGPU");
		}
		else
		{
			m_pOctahedronAtlas->FillAtlas(avpls, m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
				float(m_pConfManager->GetConfVars()->ConeFactor), m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
		}
		
		delete [] avplBuffer;

		INFO info;
		info.numLights = (int)avpls.size();
		info.drawLightingOfLight = m_pConfManager->GetConfVars()->DrawLightingOfLight;
		info.filterAVPLAtlas = m_pConfManager->GetConfVars()->FilterAvplAtlasLinear;
		m_pUBInfo->UpdateData(&info);

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);	
		glBlendFunc(GL_ONE, GL_ONE);
		
		CRenderTargetLock lockRenderTarget(pRenderTarget);

		COGLBindLock lockProgram(m_pGatherWithAtlas->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
		COGLBindLock lock1(m_pGBuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
		COGLBindLock lock2(m_pOGLLightBuffer, COGL_TEXTURE2_SLOT);

		if(m_pConfManager->GetConfVars()->FilterAvplAtlasLinear)
		{
			glBindSampler(3, m_pGLLinearSampler->GetResourceIdentifier());
		}
		else
		{
			glBindSampler(3, m_pGLPointSampler->GetResourceIdentifier());
		}
		
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU == 1)
		{
			COGLBindLock lock3(m_pOctahedronAtlas->GetAVPLAtlas(), COGL_TEXTURE3_SLOT);			
			m_pFullScreenQuad->Draw();
		}
		else
		{
			COGLBindLock lock3(m_pOctahedronAtlas->GetAVPLAtlasCPU(), COGL_TEXTURE3_SLOT);
			m_pFullScreenQuad->Draw();
		}
				
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);

		glBindSampler(3, m_pGLPointSampler->GetResourceIdentifier());
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at Renderer::GatherWithAtlas()" << std::endl;
		return;
	}
}

void Renderer::GatherWithClustering(const std::vector<AVPL>& avpls, CRenderTarget* pRenderTarget)
{
	if((int)avpls.size() > m_MaxNumAVPLs)
		std::cout << "To many avpls. Some are not considered" << std::endl;

	try
	{
		// fill light information
		AVPL_BUFFER* avplBuffer = new AVPL_BUFFER[avpls.size()];
		memset(avplBuffer, 0, sizeof(AVPL_BUFFER) * avpls.size());
		for(uint i = 0; i < avpls.size(); ++i)
		{
			AVPL_BUFFER buffer;
			avpls[i].Fill(buffer);
			avplBuffer[i] = buffer;
		}
		m_pOGLLightBuffer->SetContent(avplBuffer, sizeof(AVPL_BUFFER) * avpls.size());
	
		// fill cluster information
		CLUSTER* clustering;
		int clusteringSize;
		if(m_pConfManager->GetConfVars()->UseLightTree)
		{
			clustering = m_pLightTree->GetClustering();
			clusteringSize = m_pLightTree->GetClusteringSize();
		}
		else
		{
			clustering = m_pClusterTree->GetClustering();
			clusteringSize = m_pClusterTree->GetClusteringSize();
		}

		CLUSTER_BUFFER* clusterBuffer = new CLUSTER_BUFFER[clusteringSize];
		memset(clusterBuffer, 0, sizeof(CLUSTER_BUFFER) * clusteringSize);
		for(int i = 0; i < clusteringSize; ++i)
		{
			CLUSTER_BUFFER buffer;
			clustering[i].Fill(&buffer);
			clusterBuffer[i] = buffer;
		}
		m_pOGLClusterBuffer->SetContent(clusterBuffer, sizeof(CLUSTER_BUFFER) * clusteringSize);
		
		// create the avpl and cluster atlas
		m_pOctahedronAtlas->Clear();
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU)
		{
			m_pOctahedronAtlas->FillAtlasGPU(avplBuffer, (int)avpls.size(), m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
				float(m_pConfManager->GetConfVars()->ConeFactor), m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
			
			if(m_pConfManager->GetConfVars()->UseLightTree)
				m_pOctahedronAtlas->FillClusterAtlasGPU(m_pLightTree->GetClustering(), m_pLightTree->GetClusteringSize(), (int)avpls.size());
			else
				m_pOctahedronAtlas->FillClusterAtlasGPU(m_pClusterTree->GetClustering(), m_pClusterTree->GetClusteringSize(), (int)avpls.size());
		}
		else
		{
			m_pOctahedronAtlas->FillAtlas(avpls, m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
				float(m_pConfManager->GetConfVars()->ConeFactor), m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
			if(m_pConfManager->GetConfVars()->UseLightTree)
				m_pOctahedronAtlas->FillClusterAtlas(avpls, m_pLightTree->GetClustering(), m_pLightTree->GetClusteringSize());
			else
				m_pOctahedronAtlas->FillClusterAtlas(avpls, m_pClusterTree->GetClustering(), m_pClusterTree->GetClusteringSize());
		}
		
		delete [] avplBuffer;
		delete [] clusterBuffer;

		INFO info;
		info.numLights = (int)avpls.size();

		if(m_pConfManager->GetConfVars()->UseLightTree)
			info.numClusters = m_pLightTree->GetClusteringSize();
		else
			info.numClusters = m_pClusterTree->GetClusteringSize();
		
		info.drawLightingOfLight = m_pConfManager->GetConfVars()->DrawLightingOfLight;
		info.filterAVPLAtlas = m_pConfManager->GetConfVars()->FilterAvplAtlasLinear;
		info.lightTreeCutDepth = m_pConfManager->GetConfVars()->LightTreeCutDepth;
		info.clusterRefinementThreshold = m_pConfManager->GetConfVars()->ClusterRefinementThreshold;
		m_pUBInfo->UpdateData(&info);

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);	
		glBlendFunc(GL_ONE, GL_ONE);
		
		CRenderTargetLock lockRenderTarget(pRenderTarget);

		COGLBindLock lockProgram(m_pGatherWithClustering->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
		COGLBindLock lock1(m_pGBuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
		COGLBindLock lock2(m_pOGLLightBuffer, COGL_TEXTURE2_SLOT);
		COGLBindLock lock3(m_pOGLClusterBuffer, COGL_TEXTURE3_SLOT);

		if(m_pConfManager->GetConfVars()->FilterAvplAtlasLinear)
		{
			glBindSampler(4, m_pGLLinearSampler->GetResourceIdentifier());
			glBindSampler(5, m_pGLLinearSampler->GetResourceIdentifier());
		}
		else
		{
			glBindSampler(4, m_pGLPointSampler->GetResourceIdentifier());
			glBindSampler(5, m_pGLPointSampler->GetResourceIdentifier());
		}
		
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU == 1)
		{
			COGLBindLock lock4(m_pOctahedronAtlas->GetAVPLAtlas(), COGL_TEXTURE4_SLOT);
			COGLBindLock lock5(m_pOctahedronAtlas->GetAVPLClusterAtlas(), COGL_TEXTURE5_SLOT);
	
			m_pFullScreenQuad->Draw();
		}
		else
		{
			COGLBindLock lock4(m_pOctahedronAtlas->GetAVPLAtlasCPU(), COGL_TEXTURE4_SLOT);
			COGLBindLock lock5(m_pOctahedronAtlas->GetAVPLClusterAtlasCPU(), COGL_TEXTURE5_SLOT);
			
			m_pFullScreenQuad->Draw();
		}
				
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);

		glBindSampler(4, m_pGLPointSampler->GetResourceIdentifier());
		glBindSampler(5, m_pGLPointSampler->GetResourceIdentifier());
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at Renderer::GatherWithAtlas()" << std::endl;
		return;
	}
}

void Renderer::FillShadowMap(const AVPL& avpl)
{
	glEnable(GL_DEPTH_TEST);

	// prevent surface acne
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.1f, 4.0f);
	
	COGLBindLock lockProgram(m_pCreateSMProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
	GLenum buffer[1] = {GL_NONE};
	COGLRenderTargetLock lock(m_pShadowMap->GetRenderTarget(), 1, buffer);

	glViewport(0, 0, m_pShadowMap->GetShadowMapSize(), m_pShadowMap->GetShadowMapSize());
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	scene->DrawScene(avpl.GetViewMatrix(), avpl.GetProjectionMatrix(), m_pUBTransform);

	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::Shade(CRenderTarget* target, CRenderTarget* source)
{
	CRenderTargetLock lock(target);
	
	{
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		COGLBindLock lockProgram(m_pShadeProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(source->GetBuffer(0), COGL_TEXTURE0_SLOT);
		COGLBindLock lock2(m_pGBuffer->GetMaterialTexture(), COGL_TEXTURE1_SLOT);

		m_pFullScreenQuad->Draw();
		
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
	}

	DrawAreaLight();

	DebugRender();
}

void Renderer::DrawAreaLight()
{
	COGLBindLock lock(m_pAreaLightProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
	
	// avoid z-fighting
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(-1.0f, 1.0f);
	glDepthFunc(GL_LEQUAL);

	scene->DrawAreaLight(m_pUBTransform, m_pUBAreaLight);
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::DrawLights(const std::vector<AVPL>& avpls, CRenderTarget* target)
{	
	std::vector<POINT_CLOUD_POINT> pcp;
	
	for (int i = 0; i < avpls.size(); ++i)
	{
		int rb = m_pConfManager->GetConfVars()->RenderBounce;
		if(rb == -1 || (avpls[i].GetBounce() == rb || avpls[i].GetBounce() == rb + 1))
		{
			POINT_CLOUD_POINT p;
			glm::vec3 pos = avpls[i].GetPosition();
			p.position = glm::vec4(pos, 1.0f);
			p.color = glm::vec4(avpls[i].GetColor(), 1.f);
			pcp.push_back(p);
		}
	}
	
	try
	{
		glm::vec4* positionData = new glm::vec4[pcp.size()];
		glm::vec4* colorData = new glm::vec4[pcp.size()];
		for(uint i = 0; i < pcp.size(); ++i)
		{
			positionData[i] = pcp[i].position;
			colorData[i] = pcp[i].color;
		}
		
		{
			CRenderTargetLock lock(target);
			
			COGLBindLock lockProgram(m_pPointCloudProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
			
			TRANSFORM transform;
			transform.M = IdentityMatrix();
			transform.V = camera->GetViewMatrix();
			transform.itM = IdentityMatrix();
			transform.MVP = camera->GetProjectionMatrix() * camera->GetViewMatrix();
			m_pUBTransform->UpdateData(&transform);
		
			glDepthMask(GL_FALSE);
			glDisable(GL_DEPTH_TEST);
			m_pPointCloud->Draw(positionData, colorData, (int)pcp.size());		
			glDepthMask(GL_TRUE);
			glEnable(GL_DEPTH_TEST);
		}
		
		pcp.clear();
		delete [] positionData;
		delete [] colorData;
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at Renderer::DrawLights()" << std::endl;
	}
}

void Renderer::DrawSceneSamples(CRenderTarget* target)
{	
	std::vector<SceneSample> sceneSamples = m_pAVPLImportanceSampling->GetSceneSamples();
	std::vector<POINT_CLOUD_POINT> pcp;
	
	for (int i = 0; i < sceneSamples.size(); ++i)
	{
		POINT_CLOUD_POINT p;
		p.position = glm::vec4(sceneSamples[i].position, 1.0f);
		p.color = glm::vec4(0.5f * sceneSamples[i].normal + glm::vec3(0.5f), 1.0f);
		pcp.push_back(p);
	}
	
	try
	{
		glm::vec4* positionData = new glm::vec4[pcp.size()];
		glm::vec4* colorData = new glm::vec4[pcp.size()];
		for(uint i = 0; i < pcp.size(); ++i)
		{
			positionData[i] = pcp[i].position;
			colorData[i] = pcp[i].color;
		}
		
		{
			CRenderTargetLock lock(target);
			
			COGLBindLock lockProgram(m_pPointCloudProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
			
			TRANSFORM transform;
			transform.M = IdentityMatrix();
			transform.V = camera->GetViewMatrix();
			transform.itM = IdentityMatrix();
			transform.MVP = camera->GetProjectionMatrix() * camera->GetViewMatrix();
			m_pUBTransform->UpdateData(&transform);
		
			glDepthMask(GL_FALSE);
			m_pPointCloud->Draw(positionData, colorData, (int)pcp.size());		
			glDepthMask(GL_TRUE);
		}
		
		pcp.clear();
		delete [] positionData;
		delete [] colorData;
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at Renderer::DrawLights()" << std::endl;
	}
}

void Renderer::Add(CRenderTarget* target, CRenderTarget* source1, CRenderTarget* source2)
{
	CRenderTargetLock lock(target);
	
	{
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		COGLBindLock lockProgram(m_pAddProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(source1->GetBuffer(0), COGL_TEXTURE0_SLOT);
		COGLBindLock lock1(source2->GetBuffer(0), COGL_TEXTURE1_SLOT);

		m_pFullScreenQuad->Draw();
		
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
	}
}

void Renderer::DebugRender()
{
	// draw gbuffer info for debuging
	if(m_pConfManager->GetConfVars()->DrawDebugTextures) 
	{
		int border = 10;
		int width = (camera->GetWidth() - 4 * border) / 2;
		int height = (camera->GetHeight() - 4 * border) / 2;
		m_pTextureViewer->DrawTexture(m_pNormalizeRenderTarget->GetBuffer(0),  border, border, border+width, border+height);
		m_pTextureViewer->DrawTexture(m_pNormalizeRenderTarget->GetBuffer(1),  3 * border + width, border, 3 * border + 2 * width, border+height);
		m_pTextureViewer->DrawTexture(m_pNormalizeRenderTarget->GetBuffer(2),  border, 3 * border + height, border+width, 3 * border+ 2 * height);
	}
}

void Renderer::WindowChanged()
{
	m_pDepthBuffer->Release();
	m_pDepthBuffer->Init(camera->GetWidth(), camera->GetHeight(), GL_DEPTH_COMPONENT32F, 
		GL_DEPTH, GL_FLOAT, 1, false);

	m_pGBuffer->Release();
	m_pGBuffer->Init(camera->GetWidth(), camera->GetHeight(), m_pDepthBuffer);
	
	ClearAccumulationBuffer();
}

void Renderer::ClearAccumulationBuffer()
{
	glClearColor(0, 0, 0, 0);
	
	{
		CRenderTargetLock lock(m_pLightDebugRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pGatherRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pNormalizeRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pGatherDirectLightRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pGatherIndirectLightRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pNormalizeDirectLightRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pNormalizeIndirectLightRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pShadeDirectLightRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pShadeIndirectLightRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pAVPLRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	m_CurrentPath = 0;
	m_CurrentVPLDirect = 0;
	m_Finished = false;
	m_FinishedDirectLighting = false;
	m_FinishedIndirectLighting = false;
}

void Renderer::ClearLighting()
{
	scene->ClearLighting();
	m_CurrentPath = 0;
	m_CurrentVPLDirect = 0;
	m_Finished = false;
	m_FinishedDirectLighting = false;
	m_FinishedIndirectLighting = false;
		
	ClearAccumulationBuffer();
}

void Renderer::Stats() 
{
}

void Renderer::ConfigureLighting()
{
	CONFIG conf;
	conf.GeoTermLimit = m_pConfManager->GetConfVars()->GeoTermLimit;
	conf.ClampGeoTerm = m_pConfManager->GetConfVars()->ClampGeoTerm;
	conf.nPaths = m_CurrentPath;	
	conf.N = m_pConfManager->GetConfVars()->ConeFactor;
	conf.AntiradFilterK = GetAntiradFilterNormFactor();
	conf.AntiradFilterMode = m_pConfManager->GetConfVars()->AntiradFilterMode;
	conf.AntiradFilterGaussFactor = m_pConfManager->GetConfVars()->AntiradFilterGaussFactor;
	m_pUBConfig->UpdateData(&conf);
}

void Renderer::PrintCameraConfig()
{
	std::cout << "Camera: " << std::endl;
	std::cout << "position: (" << scene->GetCamera()->GetPosition().x << ", " 
		<< scene->GetCamera()->GetPosition().y << ", " 
		<< scene->GetCamera()->GetPosition().z << ")" << std::endl;
}

glm::vec4 Renderer::ColorForLight(const AVPL& avpl)
{
	glm::vec4 color;
	switch(avpl.GetBounce()){
		case 0: color = glm::vec4(0.8f, 0.8f, 0.8f, 1.f); break;
		case 1: color = glm::vec4(0.8f, 0.0f, 0.0f, 1.f); break;
		case 2: color = glm::vec4(0.0f, 0.8f, 0.0f, 1.f); break;
		case 3: color = glm::vec4(0.0f, 0.0f, 0.8f, 1.f); break;
		case 4: color = glm::vec4(0.8f, 0.8f, 0.0f, 1.f); break;
		case 5: color = glm::vec4(0.8f, 0.0f, 0.8f, 1.f); break;
		case 6: color = glm::vec4(0.0f, 0.8f, 0.8f, 1.f); break;
		default: color = glm::vec4(0.2f, 0.2f, 0.2f, 1.f); break;
	}
	return color;
}

void Renderer::Export()
{
	m_Export->ExportPFM(m_pNormalizeRenderTarget->GetBuffer(0), "result.pfm");
	m_Export->ExportPFM(m_pNormalizeRenderTarget->GetBuffer(1), "radiance.pfm");
	m_Export->ExportPFM(m_pNormalizeRenderTarget->GetBuffer(2), "antiradiance.pfm");
	m_Export->ExportPFM(m_pShadeRenderTarget->GetBuffer(0), "shade_result.pfm");
	m_Export->ExportPFM(m_pLightDebugRenderTarget->GetBuffer(0), "lights.pfm");
	m_Export->ExportPFM(m_pPostProcessRenderTarget->GetBuffer(0), "pp_result.pfm");
}

void Renderer::ExportPartialResult()
{
	time_t end;
	time(&end);
	double diff = difftime(end, m_StartTime);
	int seconds = (int)diff;
	std::stringstream ss;
	ss << "result" << m_Frame << "frames-" << seconds <<" sec " << "-paths-" << m_CurrentPath << ".pfm";
	m_Export->ExportPFM(m_pShadeRenderTarget->GetBuffer(0), ss.str());
}

void Renderer::NewDebugLights()
{
	ClearAccumulationBuffer();
		
	m_DebugAVPLs.clear();
	
	InitDebugLights();
}

void Renderer::InitDebugLights()
{
	m_DebugAVPLs.clear();
	
	m_pAVPLImportanceSampling->UpdateCurrentIrradiance(m_pNormalizeRenderTarget->GetBuffer(1));
	m_pAVPLImportanceSampling->UpdateCurrentAntiirradiance(m_pNormalizeRenderTarget->GetBuffer(2));
	m_pAVPLImportanceSampling->SetNumberOfSceneSamples(m_pConfManager->GetConfVars()->NumSceneSamples);
	m_pAVPLImportanceSampling->CreateSceneSamples();

	m_pCPUTimer->Start();
	scene->CreatePaths(m_DebugAVPLs, m_pConfManager->GetConfVars()->NumPaths, 
		m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
	std::cout << "Number of AVPLs: " << m_DebugAVPLs.size() << std::endl;
	m_pCPUTimer->Stop("CreatePaths");
}

void Renderer::SetConfigManager(CConfigManager* pConfManager)
{
	m_pConfManager = pConfManager;
}

void Renderer::UpdateAreaLights()
{
	scene->UpdateAreaLights();
}

float Renderer::GetAntiradFilterNormFactor()
{
	float N = float(m_pConfManager->GetConfVars()->ConeFactor);
	float K = 0.f;
	float a = 1-cos(PI/N);
	if(m_pConfManager->GetConfVars()->AntiradFilterMode == 1)
	{		
		float b = 2*(N/PI*sin(PI/N)-1);
		K = - a / b;
	}
	else if(m_pConfManager->GetConfVars()->AntiradFilterMode == 2)
	{
		float b = IntegrateGauss();
		K = a / b;
	}

	m_pConfManager->GetConfVars()->AntiradFilterK = K;

	return K;
}

float Renderer::IntegrateGauss()
{
	const float N = float(m_pConfManager->GetConfVars()->ConeFactor);
	const float M = float(m_pConfManager->GetConfVars()->AntiradFilterGaussFactor);
	const float s = PI / (M*N);
	const int numSteps = 1000;
	const float stepSize = PI / (numSteps * N);
	
	float res = 0.f;
	for(int i = 0; i < numSteps; ++i)
	{
		const float t = stepSize * (float(i) + 0.5f);
		const float val = (glm::exp(-(t*t)/(s*s)) - glm::exp(-(M*M))) * glm::sin(t);
		res += val * stepSize;
	}

	return res;
}

void Renderer::CancelRender()
{
	m_Finished = true; m_CurrentPath = m_pConfManager->GetConfVars()->NumPaths;
}
