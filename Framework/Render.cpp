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

	m_pGatherProgram = new CProgram("Renderer.m_pGatherProgram", "Shaders\\Gather.vert", "Shaders\\Gather.frag");
	m_pGatherWithAtlas = new CProgram("Renderer.m_pGatherProgram", "Shaders\\Gather.vert", "Shaders\\GatherWithAtlas.frag");
	m_pGatherWithClustering = new CProgram("Renderer.m_pGatherProgram", "Shaders\\Gather.vert", "Shaders\\GatherWithClustering.frag");
	m_pNormalizeProgram = new CProgram("Renderer.m_pNormalizeProgram", "Shaders\\Gather.vert", "Shaders\\Normalize.frag");
	m_pShadeProgram = new CProgram("Renderer.m_pShadeProgram", "Shaders\\Gather.vert", "Shaders\\Shade.frag");

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

	m_Finished = false;
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
		
	SAFE_DELETE(m_pGatherProgram);
	SAFE_DELETE(m_pGatherWithAtlas);
	SAFE_DELETE(m_pGatherWithClustering);
	SAFE_DELETE(m_pNormalizeProgram);
	SAFE_DELETE(m_pShadeProgram);
	
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
	
	SAFE_DELETE(m_pAVPLPositions);

	SAFE_DELETE(m_pOGLLightBuffer);
	SAFE_DELETE(m_pOGLClusterBuffer);
	
	SAFE_DELETE(m_pOctahedronAtlas);
	SAFE_DELETE(m_pOctahedronMap);

	SAFE_DELETE(m_pCPUTimer);
	SAFE_DELETE(m_pOCLTimer);
	SAFE_DELETE(m_pOGLTimer);

	SAFE_DELETE(m_pCLContext);
}

bool Renderer::Init() 
{	
	V_RET_FOF(m_pCLContext->Init(m_pOGLContext));
		
	V_RET_FOF(m_pDepthBuffer->Init(camera->GetWidth(), camera->GetHeight(), GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, false));
	
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
	
	V_RET_FOF(m_pGatherRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 4, 0));
	V_RET_FOF(m_pNormalizeRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 3, 0));
	V_RET_FOF(m_pShadeRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));

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
	
	V_RET_FOF(m_pDrawOctahedronProgram->Init());
	m_pDrawOctahedronProgram->BindSampler(0, m_pGLPointSampler);
	m_pDrawOctahedronProgram->BindUniformBuffer(m_pUBTransform, "transform");
	m_pDrawOctahedronProgram->BindUniformBuffer(m_pUBModel, "model");
	
	V_RET_FOF(m_pFullScreenQuad->Init());
	
	V_RET_FOF(m_pShadowMap->Init(512));
	
	V_RET_FOF(m_pGBuffer->Init(camera->GetWidth(), camera->GetHeight(), m_pDepthBuffer));
		
	scene = new Scene(camera);
	scene->Init();
	scene->LoadCornellBox();
		
	ConfigureLighting();

	time(&m_StartTime);

	int dim_atlas = 4096;
	int dim_tile = 32;
		
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

	InitDebugLights();
		
	ClearAccumulationBuffer();
#if 1
	for(int i = 0; i < 30000; ++i)
	{
		glm::vec3 pos = glm::vec3(1000 * Rand01(), 1000 * Rand01(), 400 * Rand01()); 
		AVPL* avpl = new AVPL(pos, glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f), glm::vec3(0.f, 0.f, -1.f), 0);
		m_ClusterTestAVPLs.push_back(avpl);
	}
#else	
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(0.f, 2.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(1.f, 2.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(0.f, 0.f, 4.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(1.f, 0.f, 4.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(0.f, 2.f, 4.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(1.f, 2.f, 4.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(9.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(10.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(9.f, 2.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(10.f, 2.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(9.f, 0.f, 4.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(10.f, 0.f, 4.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(9.f, 2.f, 4.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	m_ClusterTestAVPLs.push_back(new AVPL(100.f * glm::vec3(10.f, 2.f, 4.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 0.f, -1.f), 0));
	
#endif
	/*
	camera->Init(glm::vec3(500.f, 500.f, -1500.f), glm::vec3(500.f, 500.f, 200.f), 1.f);
	
	CTimer timer(CTimer::CPU);
	timer.Start();

	m_pClusterTree->BuildTree(m_ClusterTestAVPLs);
	m_pClusterTree->Color(m_ClusterTestAVPLs, m_pConfManager->GetConfVars()->ClusterDepth);

	timer.Stop();
	std::cout << "ClusterTree build with " << m_ClusterTestAVPLs.size() << " lights took " << timer.GetTime() << "ms" << std::endl;
	*/
	return true;
}

void Renderer::Release()
{
	CheckGLError("CDSRenderer", "CDSRenderer::Release()");

	m_pDepthBuffer->Release();

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

	m_pGatherRenderTarget->Release();
	m_pNormalizeRenderTarget->Release();
	m_pShadeRenderTarget->Release();

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

	for(size_t i = 0; i < m_ClusterTestAVPLs.size(); ++i)
	{
		delete m_ClusterTestAVPLs[i];
	}
	m_ClusterTestAVPLs.clear();
}

void Renderer::ClusteringTestRender()
{	
	SetUpRender();
	
	{
		CRenderTargetLock lock(m_pShadeRenderTarget);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	}

	DrawLights(m_ClusterTestAVPLs);

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

	SetUpRender();
	
	if(m_CurrentPath == 0)
	{
		m_pOGLTimer->Start();
		CreateGBuffer();
		m_pOGLTimer->Stop("CreateGBuffer");
	}
	
	if(m_pConfManager->GetConfVars()->UseAntiradiance)
	{
		if(m_pConfManager->GetConfVars()->UseDebugMode)
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
						GatherWithClustering(m_DebugAVPLs);
						m_pOGLTimer->Stop("GatherWithClustering");
					}
					else	
					{
						m_pOGLTimer->Start();
						GatherWithAtlas(m_DebugAVPLs);
						m_pOGLTimer->Stop("GatherWithAtlas");						
					}
				}
				else
				{
					m_pOGLTimer->Start();
					Gather(m_DebugAVPLs);
					m_pOGLTimer->Stop("Gather");	
				}

				if(m_pConfManager->GetConfVars()->DrawLights)
					DrawLights(m_DebugAVPLs);

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
				std::vector<AVPL*> avpls;
				
				int remaining = m_pConfManager->GetConfVars()->NumPaths - m_CurrentPath;
				if(remaining >= m_pConfManager->GetConfVars()->NumPathsPerFrame)
				{
					avpls = scene->CreatePaths(m_pConfManager->GetConfVars()->NumPathsPerFrame, 
						m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
					m_CurrentPath += m_pConfManager->GetConfVars()->NumPathsPerFrame;
				}
				else
				{
					 avpls = scene->CreatePaths(remaining, m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
					 m_CurrentPath += remaining;
				}
				
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

						GatherWithClustering(avpls);
					}
					else	
					{
						GatherWithAtlas(avpls);
					}
				}
				else
				{
					Gather(avpls);
				}

				if(m_pConfManager->GetConfVars()->DrawLights)
					DrawLights(avpls);

				
				for(uint i = 0; i < avpls.size(); ++i)
				{
					delete avpls[i];
				}

				avpls.clear();
				

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
			std::vector<AVPL*> path = scene->CreatePath(m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);

			GatherRadianceWithShadowMap(path);
	
			if(m_pConfManager->GetConfVars()->DrawLights)
				DrawLights(path);

			m_CurrentPath++;
		}
		else{
			if(!m_Finished)
				std::cout << "Finished." << std::endl;

			m_Finished = true;
		}
	}
	
	Normalize();
	
	Shade();
	
	if(m_pConfManager->GetConfVars()->UseToneMapping)
	{
		m_pPostProcess->SetGamma(m_pConfManager->GetConfVars()->Gamma);
		m_pPostProcess->SetExposure(m_pConfManager->GetConfVars()->Exposure);
		m_pPostProcess->Postprocess(m_pShadeRenderTarget->GetBuffer(0), m_pPostProcessRenderTarget);
		m_pTextureViewer->DrawTexture(m_pPostProcessRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	else
	{
		if(m_pConfManager->GetConfVars()->UseDebugMode && m_pConfManager->GetConfVars()->DrawLights)
			DrawLights(m_DebugAVPLs);
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
	
	if(m_CurrentPath % 100000 == 0 && m_CurrentPath > 0)
	{
		std::stringstream ss;
		ss << "result-" << m_CurrentPath << "paths" << ".pfm";
		m_Export->ExportPFM(m_pShadeRenderTarget->GetBuffer(0), ss.str());

		std::cout << "# paths: " << m_CurrentPath << std::endl;
	}
		
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

void Renderer::GatherRadianceWithShadowMap(std::vector<AVPL*> path)
{
	std::vector<AVPL*>::iterator it;
	for(it = path.begin(); it < path.end(); ++it)
	{
		GatherRadianceFromLightWithShadowMap(*it);
	}
}

void Renderer::Normalize()
{
	ConfigureLighting();	
	{
		CRenderTargetLock lock(m_pNormalizeRenderTarget);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		COGLBindLock lockProgram(m_pNormalizeProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(m_pGatherRenderTarget->GetBuffer(0), COGL_TEXTURE0_SLOT);
		COGLBindLock lock1(m_pGatherRenderTarget->GetBuffer(1), COGL_TEXTURE1_SLOT);
		COGLBindLock lock2(m_pGatherRenderTarget->GetBuffer(2), COGL_TEXTURE2_SLOT);

		m_pFullScreenQuad->Draw();
	}
}

void Renderer::GatherRadianceFromLightWithShadowMap(AVPL* avpl)
{
	FillShadowMap(avpl);
	
	if(avpl->GetIntensity(avpl->GetOrientation()).length() == 0.f)
		return;
	
	if(avpl->GetBounce() != m_pConfManager->GetConfVars()->RenderBounce && m_pConfManager->GetConfVars()->RenderBounce != -1)
		return;
	
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	
	CRenderTargetLock lock(m_pGatherRenderTarget);

	COGLBindLock lockProgram(m_pGatherRadianceWithSMProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	AVPL_STRUCT light_info;
	avpl->Fill(light_info);
	m_pUBLight->UpdateData(&light_info);

	COGLBindLock lock0(m_pShadowMap->GetShadowMapTexture(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_pGBuffer->GetPositionTextureWS(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_pGBuffer->GetNormalTexture(), COGL_TEXTURE2_SLOT);
	
	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

std::vector<AVPL*> Renderer::DetermineUsedAvpls(std::vector<AVPL*> avpls)
{
	// determine avpls used for lighting with m_RenderBounce config
	std::vector<AVPL*> useAVPLs;
	if(m_pConfManager->GetConfVars()->RenderBounce == -1)
	{
		// use all avpls
		useAVPLs = avpls;
	}
	else
	{
		// collect all avpls contributing to the illumination
		for(uint i = 0; i < avpls.size(); ++i)
		{
			AVPL* avpl = avpls[i];
			if(avpl->GetBounce() != m_pConfManager->GetConfVars()->RenderBounce + 1) avpl->SetAntiintensity(glm::vec3(0.f));
			if(avpl->GetBounce() != m_pConfManager->GetConfVars()->RenderBounce) avpl->SetIntensity(glm::vec3(0.f));
			useAVPLs.push_back(avpl);
		}
	}

	return useAVPLs;
}

void Renderer::Gather(std::vector<AVPL*> avpls)
{
	CTimer cpuTimer(CTimer::CPU);
	cpuTimer.Start();
	try
	{
		AVPL_BUFFER* avplBuffer = new AVPL_BUFFER[avpls.size()];
		memset(avplBuffer, 0, sizeof(AVPL_BUFFER) * avpls.size());
		for(uint i = 0; i < avpls.size(); ++i)
		{
			AVPL_BUFFER buffer;
			avpls[i]->Fill(buffer);
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
	cpuTimer.Stop("SetLightBufferContent");
		
	INFO info;
	info.numLights = (int)avpls.size();
	info.filterAVPLAtlas = m_pConfManager->GetConfVars()->FilterAvplAtlasLinear;
	info.drawLightingOfLight = m_pConfManager->GetConfVars()->DrawLightingOfLight;
	m_pUBInfo->UpdateData(&info);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	
	CRenderTargetLock lockRenderTarget(m_pGatherRenderTarget);

	COGLBindLock lockProgram(m_pGatherProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_pGBuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_pOGLLightBuffer, COGL_TEXTURE2_SLOT);

	CTimer gpuTimer(CTimer::OGL);
	gpuTimer.Start();
	
	m_pFullScreenQuad->Draw();

	gpuTimer.Stop("Render");

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::GatherWithAtlas(std::vector<AVPL*> avpls)
{
	std::vector<AVPL*> clampAvpls;

	if((int)avpls.size() > m_MaxNumAVPLs)
		std::cout << "To many avpls. Some are not considered" << std::endl;

	for(int i = 0; i < std::min((int)avpls.size(), m_MaxNumAVPLs); ++i)
	{
		clampAvpls.push_back(avpls[i]);
	}

	try
	{
		AVPL_BUFFER* avplBuffer = new AVPL_BUFFER[clampAvpls.size()];
		memset(avplBuffer, 0, sizeof(AVPL_BUFFER) * clampAvpls.size());
		for(uint i = 0; i < clampAvpls.size(); ++i)
		{
			AVPL_BUFFER buffer;
			clampAvpls[i]->Fill(buffer);
			avplBuffer[i] = buffer;
		}
		m_pOGLLightBuffer->SetContent(avplBuffer, sizeof(AVPL_BUFFER) * clampAvpls.size());
	
		m_pOctahedronAtlas->Clear();

		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU)
		{
			m_pOCLTimer->Start();
			m_pOctahedronAtlas->FillAtlasGPU(avplBuffer, (int)clampAvpls.size(), m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
				float(m_pConfManager->GetConfVars()->ConeFactor), m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
			m_pOCLTimer->Stop("FillAtlasGPU");
		}
		else
		{
			m_pOctahedronAtlas->FillAtlas(clampAvpls, m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
				float(m_pConfManager->GetConfVars()->ConeFactor), m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
		}
		
		delete [] avplBuffer;

		INFO info;
		info.numLights = (int)clampAvpls.size();
		info.drawLightingOfLight = m_pConfManager->GetConfVars()->DrawLightingOfLight;
		info.filterAVPLAtlas = m_pConfManager->GetConfVars()->FilterAvplAtlasLinear;
		m_pUBInfo->UpdateData(&info);

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);	
		glBlendFunc(GL_ONE, GL_ONE);
		
		CRenderTargetLock lockRenderTarget(m_pGatherRenderTarget);

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
			CTimer gpuTimer(CTimer::OGL);
			gpuTimer.Start();
	
			m_pFullScreenQuad->Draw();

			gpuTimer.Stop("Render");
		}
		else
		{
			COGLBindLock lock3(m_pOctahedronAtlas->GetAVPLAtlasCPU(), COGL_TEXTURE3_SLOT);
			CTimer gpuTimer(CTimer::OGL);
			
			gpuTimer.Start();
			m_pFullScreenQuad->Draw();
			gpuTimer.Stop("Render");
		}
				
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);

		glBindSampler(3, m_pGLPointSampler->GetResourceIdentifier());

		clampAvpls.clear();
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at Renderer::GatherWithAtlas()" << std::endl;
		return;
	}
}

void Renderer::GatherWithClustering(std::vector<AVPL*> avpls)
{
	std::vector<AVPL*> clampAvpls;

	if((int)avpls.size() > m_MaxNumAVPLs)
		std::cout << "To many avpls. Some are not considered" << std::endl;

	for(int i = 0; i < std::min((int)avpls.size(), m_MaxNumAVPLs); ++i)
	{
		clampAvpls.push_back(avpls[i]);
	}

	try
	{
		CTimer cpuTimer(CTimer::CPU);
		CTimer gpuTimer(CTimer::OGL);
		cpuTimer.Start();
		gpuTimer.Start();

		// fill light information
		AVPL_BUFFER* avplBuffer = new AVPL_BUFFER[clampAvpls.size()];
		memset(avplBuffer, 0, sizeof(AVPL_BUFFER) * clampAvpls.size());
		for(uint i = 0; i < clampAvpls.size(); ++i)
		{
			AVPL_BUFFER buffer;
			clampAvpls[i]->Fill(buffer);
			avplBuffer[i] = buffer;
		}
		m_pOGLLightBuffer->SetContent(avplBuffer, sizeof(AVPL_BUFFER) * clampAvpls.size());
	
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
		
		cpuTimer.Stop("FillBuffers (CPU timer)");
		gpuTimer.Stop("FillBuffers (GPU timer)");

		// create the avpl and cluster atlas
		m_pOctahedronAtlas->Clear();
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU)
		{
			m_pOCLTimer->Start();
			m_pOctahedronAtlas->FillAtlasGPU(avplBuffer, (int)clampAvpls.size(), m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
				float(m_pConfManager->GetConfVars()->ConeFactor), m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
			m_pOCLTimer->Stop("FillAtlasGPU");

			m_pOCLTimer->Start();
			if(m_pConfManager->GetConfVars()->UseLightTree)
				m_pOctahedronAtlas->FillClusterAtlasGPU(m_pLightTree->GetClustering(), m_pLightTree->GetClusteringSize(), (int)clampAvpls.size());
			else
				m_pOctahedronAtlas->FillClusterAtlasGPU(m_pClusterTree->GetClustering(), m_pClusterTree->GetClusteringSize(), (int)clampAvpls.size());
			m_pOCLTimer->Stop("FillClusterAtlasGPU");
		}
		else
		{
			m_pOctahedronAtlas->FillAtlas(clampAvpls, m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
				float(m_pConfManager->GetConfVars()->ConeFactor), m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
			if(m_pConfManager->GetConfVars()->UseLightTree)
				m_pOctahedronAtlas->FillClusterAtlas(clampAvpls, m_pLightTree->GetClustering(), m_pLightTree->GetClusteringSize());
			else
				m_pOctahedronAtlas->FillClusterAtlas(clampAvpls, m_pClusterTree->GetClustering(), m_pClusterTree->GetClusteringSize());
		}
		
		delete [] avplBuffer;
		delete [] clusterBuffer;

		INFO info;
		info.numLights = (int)clampAvpls.size();

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
		
		CRenderTargetLock lockRenderTarget(m_pGatherRenderTarget);

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
	
			gpuTimer.Start();
			m_pFullScreenQuad->Draw();
			gpuTimer.Stop("Render");
		}
		else
		{
			COGLBindLock lock4(m_pOctahedronAtlas->GetAVPLAtlasCPU(), COGL_TEXTURE4_SLOT);
			COGLBindLock lock5(m_pOctahedronAtlas->GetAVPLClusterAtlasCPU(), COGL_TEXTURE5_SLOT);
			
			gpuTimer.Start();
			m_pFullScreenQuad->Draw();
			gpuTimer.Stop("Render");
		}
				
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);

		glBindSampler(4, m_pGLPointSampler->GetResourceIdentifier());
		glBindSampler(5, m_pGLPointSampler->GetResourceIdentifier());

		clampAvpls.clear();
	}
	catch(std::bad_alloc)
	{
		std::cout << "bad_alloc exception at Renderer::GatherWithAtlas()" << std::endl;
		return;
	}
}

void Renderer::FillShadowMap(AVPL* avpl)
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

	scene->DrawScene(avpl->GetViewMatrix(), avpl->GetProjectionMatrix(), m_pUBTransform);

	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::Shade()
{
	CRenderTargetLock lock(m_pShadeRenderTarget);
	
	{
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		COGLBindLock lockProgram(m_pShadeProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(m_pNormalizeRenderTarget->GetBuffer(0), COGL_TEXTURE0_SLOT);
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

void Renderer::DrawLights(std::vector<AVPL*> avpls)
{	
	std::vector<POINT_CLOUD_POINT> pcp;
	std::vector<AVPL*>::iterator it;

	for ( it=avpls.begin() ; it < avpls.end(); it++ )
	{
		AVPL* avpl = *it;
		int rb = m_pConfManager->GetConfVars()->RenderBounce;
		if(rb == -1 || (avpl->GetBounce() == rb || avpl->GetBounce() == rb + 1))
		{
			POINT_CLOUD_POINT p;
			glm::vec3 pos = avpl->GetPosition() + 2.f * avpl->GetOrientation();
			p.position = glm::vec4(pos, 1.0f);
			p.color = glm::vec4(avpl->GetColor(), 1.f);
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
			CRenderTargetLock lock(m_pShadeRenderTarget);
			
			COGLBindLock lockProgram(m_pPointCloudProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
			
			TRANSFORM transform;
			transform.M = IdentityMatrix();
			transform.V = camera->GetViewMatrix();
			transform.itM = IdentityMatrix();
			transform.MVP = camera->GetProjectionMatrix() * camera->GetViewMatrix();
			m_pUBTransform->UpdateData(&transform);
		
			glDepthMask(GL_FALSE);
			//glDisable(GL_DEPTH_TEST);
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

void Renderer::DebugRender()
{
	// draw gbuffer info for debuging
	if(m_pConfManager->GetConfVars()->DrawDebugTextures) 
	{
		m_pTextureViewer->DrawTexture(m_pLightDebugRenderTarget->GetBuffer(0), 10, 360, 620, 340);
		m_pTextureViewer->DrawTexture(m_pNormalizeRenderTarget->GetBuffer(0),  640, 360, 620, 340);
		m_pTextureViewer->DrawTexture(m_pNormalizeRenderTarget->GetBuffer(1),  10, 10, 620, 340);
		m_pTextureViewer->DrawTexture(m_pNormalizeRenderTarget->GetBuffer(2), 640, 10, 620, 340);
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

	m_CurrentPath = 0;
	m_Finished = false;
}

void Renderer::ClearLighting()
{
	scene->ClearLighting();
	m_CurrentPath = 0;
	m_Finished = false;
		
	ClearAccumulationBuffer();
	/*
	if(m_pConfManager->GetConfVars()->ClusterMethod == 0)
	{
		m_pLightTree->Release();
		m_pLightTree->BuildTreeTweakNN(m_DebugAVPLs, 0.f);
		m_pLightTree->Color(m_DebugAVPLs, m_pConfManager->GetConfVars()->ClusterDepth);
	}
	else if(m_pConfManager->GetConfVars()->ClusterMethod == 1)
	{
		m_pLightTree->Release();
		m_pLightTree->BuildTreeTweakCP(m_DebugAVPLs, 0.f);
		m_pLightTree->Color(m_DebugAVPLs, m_pConfManager->GetConfVars()->ClusterDepth);
	}
	else
	{
		m_pLightTree->Release();
		m_pLightTree->BuildTreeNaive(m_DebugAVPLs, 0.f);
		m_pLightTree->Color(m_DebugAVPLs, m_pConfManager->GetConfVars()->ClusterDepth);
	}	
	*/
	
	//m_pClusterTree->Color(m_ClusterTestAVPLs, m_pConfManager->GetConfVars()->ClusterDepth);
}

void Renderer::Stats() 
{
}

void Renderer::ConfigureLighting()
{
	CONFIG conf;
	conf.GeoTermLimit = m_pConfManager->GetConfVars()->GeoTermLimit;
	conf.UseAntiradiance = m_pConfManager->GetConfVars()->UseAntiradiance ? 1 : 0;
	conf.nPaths = std::min(m_CurrentPath, m_pConfManager->GetConfVars()->NumPaths * m_pConfManager->GetConfVars()->NumPathsPerFrame);
	conf.N = m_pConfManager->GetConfVars()->ConeFactor;
	m_pUBConfig->UpdateData(&conf);
}

void Renderer::PrintCameraConfig()
{
	std::cout << "Camera: " << std::endl;
	std::cout << "position: (" << scene->GetCamera()->GetPosition().x << ", " 
		<< scene->GetCamera()->GetPosition().y << ", " 
		<< scene->GetCamera()->GetPosition().z << ")" << std::endl;
}

glm::vec4 Renderer::ColorForLight(AVPL* avpl)
{
	glm::vec4 color;
	switch(avpl->GetBounce()){
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

	for(uint i = 0; i < m_DebugAVPLs.size(); ++i)
	{
		delete m_DebugAVPLs[i];
	}
	m_DebugAVPLs.clear();
	
	InitDebugLights();
}

void Renderer::InitDebugLights()
{
	m_pCPUTimer->Start();
	m_DebugAVPLs = scene->CreatePaths(m_pConfManager->GetConfVars()->NumPaths, 
		m_pConfManager->GetConfVars()->ConeFactor, m_pConfManager->GetConfVars()->NumAdditionalAVPLs);
	std::cout << "Number of AVPLs: " << m_DebugAVPLs.size() << std::endl;
	m_pCPUTimer->Stop("CreatePaths");
}

void Renderer::SetConfigManager(CConfigManager* pConfManager)
{
	m_pConfManager = pConfManager;
}
