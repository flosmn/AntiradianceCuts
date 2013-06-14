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

#include "Material.h"
#include "AVPL.h"
#include "Scene.h"
#include "CCamera.h"
#include "CShadowMap.h"
#include "CPostprocess.h"
#include "CRenderTarget.h"
#include "CClusterTree.h"
#include "CPriorityQueue.h"
#include "CAVPLImportanceSampling.h"
#include "CBidirInstantRadiosity.h"
#include "CImagePlane.h"
#include "CPathTracingIntegrator.h"
#include "CMaterialBuffer.h"
#include "CReferenceImage.h"
#include "CExperimentData.h"
#include "CCubeShadowMap.h"
#include "CImage.h"

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
#include "OGLResources\COGLCubeMap.h"

#include "OCLResources\COCLContext.h"
#include "OCLResources\COCLProgram.h"
#include "OCLResources\COCLKernel.h"
#include "OCLResources\COCLBuffer.h"
#include "OCLResources\COCLTexture2D.h"

#include <memory>
#include <string>
#include <sstream>
#include <time.h>
#include <iterator>

#include <omp.h>

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

	m_pClusterTree = new CClusterTree();
	
	m_pTextureViewer = new CTextureViewer();

	m_pCubeMap = new COGLCubeMap("Renderer.m_pCubeMap");

	m_pUBTransform = new COGLUniformBuffer("Renderer.m_pUBTransform");
	m_pUBMaterial = new COGLUniformBuffer("Renderer.m_pUBMaterial");
	m_pUBLight = new COGLUniformBuffer("Renderer.m_pUBLight");
	m_pUBConfig = new COGLUniformBuffer("Renderer.m_pUBConfig");
	m_pUBCamera = new COGLUniformBuffer("Renderer.m_pUBCamera");
	m_pUBInfo = new COGLUniformBuffer("Renderer.m_pUBInfo");
	m_pUBAreaLight = new COGLUniformBuffer("Renderer.m_pUBAreaLight");
	m_pUBModel = new COGLUniformBuffer("Renderer.m_pUBModel");
	m_pUBAtlasInfo = new COGLUniformBuffer("Renderer.m_pUBAtlasInfo");
	m_pUBNormalize = new COGLUniformBuffer("Renderer.m_pUBNormalize");
	
	m_pGLLinearSampler = new COGLSampler("Renderer.m_pGLLinearSampler");
	m_pGLPointSampler = new COGLSampler("Renderer.m_pGLPointSampler");
	m_pGLShadowMapSampler = new COGLSampler("Renderer.m_pGLShadowMapSampler");

	m_pResultRenderTarget = new CRenderTarget();
	m_pErrorRenderTarget = new CRenderTarget();
	m_pGatherShadowmapRenderTarget = new CRenderTarget();
	m_pGatherAntiradianceRenderTarget = new CRenderTarget();
	m_pNormalizeShadowmapRenderTarget = new CRenderTarget();
	m_pNormalizeAntiradianceRenderTarget = new CRenderTarget();
	m_pShadeShadowmapRenderTarget = new CRenderTarget();
	m_pShadeAntiradianceRenderTarget = new CRenderTarget();
	
	m_pGatherProgram = new CProgram("Renderer.m_pGatherProgram", "Shaders/Gather.vert", "Shaders/Gather.frag");
	m_pGatherWithAtlas = new CProgram("Renderer.m_pGatherProgram", "Shaders/Gather.vert", "Shaders/GatherWithAtlas.frag");
	m_pGatherWithClustering = new CProgram("Renderer.m_pGatherProgram", "Shaders/Gather.vert", "Shaders/GatherWithClustering.frag");
	m_pNormalizeProgram = new CProgram("Renderer.m_pNormalizeProgram", "Shaders/Gather.vert", "Shaders/Normalize.frag");
	m_pShadeProgram = new CProgram("Renderer.m_pShadeProgram", "Shaders/Gather.vert", "Shaders/Shade.frag");
	m_pErrorProgram = new CProgram("Renderer.m_pErrorProgram", "Shaders/Gather.vert", "Shaders/Error.frag");
	m_pAddProgram = new CProgram("Renderer.m_pAddProgram", "Shaders/Gather.vert", "Shaders/Add.frag");
	m_pDirectEnvmapLighting = new CProgram("Renderer.m_pDirectEnvmapLighting", "Shaders/Gather.vert", "Shaders/DirectEnvMapLighting.frag");
	m_pCreateGBufferProgram = new CProgram("Renderer.m_pCreateGBufferProgram", "Shaders/CreateGBuffer.vert", "Shaders/CreateGBuffer.frag");
	m_pCreateSMProgram = new CProgram("Renderer.m_pCreateSMProgram", "Shaders/CreateSM.vert", "Shaders/CreateSM.frag");
	m_pGatherRadianceWithSMProgram = new CProgram("Renderer.m_pGatherRadianceWithSMProgram", "Shaders/Gather.vert", "Shaders/GatherRadianceWithSM.frag");
	m_pPointCloudProgram = new CProgram("Renderer.m_pPointCloudProgram", "Shaders/PointCloud.vert", "Shaders/PointCloud.frag");
	m_pAreaLightProgram = new CProgram("Renderer.m_pAreaLightProgram", "Shaders/DrawAreaLight.vert", "Shaders/DrawAreaLight.frag");
	m_pDrawOctahedronProgram = new CProgram("Renderer.m_pDrawOctahedronProgram", "Shaders/DrawOctahedron.vert", "Shaders/DrawOctahedron.frag");

	m_pFullScreenQuad = new CFullScreenQuad();
	m_pOctahedron = new CModel();

	m_pPointCloud = new CPointCloud();

	m_pOGLLightBuffer = new COGLTextureBuffer("Renderer.m_pOGLLightBuffer");
	m_pOGLClusterBuffer = new COGLTextureBuffer("Renderer.m_pOGLClusterBuffer");
	
	m_pAVPLPositions = new COGLTextureBuffer("Renderer.m_pAVPLPositions");

	m_pImagePlane = new CImagePlane();
	m_pPathTracingIntegrator = new CPathTracingIntegrator();
		
	m_pCPUTimer = new CTimer(CTimer::CPU);
	m_pOGLTimer = new CTimer(CTimer::OGL);
	m_pOCLTimer = new CTimer(CTimer::OCL, m_pCLContext);
	m_pGlobalTimer = new CTimer(CTimer::CPU);
	m_pResultTimer = new CTimer(CTimer::CPU);
	m_pCPUFrameProfiler = new CTimer(CTimer::CPU);
	m_pGPUFrameProfiler = new CTimer(CTimer::OGL);

	m_Finished = false;
	m_FinishedDirectLighting = false;
	m_FinishedIndirectLighting = false;
	m_FinishedDebug = false;

	m_ProfileFrame = false;

	m_pExperimentData = new CExperimentData();

	m_NumAVPLsForNextDataExport = 1000;
	m_NumAVPLsForNextImageExport = 10000;
	m_NumAVPLs = 0;
	m_NumPathsDebug = 0;

	m_ClearLighting = false;
	m_ClearAccumulationBuffer = false;
}

Renderer::~Renderer() {
	SAFE_DELETE(scene);
	SAFE_DELETE(m_Export);
	SAFE_DELETE(m_pShadowMap);
	SAFE_DELETE(m_pGBuffer);
	SAFE_DELETE(m_pClusterTree);

	SAFE_DELETE(m_pResultRenderTarget);
	SAFE_DELETE(m_pErrorRenderTarget);
	SAFE_DELETE(m_pGatherShadowmapRenderTarget);
	SAFE_DELETE(m_pGatherAntiradianceRenderTarget);
	SAFE_DELETE(m_pNormalizeShadowmapRenderTarget);
	SAFE_DELETE(m_pNormalizeAntiradianceRenderTarget);
	SAFE_DELETE(m_pShadeShadowmapRenderTarget);
	SAFE_DELETE(m_pShadeAntiradianceRenderTarget);
		
	SAFE_DELETE(m_pGatherProgram);
	SAFE_DELETE(m_pGatherWithAtlas);
	SAFE_DELETE(m_pGatherWithClustering);
	SAFE_DELETE(m_pNormalizeProgram);
	SAFE_DELETE(m_pShadeProgram);
	SAFE_DELETE(m_pErrorProgram);
	SAFE_DELETE(m_pAddProgram);
	SAFE_DELETE(m_pDirectEnvmapLighting);
	
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
	SAFE_DELETE(m_pUBNormalize);
	
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

	SAFE_DELETE(m_pCubeMap);
	
	SAFE_DELETE(m_pAVPLPositions);

	SAFE_DELETE(m_pOGLLightBuffer);
	SAFE_DELETE(m_pOGLClusterBuffer);
	
	SAFE_DELETE(m_pOctahedronAtlas);
	SAFE_DELETE(m_pOctahedronMap);

	SAFE_DELETE(m_pCPUTimer);
	SAFE_DELETE(m_pOCLTimer);
	SAFE_DELETE(m_pOGLTimer);
	SAFE_DELETE(m_pGlobalTimer);
	SAFE_DELETE(m_pResultTimer);

	SAFE_DELETE(m_pCLContext);

	SAFE_DELETE(m_pAVPLImportanceSampling);
	SAFE_DELETE(m_pBidirInstantRadiosity);

	SAFE_DELETE(m_pImagePlane);
	SAFE_DELETE(m_pPathTracingIntegrator);

	SAFE_DELETE(m_pExperimentData);
}

bool Renderer::Init() 
{	
	V_RET_FOF(m_pCLContext->Init(m_pOGLContext));
		
	V_RET_FOF(m_pDepthBuffer->Init(camera->GetWidth(), camera->GetHeight(), GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, false));
	V_RET_FOF(m_pTestTexture->Init(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F, GL_RGBA, GL_FLOAT, 1, false));
	
	V_RET_FOF(m_pUBTransform->Init(sizeof(TRANSFORM), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBMaterial->Init(sizeof(MATERIAL), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBLight->Init(sizeof(AVPL_STRUCT), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBConfig->Init(sizeof(CONFIG), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBCamera->Init(sizeof(CAMERA), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBInfo->Init(sizeof(INFO), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBAreaLight->Init(sizeof(AREA_LIGHT), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBModel->Init(sizeof(MODEL), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBAtlasInfo->Init(sizeof(ATLAS_INFO), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBNormalize->Init(sizeof(NORMALIZE), 0, GL_DYNAMIC_DRAW));
	
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
	V_RET_FOF(m_pErrorProgram->Init());
	V_RET_FOF(m_pAreaLightProgram->Init());
	V_RET_FOF(m_pAddProgram->Init());
	V_RET_FOF(m_pDirectEnvmapLighting->Init());
	
	V_RET_FOF(m_pResultRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));
	V_RET_FOF(m_pErrorRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, 0));
	V_RET_FOF(m_pGatherShadowmapRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 4, 0));
	V_RET_FOF(m_pGatherAntiradianceRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 4, 0));
	V_RET_FOF(m_pNormalizeShadowmapRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 3, 0));
	V_RET_FOF(m_pNormalizeAntiradianceRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 3, 0));
	V_RET_FOF(m_pShadeShadowmapRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));
	V_RET_FOF(m_pShadeAntiradianceRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));
	
	V_RET_FOF(m_pPostProcess->Init(m_pConfManager));
	V_RET_FOF(m_pPostProcessRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, 0));
	V_RET_FOF(m_pLightDebugRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 1, m_pDepthBuffer));
		
	m_pAreaLightProgram->BindUniformBuffer(m_pUBTransform, "transform");
	m_pAreaLightProgram->BindUniformBuffer(m_pUBAreaLight, "arealight");

	m_pCreateGBufferProgram->BindUniformBuffer(m_pUBTransform, "transform");
		
	V_RET_FOF(m_pGLLinearSampler->Init(GL_LINEAR, GL_LINEAR, GL_CLAMP, GL_CLAMP));
	V_RET_FOF(m_pGLPointSampler->Init(GL_NEAREST, GL_NEAREST, GL_REPEAT, GL_REPEAT));
	V_RET_FOF(m_pGLShadowMapSampler->Init(GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER));
	
	m_pDirectEnvmapLighting->BindUniformBuffer(m_pUBCamera, "camera");
	m_pDirectEnvmapLighting->BindSampler(0, m_pGLPointSampler);
	m_pDirectEnvmapLighting->BindSampler(1, m_pGLPointSampler);

	m_pGatherProgram->BindSampler(0, m_pGLPointSampler);
	m_pGatherProgram->BindSampler(1, m_pGLPointSampler);
	m_pGatherProgram->BindSampler(2, m_pGLPointSampler);
	m_pGatherProgram->BindSampler(3, m_pGLPointSampler);

	m_pGatherProgram->BindUniformBuffer(m_pUBInfo, "info_block");
	m_pGatherProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pGatherRadianceWithSMProgram->BindUniformBuffer(m_pUBCamera, "camera");
	m_pGatherRadianceWithSMProgram->BindUniformBuffer(m_pUBConfig, "config");	
	m_pGatherRadianceWithSMProgram->BindUniformBuffer(m_pUBLight, "light");

	m_pGatherRadianceWithSMProgram->BindSampler(0, m_pGLShadowMapSampler);
	m_pGatherRadianceWithSMProgram->BindSampler(1, m_pGLPointSampler);
	m_pGatherRadianceWithSMProgram->BindSampler(2, m_pGLPointSampler);
	m_pGatherRadianceWithSMProgram->BindSampler(3, m_pGLPointSampler);
	
	m_pGatherWithAtlas->BindSampler(0, m_pGLPointSampler);
	m_pGatherWithAtlas->BindSampler(1, m_pGLPointSampler);
	m_pGatherWithAtlas->BindSampler(2, m_pGLPointSampler);
	m_pGatherWithAtlas->BindSampler(3, m_pGLPointSampler);
	m_pGatherWithAtlas->BindSampler(4, m_pGLLinearSampler);

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
	m_pGatherWithClustering->BindSampler(6, m_pGLPointSampler);
	
	m_pGatherWithClustering->BindUniformBuffer(m_pUBInfo, "info_block");
	m_pGatherWithClustering->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherWithClustering->BindUniformBuffer(m_pUBCamera, "camera");
	m_pGatherWithClustering->BindUniformBuffer(m_pUBAtlasInfo, "atlas_info");
		
	m_pNormalizeProgram->BindSampler(0, m_pGLPointSampler);
	m_pNormalizeProgram->BindSampler(1, m_pGLPointSampler);
	m_pNormalizeProgram->BindSampler(2, m_pGLPointSampler);

	m_pNormalizeProgram->BindUniformBuffer(m_pUBNormalize, "norm");
	m_pNormalizeProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pShadeProgram->BindSampler(0, m_pGLPointSampler);
	m_pShadeProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pErrorProgram->BindSampler(0, m_pGLPointSampler);
	m_pErrorProgram->BindSampler(1, m_pGLPointSampler);
	m_pErrorProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pAddProgram->BindSampler(0, m_pGLPointSampler);
	m_pAddProgram->BindSampler(1, m_pGLPointSampler);
	m_pAddProgram->BindUniformBuffer(m_pUBCamera, "camera");
		
	V_RET_FOF(m_pDrawOctahedronProgram->Init());
	m_pDrawOctahedronProgram->BindSampler(0, m_pGLPointSampler);
	m_pDrawOctahedronProgram->BindUniformBuffer(m_pUBTransform, "transform");
	m_pDrawOctahedronProgram->BindUniformBuffer(m_pUBModel, "model");
	
	V_RET_FOF(m_pFullScreenQuad->Init());
	
	V_RET_FOF(m_pShadowMap->Init(1024));
	
	V_RET_FOF(m_pGBuffer->Init(camera->GetWidth(), camera->GetHeight(), m_pDepthBuffer));
	
	m_pCubeMap->Init(512, 512, GL_RGBA32F, 10);
	m_pCubeMap->LoadCubeMapFromPath("Resources\\CubeMaps\\Castle\\box\\");

	scene = new Scene(camera, m_pConfManager, m_pCLContext);
	scene->Init();
	scene->LoadCornellBox();
	scene->GetMaterialBuffer()->InitOGLMaterialBuffer();
	scene->GetMaterialBuffer()->InitOCLMaterialBuffer();
		
	UpdateUniformBuffers();

	time(&m_StartTime);

	int dim_atlas = 3072;
	int dim_tile = 16;
		
	ATLAS_INFO atlas_info;
	atlas_info.dim_atlas = dim_atlas;
	atlas_info.dim_tile = dim_tile;
	m_pUBAtlasInfo->UpdateData(&atlas_info);
	
	m_pOctahedronMap->Init(16);
	m_pOctahedronMap->FillWithDebugData();

	m_MaxNumAVPLs = int(std::pow(float(dim_atlas) / float(dim_tile), 2.f));
	std::cout << "max num avpls: " << m_MaxNumAVPLs << std::endl;

	V_RET_FOF(m_pAVPLPositions->Init(sizeof(AVPL_POSITION) * m_MaxNumAVPLs, GL_STATIC_DRAW, GL_R32F));

	V_RET_FOF(m_pOGLLightBuffer->Init(sizeof(AVPL_BUFFER) * m_MaxNumAVPLs, GL_STATIC_DRAW, GL_RGBA32F));
	V_RET_FOF(m_pOGLClusterBuffer->Init(sizeof(CLUSTER_BUFFER) * (2 * m_MaxNumAVPLs - 1), GL_STATIC_DRAW, GL_RGBA32F));

	m_pOctahedronAtlas->Init(dim_atlas, dim_tile, m_MaxNumAVPLs, scene->GetMaterialBuffer());
	
	V_RET_FOF(m_pOctahedron->Init("octahedron", "obj", scene->GetMaterialBuffer()));
		
	glm::mat4 scale = glm::scale(IdentityMatrix(), glm::vec3(70.f, 70.f, 70.f));
	glm::mat4 trans = glm::translate(IdentityMatrix(), glm::vec3(278.f, 273.f, 270.f));
	m_pOctahedron->SetWorldTransform(trans * scale);

	m_pAVPLImportanceSampling = new CAVPLImportanceSampling(scene, m_pConfManager);
	scene->SetAVPLImportanceSampling(m_pAVPLImportanceSampling);

	m_pBidirInstantRadiosity = new CBidirInstantRadiosity(scene, m_pConfManager);
	
	m_pImagePlane->Init(scene->GetCamera());
	m_pPathTracingIntegrator->Init(scene, m_pImagePlane);
	
	InitDebugLights();
		
	ClearAccumulationBuffer();
	
	m_pGlobalTimer->Start();

	CImage image(128, 512);

	glm::vec4* pData = new glm::vec4[128 * 512];
	memset(pData, 0, 128 * 512 * sizeof(glm::vec4));

	for(int h = 0; h < 512; ++h)
	{
		glm::vec3 color = hue_colormap(float(h)/512.f, 0.f, 1.f);
		for(int w = 0; w < 128; ++w)
		{
			pData[h * 128 + w] = glm::vec4(color, 1.f);
		}
	}

	image.SetData(pData);
	image.SaveAsPNG("huemap.png", false);
		
	return true;
}

void Renderer::TestClusteringSpeed()
{
	CClusterTree tree;

	for(int i = 0; i < 10; ++i)
	{
		int numAVPLs = 10000 * (i+1);
		std::stringstream ss;
		ss << "Build Tree (" << numAVPLs << ")"; 

		std::vector<AVPL> randAVPLs;
		CreateRandomAVPLs(randAVPLs, numAVPLs);

		CTimer timer(CTimer::CPU);
		timer.Start();
		
		tree.BuildTree(randAVPLs);
		
		timer.Stop(ss.str());
		tree.Release();
		randAVPLs.clear();

		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;
}

void Renderer::ClusteringTestRender()
{	
	SetUpRender();
	UpdateUniformBuffers();
	
	{
		CRenderTargetLock lock(m_pResultRenderTarget);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	}
	
	DrawLights(m_ClusterTestAVPLs, m_pResultRenderTarget);

	m_pTextureViewer->DrawTexture(m_pResultRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
}

void Renderer::Release()
{
	CheckGLError("CDSRenderer", "CDSRenderer::Release()");

	scene->GetMaterialBuffer()->ReleaseOGLMaterialBuffer();
	scene->GetMaterialBuffer()->ReleaseOCLMaterialBuffer();

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

	m_pNormalizeProgram->Release();
	m_pGatherProgram->Release();
	m_pGatherWithAtlas->Release();
	m_pGatherWithClustering->Release();
	m_pShadeProgram->Release();
	m_pErrorProgram->Release();
	m_pCreateGBufferProgram->Release();
	m_pCreateSMProgram->Release();
	m_pGatherRadianceWithSMProgram->Release();
	m_pPointCloudProgram->Release();
	m_pAreaLightProgram->Release();
	m_pDrawOctahedronProgram->Release();
	m_pAddProgram->Release();
	m_pDirectEnvmapLighting->Release();
	
	m_pResultRenderTarget->Release();
	m_pErrorRenderTarget->Release();
	m_pGatherShadowmapRenderTarget->Release();
	m_pGatherAntiradianceRenderTarget->Release();
	m_pNormalizeShadowmapRenderTarget->Release();
	m_pNormalizeAntiradianceRenderTarget->Release();
	m_pShadeShadowmapRenderTarget->Release();
	m_pShadeAntiradianceRenderTarget->Release();
		
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
	m_pUBNormalize->Release();

	m_pOctahedron->Release();
	m_pOctahedronAtlas->Release();
	m_pOctahedronMap->Release();
		
	m_pCLContext->Release();
	m_pCubeMap->Release();
	m_pClusterTree->Release();
	m_ClusterTestAVPLs.clear();
	m_pImagePlane->Release();
}

void Renderer::Render() 
{	
	if(m_ClearLighting)
		ClearLighting();
	if(m_ClearAccumulationBuffer)
		ClearAccumulationBuffer();

	CTimer frameTimer(CTimer::OGL);
	CTimer timer(CTimer::OGL);

	if(m_ProfileFrame)
	{	
		std::cout << std::endl;
		std::cout << "Profile frame --------------- " << std::endl;
		std::cout << std::endl;
		frameTimer.Start();
		timer.Start();
	}

	SetUpRender();

	if(m_ProfileFrame) timer.Stop("set up render");
	if(m_ProfileFrame) timer.Start();

	UpdateUniformBuffers();
	
	if(m_ProfileFrame) timer.Stop("update ubs");
	
	m_pAVPLImportanceSampling->CreateSceneSamples();

	if(m_pConfManager->GetConfVars()->UseAVPLImportanceSampling && !m_pConfManager->GetConfVars()->UseDebugMode)
	{
		if(m_ProfileFrame) timer.Start();
		m_pAVPLImportanceSampling->UpdateCurrentIrradiance(m_pNormalizeAntiradianceRenderTarget->GetBuffer(1));
		m_pAVPLImportanceSampling->UpdateCurrentAntiirradiance(m_pNormalizeAntiradianceRenderTarget->GetBuffer(2));
		m_pAVPLImportanceSampling->SetNumberOfSceneSamples(m_pConfManager->GetConfVars()->NumSceneSamples);
		if(m_ProfileFrame) timer.Stop("importance sample stuff");
	}

	if(m_CurrentPathAntiradiance == 0 && m_CurrentPathShadowmap == 0)
	{
		m_pExperimentData->Init("test", "nois.data");
		m_pExperimentData->MaxTime(450);

		m_pGlobalTimer->Start();
		m_pResultTimer->Start();
		m_pOGLTimer->Start();
		CreateGBuffer();
	}
	
	std::vector<AVPL> avpls_shadowmap;
	std::vector<AVPL> avpls_antiradiance;

	if(m_ProfileFrame) timer.Start();

	GetAVPLs(avpls_shadowmap, avpls_antiradiance);

	if(m_ProfileFrame) timer.Stop("get avpls");
	if(m_ProfileFrame) timer.Start();

	if(m_pConfManager->GetConfVars()->GatherWithAVPLClustering)
	{
		CreateClustering(avpls_antiradiance);
		if(m_ProfileFrame) timer.Stop("clustering");
		if(m_ProfileFrame) timer.Start();
		FillAVPLAtlas(avpls_antiradiance);
		if(m_ProfileFrame) timer.Stop("fill avpls atlas");
		if(m_ProfileFrame) timer.Start();
		FillClusterAtlas(avpls_antiradiance);
		if(m_ProfileFrame) timer.Stop("fill cluster atlas");
		if(m_ProfileFrame) timer.Start();
	}

	if(m_pConfManager->GetConfVars()->GatherWithAVPLAtlas)
	{
		FillAVPLAtlas(avpls_antiradiance);
		if(m_ProfileFrame) timer.Stop("fill avpls atlas");
		if(m_ProfileFrame) timer.Start();
	}

	Gather(avpls_shadowmap, avpls_antiradiance);
	
	if(m_ProfileFrame) timer.Stop("gather");
	if(m_ProfileFrame) timer.Start();

	Normalize();

	if(m_ProfileFrame) timer.Stop("normalize");
	if(m_ProfileFrame) timer.Start();

	Shade();

	if(m_ProfileFrame) timer.Stop("shade");
		
	if(m_ProfileFrame) timer.Start();

	Finalize();

	if(m_pConfManager->GetConfVars()->UseDebugMode) {
		DrawLights(m_DebugAVPLs, m_pResultRenderTarget);
	}
	else
	{
		DrawLights(avpls_shadowmap, m_pResultRenderTarget);
		DrawLights(avpls_antiradiance, m_pResultRenderTarget);
	}
	
	m_pPostProcess->Postprocess(m_pResultRenderTarget->GetBuffer(0), m_pPostProcessRenderTarget);
	m_pTextureViewer->DrawTexture(m_pPostProcessRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());	
	
	if(m_ProfileFrame) timer.Stop("finalize");
	if(m_ProfileFrame) timer.Start();

	DrawDebug();

	if(m_ProfileFrame) timer.Stop("draw debug");
		
	m_NumAVPLs += (int)avpls_antiradiance.size();
	m_NumAVPLs += (int)avpls_shadowmap.size();

	if(m_ProfileFrame) timer.Start();

	avpls_antiradiance.clear();
	avpls_shadowmap.clear();

	if(m_ProfileFrame) timer.Stop("clear avpls");
	
	CheckExport();

	m_Frame++;

	if(m_ProfileFrame) frameTimer.Stop("frame time");
		
	m_ProfileFrame = false;
	m_FinishedDebug = true;
}

void Renderer::GetAVPLs(std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance)
{
	if(m_pConfManager->GetConfVars()->UseDebugMode)
	{
		if(!m_FinishedDebug)
		{
			SeparateAVPLs(m_DebugAVPLs, avpls_shadowmap, avpls_antiradiance, m_NumPathsDebug);
		}
	}
	else
	{
		if(m_pConfManager->GetConfVars()->SeparateDirectIndirectLighting && m_pConfManager->GetConfVars()->LightingMode != 2)
		{
			if(m_CurrentPathShadowmap < m_pConfManager->GetConfVars()->NumVPLsDirectLight)
			{
				scene->CreatePrimaryVpls(avpls_shadowmap, m_pConfManager->GetConfVars()->NumVPLsDirectLightPerFrame);
				m_CurrentPathShadowmap += m_pConfManager->GetConfVars()->NumVPLsDirectLightPerFrame;
			}
			else if(!m_FinishedDirectLighting)
			{
				std::cout << "Finished direct lighting" << std::endl;
				m_FinishedDirectLighting = true;
			}
		}
		
		std::vector<AVPL> avpls;
		int numAVPLs = m_pConfManager->GetConfVars()->NumAVPLsPerFrame;
		int numAVPLsPerBatch = std::min(numAVPLs, 1000);
		int numPaths = 0;

		while(avpls.size() < numAVPLs)
		{
			#pragma omp parallel
			{
				int num_threads = omp_get_num_threads();
				std::vector<AVPL> avpls_thread;
				int numPaths_thread = 0;
				while(avpls_thread.size() < (numAVPLsPerBatch + num_threads - 1) / num_threads)
				{
					scene->CreatePath(avpls_thread);
					numPaths_thread++;
				}

				if(m_pConfManager->GetConfVars()->UseAVPLImportanceSampling)
				{
					std::vector<AVPL> result;
					m_pAVPLImportanceSampling->ImportanceSampling(avpls_thread, result);
					avpls_thread.clear();
					
					for(int i = 0; i < result.size(); ++i)
					avpls_thread.push_back(result[i]);
				}
				
				#pragma omp critical
				{
					avpls.insert(avpls.end(), avpls_thread.begin(), avpls_thread.end());
					numPaths += numPaths_thread;
				}
			}
		}
		SeparateAVPLs(avpls, avpls_shadowmap, avpls_antiradiance, numPaths);
	}
}

void Renderer::SeparateAVPLs(const std::vector<AVPL> avpls, 
	std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance, int numPaths)
{
	if(!m_pConfManager->GetConfVars()->UseAntiradiance)
	{
		for(int i = 0; i < avpls.size(); ++i)
		{
			AVPL avpl = avpls[i];
			if(UseAVPL(avpl))
				avpls_shadowmap.push_back(avpl);
		}
		m_CurrentPathShadowmap += numPaths;
		return;
	}
	
	if(!m_pConfManager->GetConfVars()->SeparateDirectIndirectLighting)
	{
		for(int i = 0; i < avpls.size(); ++i)
		{
			AVPL avpl = avpls[i];
			if(UseAVPL(avpl))
				avpls_antiradiance.push_back(avpl);
		}
		m_CurrentPathAntiradiance += numPaths;
		return;
	}

	for(int i = 0; i < avpls.size(); ++i)
	{
		AVPL avpl = avpls[i];
		
		if(avpl.GetBounce() != 0)
		{
			if(avpl.GetBounce() == 1)
			{
				avpl.ScaleAntiradiance(0.f);
			}

			if(UseAVPL(avpl))
				avpls_antiradiance.push_back(avpl);
		}
	}
	m_CurrentPathAntiradiance += numPaths;
}

void Renderer::Gather(std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance)
{
	if(m_pConfManager->GetConfVars()->UseDebugMode && m_FinishedDebug)
		return;
	
	if(m_pConfManager->GetConfVars()->GatherWithAVPLClustering)
		GatherWithClustering(avpls_antiradiance, m_pGatherAntiradianceRenderTarget);
	else if(m_pConfManager->GetConfVars()->GatherWithAVPLAtlas)
		GatherWithAtlas(avpls_antiradiance, m_pGatherAntiradianceRenderTarget);
	else
		Gather(avpls_antiradiance, m_pGatherAntiradianceRenderTarget);

	GatherRadianceWithShadowMap(avpls_shadowmap, m_pGatherShadowmapRenderTarget);
}

void Renderer::Normalize()
{
	Normalize(m_pNormalizeShadowmapRenderTarget, m_pGatherShadowmapRenderTarget, m_CurrentPathShadowmap);
	Normalize(m_pNormalizeAntiradianceRenderTarget, m_pGatherAntiradianceRenderTarget, m_CurrentPathAntiradiance);
}

void Renderer::Shade()
{
	Shade(m_pShadeShadowmapRenderTarget, m_pNormalizeShadowmapRenderTarget);
	Shade(m_pShadeAntiradianceRenderTarget, m_pNormalizeAntiradianceRenderTarget);
}

void Renderer::Finalize()
{
	if(m_pConfManager->GetConfVars()->LightingMode == 2)
	{
		DrawAreaLight(m_pShadeShadowmapRenderTarget, glm::vec3(0.f, 0.f, 0.f));
		DrawAreaLight(m_pShadeAntiradianceRenderTarget, glm::vec3(0.f, 0.f, 0.f));
	}
	else
	{
		DrawAreaLight(m_pShadeShadowmapRenderTarget);
	}	
	
	DrawAreaLight(m_pShadeAntiradianceRenderTarget, glm::vec3(0.f, 0.f, 0.f));
	
	Add(m_pResultRenderTarget, m_pShadeShadowmapRenderTarget, m_pShadeAntiradianceRenderTarget);
		
	CalculateError();
}

void Renderer::CalculateError()
{
	CRenderTargetLock lock(m_pErrorRenderTarget);

	if(scene->GetReferenceImage())
	{
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		COGLBindLock lockProgram(m_pErrorProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(m_pResultRenderTarget->GetBuffer(0), COGL_TEXTURE0_SLOT);
		COGLBindLock lock1(scene->GetReferenceImage()->GetOGLTexture(), COGL_TEXTURE1_SLOT);
		m_pFullScreenQuad->Draw();

		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
	}
}

void Renderer::DrawDebug()
{
	if(m_pConfManager->GetConfVars()->DrawDirectLighting)
		m_pTextureViewer->DrawTexture(m_pShadeShadowmapRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());

	if(m_pConfManager->GetConfVars()->DrawIndirectLighting)
		m_pTextureViewer->DrawTexture(m_pShadeAntiradianceRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	
	if(m_pConfManager->GetConfVars()->CollectAVPLs)
		std::cout << m_CollectedAVPLs.size() << " avpls collected." << std::endl;
	if(m_pConfManager->GetConfVars()->CollectISAVPLs)
		std::cout << m_CollectedImportanceSampledAVPLs.size() << " importance sampled avpls collected." << std::endl;
	
	if(m_pConfManager->GetConfVars()->DrawSceneSamples) DrawSceneSamples(m_pResultRenderTarget);
	if(m_pConfManager->GetConfVars()->DrawBIDIRSamples) DrawBidirSceneSamples(m_pResultRenderTarget);
	if(m_pConfManager->GetConfVars()->DrawCollectedAVPLs) DrawLights(m_CollectedAVPLs, m_pResultRenderTarget);
	if(m_pConfManager->GetConfVars()->DrawCollectedISAVPLs) DrawLights(m_CollectedImportanceSampledAVPLs, m_pResultRenderTarget);
	if(m_pConfManager->GetConfVars()->DrawError) m_pTextureViewer->DrawTexture(m_pErrorRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());	
	if(m_pConfManager->GetConfVars()->DrawCutSizes) m_pTextureViewer->DrawTexture(m_pGatherAntiradianceRenderTarget->GetBuffer(3), 0, 0, camera->GetWidth(), camera->GetHeight());
	
	if(m_pConfManager->GetConfVars()->DrawAVPLAtlas)
	{
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU) m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLAtlas(), 0, 0, camera->GetWidth(), camera->GetHeight());
		else m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLAtlasCPU(), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	
	if(m_pConfManager->GetConfVars()->DrawAVPLClusterAtlas)
	{
		if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU) m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLClusterAtlas(), 0, 0, camera->GetWidth(), camera->GetHeight());
		else m_pTextureViewer->DrawTexture(m_pOctahedronAtlas->GetAVPLClusterAtlasCPU(), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
		
	if(m_pConfManager->GetConfVars()->DrawReference)
	{
		if(scene->GetReferenceImage()) m_pTextureViewer->DrawTexture(scene->GetReferenceImage()->GetOGLTexture(), 0, 0, camera->GetWidth(), camera->GetHeight());
		else std::cout << "No reference image loaded" << std::endl;
	}

	if(m_pConfManager->GetConfVars()->DrawDebugTextures) 
	{
		int border = 10;
		int width = (camera->GetWidth() - 4 * border) / 2;
		int height = (camera->GetHeight() - 4 * border) / 2;
		m_pTextureViewer->DrawTexture(m_pGBuffer->GetNormalTexture(),  border, border, width, height);
		m_pTextureViewer->DrawTexture(m_pGBuffer->GetPositionTextureWS(),  3 * border + width, border, width, height);
		m_pTextureViewer->DrawTexture(m_pNormalizeAntiradianceRenderTarget->GetBuffer(2),  border, 3 * border + height, width, height);
		m_pTextureViewer->DrawTexture(m_pGBuffer->GetMaterialTexture(),  3 * border + width, 3 * border + height, width, height);
	}
}

void Renderer::CheckExport()
{
	float time = (float)m_pGlobalTimer->GetTime();

#define TIME 1

#if TIME
	bool exportData = time >= m_TimeForNextDataExport;
#else
	bool exportData = m_NumAVPLs >= m_NumAVPLsForNextDataExport;	
#endif

#if TIME
	bool exportImage = time >= m_TimeForNextImageExport;
#else
	bool exportImage = m_NumAVPLs >= m_NumAVPLsForNextImageExport;
#endif
	
	if(exportData)
	{
		float error = 0.f;
		if(scene->GetReferenceImage())
			error = scene->GetReferenceImage()->GetError(m_pResultRenderTarget->GetBuffer(0));

		float timeInSec = float(m_pGlobalTimer->GetTime())/1000.f;
		float AVPLsPerSecond = (float)m_NumAVPLs / timeInSec;
		m_pExperimentData->AddData(m_NumAVPLs, timeInSec, error, AVPLsPerSecond);
		std::cout << "AVPLs: " << m_NumAVPLs << ", time: " << timeInSec << ", AVPL_PER_SECOND: " << AVPLsPerSecond  << ", RMSE: " << error << std::endl;

#if TIME
		m_TimeForNextDataExport = int(float(m_TimeForNextDataExport) * std::pow(10.f, 0.25f));
#else
		m_NumAVPLsForNextDataExport = int(float(m_NumAVPLsForNextDataExport) * std::pow(10.f, 0.25f));
#endif
	}
	
	if(exportImage)
	{
		ExportPartialResult();

#if TIME
		m_TimeForNextImageExport *= 10;
#else
		m_NumAVPLsForNextImageExport *= 10;
#endif
	}
}

void Renderer::Gather(const std::vector<AVPL>& avpls, CRenderTarget* pRenderTarget)
{
	FillLightBuffer(avpls);	
	
	INFO info;
	info.numLights = (int)avpls.size();
	info.filterAVPLAtlas = m_pConfManager->GetConfVars()->FilterAvplAtlasLinear;
	info.UseIBL = m_pConfManager->GetConfVars()->UseIBL;
	m_pUBInfo->UpdateData(&info);
	
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	
	COGLBindLock lockProgram(m_pGatherProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	CRenderTargetLock lock(pRenderTarget);

	COGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_pGBuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_pOGLLightBuffer, COGL_TEXTURE2_SLOT);
	COGLBindLock lock3(scene->GetMaterialBuffer()->GetOGLMaterialBuffer(), COGL_TEXTURE3_SLOT);

	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::GatherWithAtlas(const std::vector<AVPL>& avpls, CRenderTarget* pRenderTarget)
{	
	if((int)avpls.size() > m_MaxNumAVPLs)
		std::cout << "To many avpls. Some are not considered" << std::endl;

	FillLightBuffer(avpls);
		
	INFO info;
	info.numLights = (int)avpls.size();
	info.UseIBL = m_pConfManager->GetConfVars()->UseIBL;
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
	COGLBindLock lock3(scene->GetMaterialBuffer()->GetOGLMaterialBuffer(), COGL_TEXTURE3_SLOT);

	if(m_pConfManager->GetConfVars()->FilterAvplAtlasLinear)
	{
		glBindSampler(4, m_pGLLinearSampler->GetResourceIdentifier());
	}
	else
	{
		glBindSampler(4, m_pGLPointSampler->GetResourceIdentifier());
	}
	
	if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU == 1)
	{
		COGLBindLock lock4(m_pOctahedronAtlas->GetAVPLAtlas(), COGL_TEXTURE4_SLOT);			
		m_pFullScreenQuad->Draw();
	}
	else
	{
		COGLBindLock lock4(m_pOctahedronAtlas->GetAVPLAtlasCPU(), COGL_TEXTURE4_SLOT);
		m_pFullScreenQuad->Draw();
	}
			
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	glBindSampler(4, m_pGLPointSampler->GetResourceIdentifier());
}

void Renderer::GatherWithClustering(const std::vector<AVPL>& avpls, CRenderTarget* pRenderTarget)
{
	if((int)avpls.size() > m_MaxNumAVPLs)
		std::cout << "To many avpls. Some are not considered" << std::endl;
	
	FillLightBuffer(avpls);
		
	INFO info;
	info.numLights = (int)avpls.size();
	info.numClusters = m_pClusterTree->GetClusteringSize();
	info.UseIBL = m_pConfManager->GetConfVars()->UseIBL;
	info.filterAVPLAtlas = m_pConfManager->GetConfVars()->FilterAvplAtlasLinear;
	info.lightTreeCutDepth = m_pConfManager->GetConfVars()->LightTreeCutDepth;
	info.clusterRefinementThreshold = m_pConfManager->GetConfVars()->ClusterRefinementThreshold;
	info.clusterRefinementMaxRadiance = m_pConfManager->GetConfVars()->ClusterRefinementMaxRadiance;
	info.clusterRefinementWeight = m_pConfManager->GetConfVars()->ClusterRefinementWeight;
	m_pUBInfo->UpdateData(&info);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	
	CRenderTargetLock lockRenderTarget(pRenderTarget);

	COGLBindLock lockProgram(m_pGatherWithClustering->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_pGBuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_pOGLLightBuffer, COGL_TEXTURE2_SLOT);
	COGLBindLock lock3(scene->GetMaterialBuffer()->GetOGLMaterialBuffer(), COGL_TEXTURE3_SLOT);
	COGLBindLock lock4(m_pOGLClusterBuffer, COGL_TEXTURE4_SLOT);

	if(m_pConfManager->GetConfVars()->FilterAvplAtlasLinear)
	{
		glBindSampler(5, m_pGLLinearSampler->GetResourceIdentifier());
		glBindSampler(6, m_pGLLinearSampler->GetResourceIdentifier());
	}
	else
	{
		glBindSampler(5, m_pGLPointSampler->GetResourceIdentifier());
		glBindSampler(6, m_pGLPointSampler->GetResourceIdentifier());
	}
	
	if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU == 1)
	{
		COGLBindLock lock5(m_pOctahedronAtlas->GetAVPLAtlas(), COGL_TEXTURE5_SLOT);
		COGLBindLock lock6(m_pOctahedronAtlas->GetAVPLClusterAtlas(), COGL_TEXTURE6_SLOT);
	
		CTimer timer(CTimer::OGL);
		if(m_ProfileFrame) timer.Start();
		m_pFullScreenQuad->Draw();
		if(m_ProfileFrame) timer.Stop("draw");
	}
	else
	{
		COGLBindLock lock5(m_pOctahedronAtlas->GetAVPLAtlasCPU(), COGL_TEXTURE5_SLOT);
		COGLBindLock lock6(m_pOctahedronAtlas->GetAVPLClusterAtlasCPU(), COGL_TEXTURE6_SLOT);
		
		CTimer timer(CTimer::OGL);
		if(m_ProfileFrame) timer.Start();
		m_pFullScreenQuad->Draw();
		if(m_ProfileFrame) timer.Stop("draw");
	}
			
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	glBindSampler(5, m_pGLPointSampler->GetResourceIdentifier());
	glBindSampler(6, m_pGLPointSampler->GetResourceIdentifier());
}

void Renderer::GatherRadianceWithShadowMap(const std::vector<AVPL>& path, CRenderTarget* pRenderTarget)
{
	for(int i = 0; i < path.size(); ++i)
	{
		GatherRadianceFromLightWithShadowMap(path[i], pRenderTarget);
	}
}

void Renderer::GatherRadianceFromLightWithShadowMap(const AVPL& avpl, CRenderTarget* pRenderTarget)
{
	FillShadowMap(avpl);
	
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
	COGLBindLock lock3(scene->GetMaterialBuffer()->GetOGLMaterialBuffer(), COGL_TEXTURE3_SLOT);

	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
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

void Renderer::Normalize(CRenderTarget* pTarget, CRenderTarget* source, int normFactor)
{
	NORMALIZE norm;
	norm.factor = 1.f / float(normFactor);
	m_pUBNormalize->UpdateData(&norm);

	CRenderTargetLock lock(pTarget);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	COGLBindLock lockProgram(m_pNormalizeProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lock0(source->GetBuffer(0), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(source->GetBuffer(1), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(source->GetBuffer(2), COGL_TEXTURE2_SLOT);

	m_pFullScreenQuad->Draw();
}

void Renderer::Shade(CRenderTarget* target, CRenderTarget* source)
{
	CRenderTargetLock lock(target);
	
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	COGLBindLock lockProgram(m_pShadeProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	if(m_pConfManager->GetConfVars()->NoAntiradiance)
	{
		COGLBindLock lock0(source->GetBuffer(1), COGL_TEXTURE0_SLOT);
		m_pFullScreenQuad->Draw();
	}
	else
	{
		COGLBindLock lock0(source->GetBuffer(0), COGL_TEXTURE0_SLOT);
		m_pFullScreenQuad->Draw();
	}
	
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
}

void Renderer::FillLightBuffer(const std::vector<AVPL>& avpls)
{
	if(avpls.size() == 0) return;

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

void Renderer::FillAVPLAtlas(const std::vector<AVPL>& avpls)
{
	if(avpls.size() == 0) return;

	m_pOctahedronAtlas->Clear();
	if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU)
	{
		m_pOctahedronAtlas->FillAtlasGPU(avpls, m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
			float(m_pConfManager->GetConfVars()->ConeFactor), 
			m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
	}
	else
	{
		m_pOctahedronAtlas->FillAtlas(avpls, m_pConfManager->GetConfVars()->NumSqrtAtlasSamples,
			float(m_pConfManager->GetConfVars()->ConeFactor), 
			m_pConfManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
	}
}

void Renderer::FillClusterAtlas(const std::vector<AVPL>& avpls)
{
	if(avpls.size() == 0) return;

	if(m_pConfManager->GetConfVars()->FillAvplAltasOnGPU) {
		m_pOctahedronAtlas->FillClusterAtlasGPU(m_pClusterTree->GetClustering(), 
			m_pClusterTree->GetClusteringSize(), (int)avpls.size());
	} else {
		m_pOctahedronAtlas->FillClusterAtlas(avpls, m_pClusterTree->GetClustering(),
			m_pClusterTree->GetClusteringSize());
	}
}

void Renderer::CreateClustering(std::vector<AVPL>& avpls)
{
	if(avpls.size() == 0) return;

	m_pClusterTree->Release();		
	m_pClusterTree->BuildTree(avpls);

	if(m_pConfManager->GetConfVars()->UseDebugMode)
		m_pClusterTree->Color(m_DebugAVPLs, m_pConfManager->GetConfVars()->ClusterDepth);

	// fill cluster information
	CLUSTER* clustering;
	int clusteringSize;
	
	clustering = m_pClusterTree->GetClustering();
	clusteringSize = m_pClusterTree->GetClusteringSize();
	
	CLUSTER_BUFFER* clusterBuffer = new CLUSTER_BUFFER[clusteringSize];
	memset(clusterBuffer, 0, sizeof(CLUSTER_BUFFER) * clusteringSize);
	for(int i = 0; i < clusteringSize; ++i)
	{
		CLUSTER_BUFFER buffer;
		clustering[i].Fill(&buffer);
		clusterBuffer[i] = buffer;
	}
	m_pOGLClusterBuffer->SetContent(clusterBuffer, sizeof(CLUSTER_BUFFER) * clusteringSize);
	delete [] clusterBuffer;
}

bool Renderer::UseAVPL(AVPL& avpl)
{
	int mode = m_pConfManager->GetConfVars()->LightingMode;

	if(mode == 0)
		return true;

	if(mode == 1)
	{
		if(avpl.GetBounce() == 0) 
			return true;
		
		if(avpl.GetBounce() == 1)
		{
			avpl.ScaleIncidentRadiance(0.f);
			return true;
		}
		
		return false;
	}
	
	if(mode == 2 && avpl.GetBounce() > 0)
	{
		if(avpl.GetBounce() == 1)
		{
			avpl.ScaleAntiradiance(0.f);
			return true;
		}

		return true;
	}

	return false;
}

void Renderer::SetUpRender()
{
	glViewport(0, 0, (GLsizei)scene->GetCamera()->GetWidth(), (GLsizei)scene->GetCamera()->GetHeight());

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);
}

void Renderer::UpdateUniformBuffers()
{
	CAMERA cam;
	cam.positionWS = scene->GetCamera()->GetPosition();
	cam.width = (int)scene->GetCamera()->GetWidth();
	cam.height = (int)scene->GetCamera()->GetHeight();
	m_pUBCamera->UpdateData(&cam);
	
	CONFIG conf;
	conf.GeoTermLimitRadiance = m_pConfManager->GetConfVars()->GeoTermLimitRadiance;
	conf.GeoTermLimitAntiradiance = m_pConfManager->GetConfVars()->GeoTermLimitAntiradiance;
	conf.ClampGeoTerm = m_pConfManager->GetConfVars()->ClampGeoTerm;
	conf.AntiradFilterK = GetAntiradFilterNormFactor();
	conf.AntiradFilterMode = m_pConfManager->GetConfVars()->AntiradFilterMode;
	conf.AntiradFilterGaussFactor = m_pConfManager->GetConfVars()->AntiradFilterGaussFactor;
	m_pUBConfig->UpdateData(&conf);

	SetTranformToCamera();
}

void Renderer::SetTranformToCamera()
{
	TRANSFORM transform;
	transform.M = IdentityMatrix();
	transform.V = scene->GetCamera()->GetViewMatrix();
	transform.itM = IdentityMatrix();
	transform.MVP = scene->GetCamera()->GetProjectionMatrix() * scene->GetCamera()->GetViewMatrix();
	m_pUBTransform->UpdateData(&transform);
}

void Renderer::CreateGBuffer()
{
	GLenum buffers [3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
	COGLRenderTargetLock lockRenderTarget(m_pGBuffer->GetRenderTarget(), 3, buffers);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	COGLBindLock lockProgram(m_pCreateGBufferProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	scene->DrawScene(m_pUBTransform, m_pUBMaterial);
}

void Renderer::DrawAreaLight(CRenderTarget* pTarget)
{
	CRenderTargetLock lockRT(pTarget);

	COGLBindLock lock(m_pAreaLightProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
	
	// avoid z-fighting
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(-1.0f, 1.0f);
	glDepthFunc(GL_LEQUAL);

	scene->DrawAreaLight(m_pUBTransform, m_pUBAreaLight);
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::DrawAreaLight(CRenderTarget* pTarget, glm::vec3 color)
{
	CRenderTargetLock lockRT(pTarget);

	COGLBindLock lock(m_pAreaLightProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
	
	// avoid z-fighting
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(-1.0f, 1.0f);
	glDepthFunc(GL_LEQUAL);

	scene->DrawAreaLight(m_pUBTransform, m_pUBAreaLight, color);
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::DrawLights(const std::vector<AVPL>& avpls, CRenderTarget* target)
{	
	if(!m_pConfManager->GetConfVars()->DrawLights)
		return;

	std::vector<POINT_CLOUD_POINT> pcp;
	
	for (int i = 0; i < avpls.size(); ++i)
	{
		int rb = m_pConfManager->GetConfVars()->RenderBounce;
		if(rb == -1 || (avpls[i].GetBounce() == rb || avpls[i].GetBounce() == rb + 1))
		{
			POINT_CLOUD_POINT p;
			p.position = glm::vec4(avpls[i].GetPosition() + m_pConfManager->GetConfVars()->DisplacePCP * avpls[i].GetOrientation(), 1.0f);
			p.color = glm::vec4(avpls[i].GetColor(), 1.f);
			pcp.push_back(p);
		}
	}

	DrawPointCloud(pcp, target);
	pcp.clear();
}

void Renderer::DrawSceneSamples(CRenderTarget* target)
{	
	std::vector<SceneSample> sceneSamples = m_pAVPLImportanceSampling->GetSceneSamples();
	std::vector<POINT_CLOUD_POINT> pcp;
	
	for (int i = 0; i < sceneSamples.size(); ++i)
	{
		POINT_CLOUD_POINT p;
		p.position = glm::vec4(sceneSamples[i].position, 1.0f);
		p.color = glm::vec4(1.f, 1.f, 0.f, 1.0f);
		pcp.push_back(p);
	}
	
	DrawPointCloud(pcp, target);
	pcp.clear();
}

void Renderer::DrawBidirSceneSamples(CRenderTarget* target)
{	
	std::vector<SceneSample> sceneSamples = m_pBidirInstantRadiosity->GetSceneSamples();
	std::vector<SceneSample> antiSceneSamples = m_pBidirInstantRadiosity->GetAntiSceneSamples();
	std::vector<SceneSample> visibles = m_pBidirInstantRadiosity->GetVisibles();
	std::vector<POINT_CLOUD_POINT> pcp;
	
	if(m_pConfManager->GetConfVars()->DrawBIDIRSamplesMode == 0)
	{
		for (int i = 0; i < sceneSamples.size(); ++i)
		{
			POINT_CLOUD_POINT p;
			p.position = glm::vec4(sceneSamples[i].position + m_pConfManager->GetConfVars()->DisplacePCP * sceneSamples[i].normal, 1.0f);
			p.color = glm::vec4(0.5f * sceneSamples[i].normal + glm::vec3(0.5f), 1.0f);
			pcp.push_back(p);
		}
	}
	
	if(m_pConfManager->GetConfVars()->DrawBIDIRSamplesMode == 1)
	{
		for (int i = 0; i < antiSceneSamples.size(); ++i)
		{
			POINT_CLOUD_POINT p;
			p.position = glm::vec4(antiSceneSamples[i].position + m_pConfManager->GetConfVars()->DisplacePCP * antiSceneSamples[i].normal, 1.0f);
			p.color = glm::vec4(0.5f * antiSceneSamples[i].normal + glm::vec3(0.5f), 1.0f);
			pcp.push_back(p);
		}
	}

	if(m_pConfManager->GetConfVars()->DrawBIDIRSamplesMode == 2)
	{
		for (int i = 0; i < visibles.size(); ++i)
		{
			POINT_CLOUD_POINT p;
			p.position = glm::vec4(visibles[i].position + m_pConfManager->GetConfVars()->DisplacePCP * visibles[i].normal, 1.0f);
			p.color = glm::vec4(0.5f * visibles[i].normal + glm::vec3(0.5f), 1.0f);
			pcp.push_back(p);
		}
	}

	DrawPointCloud(pcp, target);

	pcp.clear();
}

void Renderer::DrawPointCloud(const std::vector<POINT_CLOUD_POINT>& pcp, CRenderTarget* target)
{
	glm::vec4* positionData = new glm::vec4[pcp.size()];
	glm::vec4* colorData = new glm::vec4[pcp.size()];
	for(uint i = 0; i < pcp.size(); ++i)
	{
		positionData[i] = pcp[i].position;
		colorData[i] = pcp[i].color;
	}
		
	CRenderTargetLock lock(target);
			
	COGLBindLock lockProgram(m_pPointCloudProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
	
	SetTranformToCamera();

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);
	m_pPointCloud->Draw(positionData, colorData, (int)pcp.size());
	glDepthMask(GL_TRUE);
		
	delete [] positionData;
	delete [] colorData;
}

void Renderer::Add(CRenderTarget* target, CRenderTarget* source1, CRenderTarget* source2)
{
	CRenderTargetLock lock(target);
	
	glClear(GL_COLOR_BUFFER_BIT);

	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	COGLBindLock lockProgram(m_pAddProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lock0(source1->GetBuffer(0), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(source2->GetBuffer(0), COGL_TEXTURE1_SLOT);

	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
}

void Renderer::DirectEnvMapLighting()
{
	CRenderTargetLock lock(m_pNormalizeShadowmapRenderTarget);
	{
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);

		COGLBindLock lockProgram(m_pDirectEnvmapLighting->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
		COGLBindLock lock1(m_pGBuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
		COGLBindLock lock2(m_pCubeMap, COGL_TEXTURE2_SLOT);
		
		m_pFullScreenQuad->Draw();
		
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
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
	m_NumAVPLsForNextDataExport = 50000;
	m_NumAVPLsForNextImageExport = 10000;
	m_TimeForNextDataExport = 1000;
	m_TimeForNextImageExport = 60000;
	m_NumAVPLs = 0;

	glClearColor(0, 0, 0, 0);
	
	{
		CRenderTargetLock lock(m_pLightDebugRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pGatherShadowmapRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pGatherAntiradianceRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pNormalizeShadowmapRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pNormalizeAntiradianceRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pShadeShadowmapRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pShadeAntiradianceRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_pResultRenderTarget);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	m_pImagePlane->Clear();
	m_pExperimentData->ClearData();

	m_CurrentPathShadowmap = 0;
	m_CurrentPathAntiradiance = 0;
	m_Finished = false;
	m_FinishedDirectLighting = false;
	m_FinishedIndirectLighting = false;
	m_FinishedDebug = false;

	m_ClearAccumulationBuffer = false;
}

void Renderer::ClearLighting()
{
	scene->ClearLighting();
	m_CurrentPathShadowmap = 0;
	m_CurrentPathAntiradiance = 0;
	m_Finished = false;
	m_FinishedDirectLighting = false;
	m_FinishedIndirectLighting = false;
	m_FinishedDebug = false;

	ClearAccumulationBuffer();

	m_ClearLighting = false;
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
	m_Export->ExportHDR(m_pPostProcessRenderTarget->GetBuffer(0), "result.hdr");
	m_Export->ExportPNG(m_pPostProcessRenderTarget->GetBuffer(0), "result.png");
	
	m_pExperimentData->WriteToFile();
}

void Renderer::ExportPartialResult()
{
	int seconds = (int)(m_pGlobalTimer->GetTime() / 1000.f);
	std::stringstream ss0;
	ss0 << "result-" << m_NumAVPLs << "-numAVPLs-" << seconds <<"-sec" << ".pfm";
	
	m_Export->ExportPFM(m_pPostProcessRenderTarget->GetBuffer(0), ss0.str());
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
	
	m_pAVPLImportanceSampling->UpdateCurrentIrradiance(m_pNormalizeAntiradianceRenderTarget->GetBuffer(1));
	m_pAVPLImportanceSampling->UpdateCurrentAntiirradiance(m_pNormalizeAntiradianceRenderTarget->GetBuffer(2));
	m_pAVPLImportanceSampling->SetNumberOfSceneSamples(m_pConfManager->GetConfVars()->NumSceneSamples);
	m_pAVPLImportanceSampling->CreateSceneSamples();

	m_pCPUTimer->Start();
	if(m_pConfManager->GetConfVars()->UseBIDIR)
	{
		int numPaths = 0;
		m_pBidirInstantRadiosity->CreatePaths(m_DebugAVPLs, numPaths, false);
	}
	else
	{
		int numAVPLs = m_pConfManager->GetConfVars()->NumAVPLsDebug;
		int numPaths = 0;
		while(m_DebugAVPLs.size() < numAVPLs)
		{
			scene->CreatePath(m_DebugAVPLs);
			numPaths++;
		}
		
		m_NumPathsDebug = numPaths;
	}
	
	std::cout << "Number of AVPLs: " << m_DebugAVPLs.size() << std::endl;
	m_pCPUTimer->Stop("CreatePaths");

	/*
	m_ClusterTestAVPLs.clear();
	CreateRandomAVPLs(m_ClusterTestAVPLs, m_pConfManager->GetConfVars()->NumAVPLsDebug);
	m_pClusterTree->BuildTree(m_ClusterTestAVPLs);
	m_pClusterTree->Color(m_ClusterTestAVPLs, m_pConfManager->GetConfVars()->ClusterDepth);
	*/
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
	float coneFactor = m_pConfManager->GetConfVars()->ConeFactor;
	float K = 0.f;
	float a = 1-cos(PI/coneFactor);
	if(m_pConfManager->GetConfVars()->AntiradFilterMode == 1)
	{		
		float b = 2*(coneFactor/PI*sin(PI/coneFactor)-1);
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
	const float coneFactor = float(m_pConfManager->GetConfVars()->ConeFactor);
	const float M = float(m_pConfManager->GetConfVars()->AntiradFilterGaussFactor);
	const float s = PI / (M*coneFactor);
	const int numSteps = 1000;
	const float stepSize = PI / (numSteps * coneFactor);
	
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
	m_Finished = true;
}

void Renderer::StartCollectingAVPLs()
{
	m_pConfManager->GetConfVars()->CollectAVPLs = m_pConfManager->GetConfVarsGUI()->CollectAVPLs = 1;
}

void Renderer::EndCollectingAVPLs()
{
	m_pConfManager->GetConfVars()->CollectAVPLs = m_pConfManager->GetConfVarsGUI()->CollectAVPLs = 0;
}

void Renderer::StartCollectingISAVPLs()
{
	m_pConfManager->GetConfVars()->CollectISAVPLs = m_pConfManager->GetConfVarsGUI()->CollectISAVPLs = 1;
}

void Renderer::EndCollectingISAVPLs()
{
	m_pConfManager->GetConfVars()->CollectISAVPLs = m_pConfManager->GetConfVarsGUI()->CollectISAVPLs = 0;
}

void Renderer::RenderPathTracingReference()
{
	static long num = 1;

	bool mis = m_pConfManager->GetConfVars()->UseMIS == 1 ? true : false;
	m_pPathTracingIntegrator->Integrate(m_pConfManager->GetConfVars()->NumSamples, mis);
	m_pTextureViewer->DrawTexture(m_pImagePlane->GetOGLTexture(), 0, 0, scene->GetCamera()->GetWidth(), scene->GetCamera()->GetHeight());
		
	if(num % 1000 == 0)
	{
		double time = m_pGlobalTimer->GetTime();
		std::stringstream ss;
		ss << "ref-" << m_CurrentPathAntiradiance << num << "-" <<time << "ms" << ".pfm";
		m_Export->ExportPFM(m_pImagePlane->GetOGLTexture(), ss.str());
	}

	if(m_pConfManager->GetConfVars()->DrawReference)
	{
		if(scene->GetReferenceImage())
			m_pTextureViewer->DrawTexture(scene->GetReferenceImage()->GetOGLTexture(), 0, 0, camera->GetWidth(), camera->GetHeight());
		else
			std::cout << "No reference image loaded" << std::endl;
	}

	num++;
}

void Renderer::CreateRandomAVPLs(std::vector<AVPL>& avpls, int numAVPLs)
{
	for(int i = 0; i < numAVPLs; ++i)
	{
		const glm::vec3 position = glm::vec3(500 * Rand01(), 500 * Rand01(), 100 * Rand01());
		const glm::vec3 normal = glm::vec3(Rand01(), Rand01(), Rand01());
		const glm::vec3 L = glm::vec3(Rand01(), Rand01(), Rand01());
		const glm::vec3 A = glm::vec3(Rand01(), Rand01(), Rand01());
		const glm::vec3 w = glm::vec3(Rand01(), Rand01(), Rand01());

		AVPL a(position, normal, L, A, w, 10.f, 0, 0, scene->GetMaterialBuffer(), m_pConfManager);
		avpls.push_back(a);
	}
}

void Renderer::IssueClearLighting()
{
	m_ClearLighting = true;
}

void Renderer::IssueClearAccumulationBuffer()
{
	m_ClearAccumulationBuffer = true;
}