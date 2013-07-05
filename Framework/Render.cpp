#include "Render.h"

#include "SimpleObjects.h"
#include "ObjectClouds.h"
#include "Utils/stream.h"

#include "CudaGather.h"
#include "bvh.h"

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
#include "CImagePlane.h"
#include "CPathTracingIntegrator.h"
#include "CMaterialBuffer.h"
#include "CReferenceImage.h"
#include "CExperimentData.h"
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

#include "OGLResources\COGLResources.h"

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
#include <algorithm>

#include <omp.h>

std::vector<Light*> initialLights;

Renderer::Renderer(CCamera* m_camera, COGLContext* glContext, CConfigManager* confManager) 
	: m_camera(m_camera), m_glContext(glContext), m_confManager(confManager) 
{
	m_clContext			.reset(new COCLContext(m_glContext));
	m_cudaContext		.reset(new cuda::CudaContext());
	m_textureViewer 	.reset(new CTextureViewer());
	m_postProcess		.reset(new CPostprocess(m_confManager));
	m_fullScreenQuad	.reset(new CFullScreenQuad());
	m_shadowMap			.reset(new CShadowMap(512));
	m_experimentData	.reset(new CExperimentData());
	m_export			.reset(new CExport());
	m_clusterTree		.reset(new CClusterTree());
	
	m_depthBuffer.reset(new COGLTexture2D(m_camera->GetWidth(), m_camera->GetHeight(), GL_DEPTH_COMPONENT32F,
		GL_DEPTH_COMPONENT, GL_FLOAT, 1, false, "Renderer.m_pDepthBuffer"));

	m_testTexture.reset(new COGLTexture2D(m_camera->GetWidth(), m_camera->GetHeight(), GL_RGBA32F, 
		GL_RGBA, GL_FLOAT, 1, false, "Renderer.m_pTestTexture"));

	m_gbuffer.reset(new CGBuffer(m_camera->GetWidth(), m_camera->GetHeight(), m_depthBuffer.get()));

	m_ubTransform.reset(new COGLUniformBuffer(sizeof(TRANSFORM),	0, GL_DYNAMIC_DRAW	, "UBTransform"	));
	m_ubMaterial .reset(new COGLUniformBuffer(sizeof(MATERIAL),		0, GL_DYNAMIC_DRAW	, "UBMaterial"	));
	m_ubLight	 .reset(new COGLUniformBuffer(sizeof(AVPL_STRUCT),	0, GL_DYNAMIC_DRAW	, "UBLight"		));
	m_ubConfig	 .reset(new COGLUniformBuffer(sizeof(CONFIG),		0, GL_DYNAMIC_DRAW	, "UBConfig"	));
	m_ubCamera	 .reset(new COGLUniformBuffer(sizeof(CAMERA),		0, GL_DYNAMIC_DRAW	, "UBCamera"	));
	m_ubInfo	 .reset(new COGLUniformBuffer(sizeof(INFO),			0, GL_DYNAMIC_DRAW	, "UBInfo"		));
	m_ubAreaLight.reset(new COGLUniformBuffer(sizeof(AREA_LIGHT),	0, GL_DYNAMIC_DRAW	, "UBAreaLight"	));
	m_ubModel	 .reset(new COGLUniformBuffer(sizeof(MODEL),		0, GL_DYNAMIC_DRAW	, "UBModel"		));
	m_ubAtlasInfo.reset(new COGLUniformBuffer(sizeof(ATLAS_INFO),	0, GL_DYNAMIC_DRAW	, "UBAtlasInfo"	));
	m_ubNormalize.reset(new COGLUniformBuffer(sizeof(NORMALIZE),	0, GL_DYNAMIC_DRAW	, "UBNormalize"	));
	
	m_linearSampler		.reset(new COGLSampler(GL_LINEAR,	GL_LINEAR,	GL_CLAMP,			GL_CLAMP,			"LinearSampler"));
	m_pointSampler		.reset(new COGLSampler(GL_NEAREST,	GL_NEAREST, GL_REPEAT,			GL_REPEAT,			"PointSampler"));
	m_shadowMapSampler	.reset(new COGLSampler(GL_NEAREST,	GL_NEAREST, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER, "ShadowMapSampler"));

	m_postProcessRenderTarget			.reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 1, 0));
	m_errorRenderTarget					.reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 1, 0));
	m_gatherShadowmapRenderTarget		.reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 4, 0));
	m_gatherAntiradianceRenderTarget	.reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 4, 0));
	m_normalizeShadowmapRenderTarget	.reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 3, m_depthBuffer.get()));
	m_normalizeAntiradianceRenderTarget .reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 3, m_depthBuffer.get()));
	m_resultRenderTarget				.reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 1, m_depthBuffer.get()));
	m_lightDebugRenderTarget			.reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 1, m_depthBuffer.get()));
	m_cudaRenderTarget					.reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 4, 0));
	m_cudaRenderTargetSum				.reset(new CRenderTarget(m_camera->GetWidth(), m_camera->GetHeight(), 4, 0));

	m_gatherProgram					.reset(new CProgram("Shaders/Gather.vert"			, "Shaders/Gather.frag"					, "GatherProgram"				));
	m_gatherWithAtlas				.reset(new CProgram("Shaders/Gather.vert"			, "Shaders/GatherWithAtlas.frag"		, "GatherProgram"				));
	m_gatherWithClustering			.reset(new CProgram("Shaders/Gather.vert"			, "Shaders/GatherWithClustering.frag"	, "GatherProgram"				));
	m_normalizeProgram				.reset(new CProgram("Shaders/Gather.vert"			, "Shaders/Normalize.frag"				, "NormalizeProgram"			));
	m_errorProgram					.reset(new CProgram("Shaders/Gather.vert"			, "Shaders/Error.frag"					, "ErrorProgram"				));
	m_addProgram					.reset(new CProgram("Shaders/Gather.vert"			, "Shaders/Add.frag"					, "AddProgram"					));
	m_directEnvmapLighting			.reset(new CProgram("Shaders/Gather.vert"			, "Shaders/DirectEnvMapLighting.frag"	, "DirectEnvmapLighting"		));
	m_createGBufferProgram			.reset(new CProgram("Shaders/CreateGBuffer.vert"	, "Shaders/CreateGBuffer.frag"			, "CreateGBufferProgram"		));
	m_createSMProgram				.reset(new CProgram("Shaders/CreateSM.vert"			, "Shaders/CreateSM.frag"				, "CreateSMProgram"				));
	m_gatherRadianceWithSMProgram	.reset(new CProgram("Shaders/Gather.vert"			, "Shaders/GatherRadianceWithSM.frag"	, "GatherRadianceWithSMProgram"	));
	m_areaLightProgram				.reset(new CProgram("Shaders/DrawAreaLight.vert"	, "Shaders/DrawAreaLight.frag"			, "AreaLightProgram"			));
	m_drawOctahedronProgram 		.reset(new CProgram("Shaders/DrawOctahedron.vert"	, "Shaders/DrawOctahedron.frag"			, "DrawOctahedronProgram"		));
	m_drawSphere			 		.reset(new CProgram("Shaders/DrawSphere.vert"		, "Shaders/DrawSphere.frag"				, "DrawSphere"					));
	m_debugProgram			 		.reset(new CProgram("Shaders/Debug.vert"			, "Shaders/Debug.frag"					, "Debug"					));

	m_scene.reset(new Scene(m_camera, m_confManager, m_clContext.get()));
	m_scene->LoadCornellBox();
	m_scene->GetMaterialBuffer()->InitOCLMaterialBuffer();
	m_scene->GetMaterialBuffer()->FillOGLMaterialBuffer();

	int dim_atlas = 3072;
	int dim_tile = 16;
		
	ATLAS_INFO atlas_info;
	atlas_info.dim_atlas = dim_atlas;
	atlas_info.dim_tile = dim_tile;
	m_ubAtlasInfo->UpdateData(&atlas_info);

	m_MaxNumAVPLs = int(std::pow(float(dim_atlas) / float(dim_tile), 2.f));
	std::cout << "max num avpls: " << m_MaxNumAVPLs << std::endl;

	m_octahedronMap.reset(new COctahedronMap(dim_tile));
	m_octahedronAtlas.reset(new COctahedronAtlas(m_clContext.get(), dim_atlas, dim_tile, 
		m_MaxNumAVPLs, m_scene->GetMaterialBuffer()));
	m_octahedronMap->FillWithDebugData();

	m_cubeMap.reset(new COGLCubeMap(512, 512, GL_RGBA32F, 10, "CubeMap"));
	m_cubeMap->LoadCubeMapFromPath("Resources\\CubeMaps\\Castle\\box\\");

	m_lightBuffer	.reset(new COGLTextureBuffer(GL_RGBA32F, "LightBuffer"	));
	m_clusterBuffer .reset(new COGLTextureBuffer(GL_RGBA32F, "ClusterBuffer"));
	m_avplPositions .reset(new COGLTextureBuffer(GL_R32F, "AVPLPositions"));

	m_imagePlane				.reset(new CImagePlane(m_scene->GetCamera()));
	m_pathTracingIntegrator		.reset(new CPathTracingIntegrator(m_scene.get(), m_imagePlane.get()));
	
	m_cpuTimer			.reset(new CTimer(CTimer::CPU));
	m_glTimer			.reset(new CTimer(CTimer::OGL));
	m_globalTimer		.reset(new CTimer(CTimer::CPU));
	m_resultTimer		.reset(new CTimer(CTimer::CPU));
	m_cpuFrameProfiler	.reset(new CTimer(CTimer::CPU));
	m_gpuFrameProfiler	.reset(new CTimer(CTimer::OGL));
	m_clTimer			.reset(new CTimer(CTimer::OCL, m_clContext.get()));

	m_Finished = false;
	m_FinishedDirectLighting = false;
	m_FinishedIndirectLighting = false;
	m_FinishedDebug = false;
	m_ProfileFrame = false;
	m_NumAVPLsForNextDataExport = 1000;
	m_NumAVPLsForNextImageExport = 10000;
	m_NumAVPLs = 0;
	m_NumPathsDebug = 0;
	m_ClearLighting = false;
	m_ClearAccumulationBuffer = false;

	BindSamplers();

	UpdateUniformBuffers();

	time(&m_StartTime);
	
	InitDebugLights();
		
	ClearAccumulationBuffer();
	
	m_globalTimer->Start();

	m_cudaTargetTexture.reset(new COGLTexture2D(m_camera->GetWidth(), m_camera->GetWidth(), GL_RGBA32F,
		GL_RGBA, GL_FLOAT, 1, false, "m_cudaTarget"));
	
	m_cudaGather.reset(new CudaGather(m_camera->GetWidth(), m_camera->GetHeight(),
		m_gbuffer->GetPositionTextureWS()->GetResourceIdentifier(),
		m_gbuffer->GetNormalTexture()->GetResourceIdentifier(),
		m_cudaRenderTarget->GetTarget(0)->GetResourceIdentifier(),
		m_cudaRenderTarget->GetTarget(1)->GetResourceIdentifier(),
		m_cudaRenderTarget->GetTarget(2)->GetResourceIdentifier(),
		m_scene->GetMaterialBuffer()->getMaterials()
	));

	/*
	{
		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> normals;
		for (int i = 0; i < 20; ++i)
		{
			positions.push_back(glm::vec3(Rand01(), Rand01(), Rand01()));
			normals.push_back(glm::normalize(glm::vec3(Rand01(), Rand01(), Rand01())));
		}

		AvplBvh avplbvh(positions, normals, true);
	}
	*/
}

Renderer::~Renderer() 
{
	m_scene->GetMaterialBuffer()->ReleaseOCLMaterialBuffer();
}

void Renderer::BindSamplers() 
{	
	m_areaLightProgram->BindUniformBuffer(m_ubTransform.get(), "transform");
	m_areaLightProgram->BindUniformBuffer(m_ubAreaLight.get(), "arealight");

	m_createGBufferProgram->BindUniformBuffer(m_ubTransform.get(), "transform");
		
	m_directEnvmapLighting->BindUniformBuffer(m_ubCamera.get(), "camera");
	m_directEnvmapLighting->BindSampler(0, m_pointSampler.get());
	m_directEnvmapLighting->BindSampler(1, m_pointSampler.get());

	m_gatherProgram->BindSampler(0, m_pointSampler.get());
	m_gatherProgram->BindSampler(1, m_pointSampler.get());
	m_gatherProgram->BindSampler(2, m_pointSampler.get());
	m_gatherProgram->BindSampler(3, m_pointSampler.get());

	m_gatherProgram->BindUniformBuffer(m_ubInfo.get(), "info_block");
	m_gatherProgram->BindUniformBuffer(m_ubConfig.get(), "config");
	m_gatherProgram->BindUniformBuffer(m_ubCamera.get(), "camera");
	
	m_drawSphere->BindUniformBuffer(m_ubTransform.get(), "transform");
	m_debugProgram->BindUniformBuffer(m_ubTransform.get(), "transform");

	m_gatherRadianceWithSMProgram->BindUniformBuffer(m_ubCamera.get(), "camera");
	m_gatherRadianceWithSMProgram->BindUniformBuffer(m_ubConfig.get(), "config");	
	m_gatherRadianceWithSMProgram->BindUniformBuffer(m_ubLight.get(), "light");

	m_gatherRadianceWithSMProgram->BindSampler(0, m_shadowMapSampler.get());
	m_gatherRadianceWithSMProgram->BindSampler(1, m_pointSampler.get());
	m_gatherRadianceWithSMProgram->BindSampler(2, m_pointSampler.get());
	m_gatherRadianceWithSMProgram->BindSampler(3, m_pointSampler.get());
	
	m_gatherWithAtlas->BindSampler(0, m_pointSampler.get());
	m_gatherWithAtlas->BindSampler(1, m_pointSampler.get());
	m_gatherWithAtlas->BindSampler(2, m_pointSampler.get());
	m_gatherWithAtlas->BindSampler(3, m_pointSampler.get());
	m_gatherWithAtlas->BindSampler(4, m_linearSampler.get());

	m_gatherWithAtlas->BindUniformBuffer(m_ubInfo.get(), "info_block");
	m_gatherWithAtlas->BindUniformBuffer(m_ubConfig.get(), "config");
	m_gatherWithAtlas->BindUniformBuffer(m_ubCamera.get(), "camera");
	m_gatherWithAtlas->BindUniformBuffer(m_ubAtlasInfo.get(), "atlas_info");

	m_gatherWithClustering->BindSampler(0, m_pointSampler.get());
	m_gatherWithClustering->BindSampler(1, m_pointSampler.get());
	m_gatherWithClustering->BindSampler(2, m_pointSampler.get());
	m_gatherWithClustering->BindSampler(3, m_pointSampler.get());
	m_gatherWithClustering->BindSampler(4, m_pointSampler.get());
	m_gatherWithClustering->BindSampler(5, m_pointSampler.get());
	m_gatherWithClustering->BindSampler(6, m_pointSampler.get());
	
	m_gatherWithClustering->BindUniformBuffer(m_ubInfo.get(), "info_block");
	m_gatherWithClustering->BindUniformBuffer(m_ubConfig.get(), "config");
	m_gatherWithClustering->BindUniformBuffer(m_ubCamera.get(), "camera");
	m_gatherWithClustering->BindUniformBuffer(m_ubAtlasInfo.get(), "atlas_info");
		
	m_normalizeProgram->BindSampler(0, m_pointSampler.get());
	m_normalizeProgram->BindSampler(1, m_pointSampler.get());
	m_normalizeProgram->BindSampler(2, m_pointSampler.get());

	m_normalizeProgram->BindUniformBuffer(m_ubNormalize.get(), "norm");
	m_normalizeProgram->BindUniformBuffer(m_ubCamera.get(), "camera");

	m_errorProgram->BindSampler(0, m_pointSampler.get());
	m_errorProgram->BindSampler(1, m_pointSampler.get());
	m_errorProgram->BindUniformBuffer(m_ubCamera.get(), "camera");

	m_addProgram->BindSampler(0, m_pointSampler.get());
	m_addProgram->BindSampler(1, m_pointSampler.get());
	m_addProgram->BindUniformBuffer(m_ubCamera.get(), "camera");
		
	m_drawOctahedronProgram->BindSampler(0, m_pointSampler.get());
	m_drawOctahedronProgram->BindUniformBuffer(m_ubTransform.get(), "transform");
	m_drawOctahedronProgram->BindUniformBuffer(m_ubModel.get(), "model");
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
		CRenderTargetLock lock(m_resultRenderTarget.get());
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	}
	
	DrawLights(m_ClusterTestAVPLs, m_resultRenderTarget.get());

	m_textureViewer->DrawTexture(m_resultRenderTarget->GetTarget(0), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());
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
	
	if(m_CurrentPathAntiradiance == 0 && m_CurrentPathShadowmap == 0)
	{
		m_experimentData->Init("test", "nois.data");
		m_experimentData->MaxTime(450);

		m_globalTimer->Start();
		m_resultTimer->Start();
		m_glTimer->Start();
		CreateGBuffer();
	}

	
	std::vector<AVPL> avpls_shadowmap;
	std::vector<AVPL> avpls_antiradiance;

	if(m_ProfileFrame) timer.Start();

	GetAVPLs(avpls_shadowmap, avpls_antiradiance);

	if(m_ProfileFrame) timer.Stop("get avpls");
	if(m_ProfileFrame) timer.Start();

	if (m_confManager->GetConfVars()->gatherWithCuda) 
	{
		if (avpls_antiradiance.size() > 0) {
			m_avplBvh.reset(new AvplBvh(avpls_antiradiance, false));
		
			if(m_ProfileFrame) timer.Stop("build bvh");
		
			m_cudaGather->run_bvh(m_avplBvh.get(), m_camera->GetPosition(), m_confManager->GetConfVars()->bvhLevel, 
				m_confManager->GetConfVars()->ClusterRefinementThreshold);
			//m_cudaGather->run(avpls_antiradiance, m_camera->GetPosition());
			
			Add(m_gatherAntiradianceRenderTarget.get(), m_cudaRenderTarget.get());
		}
		
		if(m_ProfileFrame) timer.Stop("gather");
	}
	else
	{
		if(m_confManager->GetConfVars()->GatherWithAVPLClustering)
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

		if(m_confManager->GetConfVars()->GatherWithAVPLAtlas)
		{
			FillAVPLAtlas(avpls_antiradiance);
			if(m_ProfileFrame) timer.Stop("fill avpls atlas");
			if(m_ProfileFrame) timer.Start();
		}

		Gather(avpls_shadowmap, avpls_antiradiance);
		
		if(m_ProfileFrame) timer.Stop("gather");
		if(m_ProfileFrame) timer.Start();
	}
	
	Normalize(m_normalizeShadowmapRenderTarget.get(), m_gatherShadowmapRenderTarget.get(), m_CurrentPathShadowmap);
	Normalize(m_normalizeAntiradianceRenderTarget.get(), m_gatherAntiradianceRenderTarget.get(), m_CurrentPathAntiradiance);
	
	if(m_ProfileFrame) timer.Stop("normalize");
	if(m_ProfileFrame) timer.Start();

	if(m_confManager->GetConfVars()->LightingMode == 2)
	{
		DrawAreaLight(m_normalizeShadowmapRenderTarget.get(), glm::vec3(0.f, 0.f, 0.f));
		DrawAreaLight(m_normalizeAntiradianceRenderTarget.get(), glm::vec3(0.f, 0.f, 0.f));
	}
	else
	{
		DrawAreaLight(m_normalizeShadowmapRenderTarget.get());
	}
	
	DrawAreaLight(m_normalizeAntiradianceRenderTarget.get(), glm::vec3(0.f, 0.f, 0.f)); 
	
	SetTransformToCamera();

	Add(m_resultRenderTarget.get(), m_normalizeAntiradianceRenderTarget.get(), m_normalizeShadowmapRenderTarget.get());

	if (m_confManager->GetConfVars()->UseDebugMode)
	{
		if (m_confManager->GetConfVars()->DrawLights) {
			CRenderTargetLock lock(m_resultRenderTarget.get());
			m_pointCloud->Draw();
		}
		if (m_confManager->GetConfVars()->DrawAABBs) {
			CRenderTargetLock lock(m_resultRenderTarget.get());
			m_aabbCloud->Draw();
		}

		if (m_sceneProbe) {
			drawSceneProbe();
		}
	}
	DrawDebug();

	if(m_ProfileFrame) timer.Stop("draw debug");
		
	m_postProcess->Postprocess(m_resultRenderTarget->GetTarget(0), m_postProcessRenderTarget.get());
	m_textureViewer->DrawTexture(m_postProcessRenderTarget->GetTarget(0), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());	
	
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
	if(m_confManager->GetConfVars()->UseDebugMode)
	{
		if(!m_FinishedDebug)
		{
			SeparateAVPLs(m_DebugAVPLs, avpls_shadowmap, avpls_antiradiance, m_NumPathsDebug);
		}
	}
	else
	{
		if(m_confManager->GetConfVars()->SeparateDirectIndirectLighting && m_confManager->GetConfVars()->LightingMode != 2)
		{
			if(m_CurrentPathShadowmap < m_confManager->GetConfVars()->NumVPLsDirectLight)
			{
				m_scene->CreatePrimaryVpls(avpls_shadowmap, m_confManager->GetConfVars()->NumVPLsDirectLightPerFrame);
				m_CurrentPathShadowmap += m_confManager->GetConfVars()->NumVPLsDirectLightPerFrame;
			}
			else if(!m_FinishedDirectLighting)
			{
				std::cout << "Finished direct lighting" << std::endl;
				m_FinishedDirectLighting = true;
			}
		}
		
		std::vector<AVPL> avpls;
		int numAVPLs = m_confManager->GetConfVars()->NumAVPLsPerFrame;
		int numAVPLsPerBatch = numAVPLs > 1000 ? 1000 : numAVPLs;
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
					m_scene->CreatePath(avpls_thread);
					numPaths_thread++;
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
	if(!m_confManager->GetConfVars()->UseAntiradiance)
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
	
	if(!m_confManager->GetConfVars()->SeparateDirectIndirectLighting)
	{
		for(int i = 0; i < avpls.size(); ++i)
		{
			AVPL avpl = avpls[i];
			if(UseAVPL(avpl))
				avpls_antiradiance.push_back(avpl);
		}
		m_CurrentPathAntiradiance += numPaths;
		m_numPathsAntiradiance = numPaths;
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
	m_numPathsAntiradiance = numPaths;
}

void Renderer::Gather(std::vector<AVPL>& avpls_shadowmap, std::vector<AVPL>& avpls_antiradiance)
{
	if(m_confManager->GetConfVars()->UseDebugMode && m_FinishedDebug)
		return;
	
	if(m_confManager->GetConfVars()->GatherWithAVPLClustering)
		GatherWithClustering(avpls_antiradiance, m_gatherAntiradianceRenderTarget.get());
	else if(m_confManager->GetConfVars()->GatherWithAVPLAtlas)
		GatherWithAtlas(avpls_antiradiance, m_gatherAntiradianceRenderTarget.get());
	else
		Gather(avpls_antiradiance, m_gatherAntiradianceRenderTarget.get());

	GatherRadianceWithShadowMap(avpls_shadowmap, m_gatherShadowmapRenderTarget.get());
}

void Renderer::CalculateError()
{
	CRenderTargetLock lock(m_errorRenderTarget.get());

	if(m_scene->GetReferenceImage())
	{
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);

		COGLBindLock lockProgram(m_errorProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(m_resultRenderTarget->GetTarget(0), COGL_TEXTURE0_SLOT);
		COGLBindLock lock1(m_scene->GetReferenceImage()->GetOGLTexture(), COGL_TEXTURE1_SLOT);
		m_fullScreenQuad->Draw();

		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
	}
}

void Renderer::DrawDebug()
{
	if(m_confManager->GetConfVars()->DrawError) m_textureViewer->DrawTexture(m_errorRenderTarget->GetTarget(0), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());	
	if(m_confManager->GetConfVars()->DrawCutSizes) m_textureViewer->DrawTexture(m_gatherAntiradianceRenderTarget->GetTarget(3), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());
	
	if(m_confManager->GetConfVars()->DrawAVPLAtlas)
	{
		if(m_confManager->GetConfVars()->FillAvplAltasOnGPU) m_textureViewer->DrawTexture(m_octahedronAtlas->GetAVPLAtlas(), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());
		else m_textureViewer->DrawTexture(m_octahedronAtlas->GetAVPLAtlasCPU(), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());
	}
	
	if(m_confManager->GetConfVars()->DrawAVPLClusterAtlas)
	{
		if(m_confManager->GetConfVars()->FillAvplAltasOnGPU) m_textureViewer->DrawTexture(m_octahedronAtlas->GetAVPLClusterAtlas(), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());
		else m_textureViewer->DrawTexture(m_octahedronAtlas->GetAVPLClusterAtlasCPU(), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());
	}
		
	if(m_confManager->GetConfVars()->DrawReference)
	{
		if(m_scene->GetReferenceImage()) m_textureViewer->DrawTexture(m_scene->GetReferenceImage()->GetOGLTexture(), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());
		else std::cout << "No reference image loaded" << std::endl;
	}

	if(m_confManager->GetConfVars()->DrawDebugTextures) 
	{
		int border = 10;
		int width = (m_camera->GetWidth() - 4 * border) / 2;
		int height = (m_camera->GetHeight() - 4 * border) / 2;
		m_textureViewer->DrawTexture(m_gbuffer->GetNormalTexture(),  border, border, width, height);
		m_textureViewer->DrawTexture(m_gbuffer->GetPositionTextureWS(),  3 * border + width, border, width, height);
		m_textureViewer->DrawTexture(m_normalizeAntiradianceRenderTarget->GetTarget(2),  border, 3 * border + height, width, height);
		m_textureViewer->DrawTexture(m_depthBuffer.get(),  3 * border + width, 3 * border + height, width, height);
	}
}

void Renderer::CheckExport()
{
	float time = (float)m_globalTimer->GetTime();

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
		if(m_scene->GetReferenceImage())
			error = m_scene->GetReferenceImage()->GetError(m_resultRenderTarget->GetTarget(0));

		float timeInSec = float(m_globalTimer->GetTime())/1000.f;
		float AVPLsPerSecond = (float)m_NumAVPLs / timeInSec;
		m_experimentData->AddData(m_NumAVPLs, timeInSec, error, AVPLsPerSecond);
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
	info.filterAVPLAtlas = m_confManager->GetConfVars()->FilterAvplAtlasLinear;
	info.UseIBL = m_confManager->GetConfVars()->UseIBL;
	m_ubInfo->UpdateData(&info);
	
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	
	COGLBindLock lockProgram(m_gatherProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	CRenderTargetLock lock(pRenderTarget);

	COGLBindLock lock0(m_gbuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_gbuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_lightBuffer.get(), COGL_TEXTURE2_SLOT);
	COGLBindLock lock3(m_scene->GetMaterialBuffer()->GetOGLMaterialBuffer(), COGL_TEXTURE3_SLOT);

	m_fullScreenQuad->Draw();
	
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
	info.UseIBL = m_confManager->GetConfVars()->UseIBL;
	info.filterAVPLAtlas = m_confManager->GetConfVars()->FilterAvplAtlasLinear;
	m_ubInfo->UpdateData(&info);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	
	CRenderTargetLock lockRenderTarget(pRenderTarget);

	COGLBindLock lockProgram(m_gatherWithAtlas->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lock0(m_gbuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_gbuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_lightBuffer.get(), COGL_TEXTURE2_SLOT);
	COGLBindLock lock3(m_scene->GetMaterialBuffer()->GetOGLMaterialBuffer(), COGL_TEXTURE3_SLOT);

	if(m_confManager->GetConfVars()->FilterAvplAtlasLinear)
	{
		glBindSampler(4, m_linearSampler->GetResourceIdentifier());
	}
	else
	{
		glBindSampler(4, m_pointSampler->GetResourceIdentifier());
	}
	
	if(m_confManager->GetConfVars()->FillAvplAltasOnGPU == 1)
	{
		COGLBindLock lock4(m_octahedronAtlas->GetAVPLAtlas(), COGL_TEXTURE4_SLOT);			
		m_fullScreenQuad->Draw();
	}
	else
	{
		COGLBindLock lock4(m_octahedronAtlas->GetAVPLAtlasCPU(), COGL_TEXTURE4_SLOT);
		m_fullScreenQuad->Draw();
	}
			
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	glBindSampler(4, m_pointSampler->GetResourceIdentifier());
}

void Renderer::GatherWithClustering(const std::vector<AVPL>& avpls, CRenderTarget* pRenderTarget)
{
	if((int)avpls.size() > m_MaxNumAVPLs)
		std::cout << "To many avpls. Some are not considered" << std::endl;
	
	FillLightBuffer(avpls);
		
	INFO info;
	info.numLights = (int)avpls.size();
	info.numClusters = m_clusterTree->GetClusteringSize();
	info.UseIBL = m_confManager->GetConfVars()->UseIBL;
	info.filterAVPLAtlas = m_confManager->GetConfVars()->FilterAvplAtlasLinear;
	info.lightTreeCutDepth = m_confManager->GetConfVars()->LightTreeCutDepth;
	info.clusterRefinementThreshold = m_confManager->GetConfVars()->ClusterRefinementThreshold;
	info.clusterRefinementMaxRadiance = m_confManager->GetConfVars()->ClusterRefinementMaxRadiance;
	info.clusterRefinementWeight = m_confManager->GetConfVars()->ClusterRefinementWeight;
	m_ubInfo->UpdateData(&info);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	
	CRenderTargetLock lockRenderTarget(pRenderTarget);

	COGLBindLock lockProgram(m_gatherWithClustering->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lock0(m_gbuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_gbuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_lightBuffer.get(), COGL_TEXTURE2_SLOT);
	COGLBindLock lock3(m_scene->GetMaterialBuffer()->GetOGLMaterialBuffer(), COGL_TEXTURE3_SLOT);
	COGLBindLock lock4(m_clusterBuffer.get(), COGL_TEXTURE4_SLOT);

	if(m_confManager->GetConfVars()->FilterAvplAtlasLinear)
	{
		glBindSampler(5, m_linearSampler->GetResourceIdentifier());
		glBindSampler(6, m_linearSampler->GetResourceIdentifier());
	}
	else
	{
		glBindSampler(5, m_pointSampler->GetResourceIdentifier());
		glBindSampler(6, m_pointSampler->GetResourceIdentifier());
	}
	
	if(m_confManager->GetConfVars()->FillAvplAltasOnGPU == 1)
	{
		COGLBindLock lock5(m_octahedronAtlas->GetAVPLAtlas(), COGL_TEXTURE5_SLOT);
		COGLBindLock lock6(m_octahedronAtlas->GetAVPLClusterAtlas(), COGL_TEXTURE6_SLOT);
	
		CTimer timer(CTimer::OGL);
		if(m_ProfileFrame) timer.Start();
		m_fullScreenQuad->Draw();
		if(m_ProfileFrame) timer.Stop("draw");
	}
	else
	{
		COGLBindLock lock5(m_octahedronAtlas->GetAVPLAtlasCPU(), COGL_TEXTURE5_SLOT);
		COGLBindLock lock6(m_octahedronAtlas->GetAVPLClusterAtlasCPU(), COGL_TEXTURE6_SLOT);
		
		CTimer timer(CTimer::OGL);
		if(m_ProfileFrame) timer.Start();
		m_fullScreenQuad->Draw();
		if(m_ProfileFrame) timer.Stop("draw");
	}
			
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	glBindSampler(5, m_pointSampler->GetResourceIdentifier());
	glBindSampler(6, m_pointSampler->GetResourceIdentifier());
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
	glViewport(0, 0, m_camera->GetWidth(), m_camera->GetHeight());
		
	COGLBindLock lockProgram(m_gatherRadianceWithSMProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	AVPL_STRUCT light_info;
	avpl.Fill(light_info);
	m_ubLight->UpdateData(&light_info);

	CRenderTargetLock lock(pRenderTarget);

	COGLBindLock lock0(m_shadowMap->GetShadowMapTexture(), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(m_gbuffer->GetPositionTextureWS(), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(m_gbuffer->GetNormalTexture(), COGL_TEXTURE2_SLOT);
	COGLBindLock lock3(m_scene->GetMaterialBuffer()->GetOGLMaterialBuffer(), COGL_TEXTURE3_SLOT);

	m_fullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::FillShadowMap(const AVPL& avpl)
{
	glEnable(GL_DEPTH_TEST);

	// prevent surface acne
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.1f, 4.0f);
	
	COGLBindLock lockProgram(m_createSMProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
	GLenum buffer[1] = {GL_NONE};
	COGLRenderTargetLock lock(m_shadowMap->GetRenderTarget(), 1, buffer);

	glViewport(0, 0, m_shadowMap->GetShadowMapSize(), m_shadowMap->GetShadowMapSize());
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	m_scene->DrawScene(avpl.GetViewMatrix(), avpl.GetProjectionMatrix(), m_ubTransform.get());

	glViewport(0, 0, m_camera->GetWidth(), m_camera->GetHeight());
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::Normalize(CRenderTarget* pTarget, CRenderTarget* source, int normFactor)
{
	NORMALIZE norm;
	if (normFactor == 0) {
		norm.factor = 0.f;
	} else {
		norm.factor = 1.f / float(normFactor);
	}
	m_ubNormalize->UpdateData(&norm);

	CRenderTargetLock lock(pTarget);

	glDisable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT);

	COGLBindLock lockProgram(m_normalizeProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lock0(source->GetTarget(0), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(source->GetTarget(1), COGL_TEXTURE1_SLOT);
	COGLBindLock lock2(source->GetTarget(2), COGL_TEXTURE2_SLOT);

	m_fullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
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
	m_lightBuffer->SetContent(sizeof(AVPL_BUFFER) * avpls.size(), GL_DYNAMIC_READ, avplBuffer);
	delete [] avplBuffer;
}

void Renderer::FillAVPLAtlas(const std::vector<AVPL>& avpls)
{
	if(avpls.size() == 0) return;

	m_octahedronAtlas->Clear();
	if(m_confManager->GetConfVars()->FillAvplAltasOnGPU)
	{
		m_octahedronAtlas->FillAtlasGPU(avpls, m_confManager->GetConfVars()->NumSqrtAtlasSamples,
			float(m_confManager->GetConfVars()->ConeFactor), 
			m_confManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
	}
	else
	{
		m_octahedronAtlas->FillAtlas(avpls, m_confManager->GetConfVars()->NumSqrtAtlasSamples,
			float(m_confManager->GetConfVars()->ConeFactor), 
			m_confManager->GetConfVars()->FilterAvplAtlasLinear == 1 ? true : false);
	}
}

void Renderer::FillClusterAtlas(const std::vector<AVPL>& avpls)
{
	if(avpls.size() == 0) return;

	if(m_confManager->GetConfVars()->FillAvplAltasOnGPU) {
		m_octahedronAtlas->FillClusterAtlasGPU(m_clusterTree->GetClustering(), 
			m_clusterTree->GetClusteringSize(), (int)avpls.size());
	} else {
		m_octahedronAtlas->FillClusterAtlas(avpls, m_clusterTree->GetClustering(),
			m_clusterTree->GetClusteringSize());
	}
}

void Renderer::CreateClustering(std::vector<AVPL>& avpls)
{
	if(avpls.size() == 0) return;

	m_clusterTree->Release();		
	m_clusterTree->BuildTree(avpls);

	if(m_confManager->GetConfVars()->UseDebugMode)
		m_clusterTree->Color(m_DebugAVPLs, m_confManager->GetConfVars()->ClusterDepth);

	// fill cluster information
	CLUSTER* clustering;
	int clusteringSize;
	
	clustering = m_clusterTree->GetClustering();
	clusteringSize = m_clusterTree->GetClusteringSize();
	
	CLUSTER_BUFFER* clusterBuffer = new CLUSTER_BUFFER[clusteringSize];
	memset(clusterBuffer, 0, sizeof(CLUSTER_BUFFER) * clusteringSize);
	for(int i = 0; i < clusteringSize; ++i)
	{
		CLUSTER_BUFFER buffer;
		clustering[i].Fill(&buffer);
		clusterBuffer[i] = buffer;
	}
	m_clusterBuffer->SetContent(sizeof(CLUSTER_BUFFER) * clusteringSize, GL_DYNAMIC_READ, clusterBuffer);
	delete [] clusterBuffer;
}

bool Renderer::UseAVPL(AVPL& avpl)
{
	int mode = m_confManager->GetConfVars()->LightingMode;

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
	glViewport(0, 0, (GLsizei)m_scene->GetCamera()->GetWidth(), (GLsizei)m_scene->GetCamera()->GetHeight());

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);
}

void Renderer::UpdateUniformBuffers()
{
	CAMERA cam;
	cam.positionWS = m_scene->GetCamera()->GetPosition();
	cam.width = (int)m_scene->GetCamera()->GetWidth();
	cam.height = (int)m_scene->GetCamera()->GetHeight();
	m_ubCamera->UpdateData(&cam);
	
	CONFIG conf;
	conf.GeoTermLimitRadiance = m_confManager->GetConfVars()->GeoTermLimitRadiance;
	conf.GeoTermLimitAntiradiance = m_confManager->GetConfVars()->GeoTermLimitAntiradiance;
	conf.ClampGeoTerm = m_confManager->GetConfVars()->ClampGeoTerm;
	conf.AntiradFilterK = GetAntiradFilterNormFactor();
	conf.AntiradFilterMode = m_confManager->GetConfVars()->AntiradFilterMode;
	conf.AntiradFilterGaussFactor = m_confManager->GetConfVars()->AntiradFilterGaussFactor;
	m_ubConfig->UpdateData(&conf);

	SetTransformToCamera();
}

void Renderer::SetTransformToCamera()
{
	TRANSFORM transform;
	transform.M = IdentityMatrix();
	transform.V = m_scene->GetCamera()->GetViewMatrix();
	transform.itM = IdentityMatrix();
	transform.MVP = m_scene->GetCamera()->GetProjectionMatrix() * m_scene->GetCamera()->GetViewMatrix();
	m_ubTransform->UpdateData(&transform);
}

void Renderer::CreateGBuffer()
{
	GLenum buffers [3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
	COGLRenderTargetLock lockRenderTarget(m_gbuffer->GetRenderTarget(), 3, buffers);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	COGLBindLock lockProgram(m_createGBufferProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	m_scene->DrawScene(m_ubTransform.get(), m_ubMaterial.get());
}

void Renderer::DrawAreaLight(CRenderTarget* pTarget)
{
	CRenderTargetLock lockRT(pTarget);

	COGLBindLock lock(m_areaLightProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
	
	// avoid z-fighting
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(-1.0f, 1.0f);
	glDepthFunc(GL_LEQUAL);

	m_scene->DrawAreaLight(m_ubTransform.get(), m_ubAreaLight.get());
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::DrawAreaLight(CRenderTarget* pTarget, glm::vec3 color)
{
	CRenderTargetLock lockRT(pTarget);

	COGLBindLock lock(m_areaLightProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	// avoid z-fighting
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(-1.0f, 1.0f);
	glDepthFunc(GL_LEQUAL);

	m_scene->DrawAreaLight(m_ubTransform.get(), m_ubAreaLight.get(), color);
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::DrawLights(const std::vector<AVPL>& avpls, CRenderTarget* target)
{	
	if(!m_confManager->GetConfVars()->DrawLights || avpls.size() == 0)
		return;

	std::vector<glm::vec3> positions(avpls.size());
	std::vector<glm::vec3> colors(avpls.size());
	for (int i = 0; i < avpls.size(); ++i) {
		positions[i] = avpls[i].GetPosition();
		colors[i] = glm::vec3(1.f, 0.f, 1.f);
	}
	std::shared_ptr<PointCloud> pointCloud = std::make_shared<PointCloud>(positions, colors, m_ubTransform.get());

	CRenderTargetLock lock(target);
	pointCloud->Draw();
}

void Renderer::Add(CRenderTarget* target, CRenderTarget* source1, CRenderTarget* source2)
{
	CRenderTargetLock lock(target);
	
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);

	COGLBindLock lockProgram(m_addProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lock0(source1->GetTarget(0), COGL_TEXTURE0_SLOT);
	COGLBindLock lock1(source2->GetTarget(0), COGL_TEXTURE1_SLOT);

	m_fullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
}

void Renderer::Add(CRenderTarget* target, CRenderTarget* source)
{
	CRenderTargetLock lock(target);
	
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);

	COGLBindLock lockProgram(m_addProgram->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lock0(source->GetTarget(0), COGL_TEXTURE0_SLOT);

	m_fullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDisable(GL_BLEND);
}

void Renderer::DirectEnvMapLighting()
{
	CRenderTargetLock lock(m_normalizeShadowmapRenderTarget.get());
	{
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);

		COGLBindLock lockProgram(m_directEnvmapLighting->GetGLProgram(), COGL_PROGRAM_SLOT);

		COGLBindLock lock0(m_gbuffer->GetPositionTextureWS(), COGL_TEXTURE0_SLOT);
		COGLBindLock lock1(m_gbuffer->GetNormalTexture(), COGL_TEXTURE1_SLOT);
		COGLBindLock lock2(m_cubeMap.get(), COGL_TEXTURE2_SLOT);
		
		m_fullScreenQuad->Draw();
		
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
	}
}

void Renderer::WindowChanged()
{
	m_depthBuffer.reset(new COGLTexture2D(m_camera->GetWidth(), m_camera->GetHeight(), GL_DEPTH_COMPONENT32F, 
		GL_DEPTH, GL_FLOAT, 1, false));

	m_gbuffer.reset(new CGBuffer(m_camera->GetWidth(), m_camera->GetHeight(), m_depthBuffer.get()));
	
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
		CRenderTargetLock lock(m_lightDebugRenderTarget.get());
		glClear(GL_COLOR_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_gatherShadowmapRenderTarget.get());
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_gatherAntiradianceRenderTarget.get());
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_normalizeShadowmapRenderTarget.get());
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_normalizeAntiradianceRenderTarget.get());
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	{
		CRenderTargetLock lock(m_resultRenderTarget.get());
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}
	
	{
		CRenderTargetLock lock(m_cudaRenderTargetSum.get());
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	m_imagePlane->Clear();
	m_experimentData->ClearData();

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
	m_scene->ClearLighting();
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
	std::cout << "position: (" << m_scene->GetCamera()->GetPosition().x << ", " 
		<< m_scene->GetCamera()->GetPosition().y << ", " 
		<< m_scene->GetCamera()->GetPosition().z << ")" << std::endl;
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
	m_export->ExportHDR(m_postProcessRenderTarget->GetTarget(0), "result.hdr");
	m_export->ExportPNG(m_postProcessRenderTarget->GetTarget(0), "result.png");
	
	m_experimentData->WriteToFile();
}

void Renderer::ExportPartialResult()
{
	int seconds = (int)(m_globalTimer->GetTime() / 1000.f);
	std::stringstream ss0;
	ss0 << "result-" << m_NumAVPLs << "-numAVPLs-" << seconds <<"-sec" << ".pfm";
	
	m_export->ExportPFM(m_postProcessRenderTarget->GetTarget(0), ss0.str());
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
	m_cpuTimer->Start();

	int numAVPLs = m_confManager->GetConfVars()->NumAVPLsDebug;
	int numPaths = 0;
	while(m_DebugAVPLs.size() < numAVPLs)
	{
		m_scene->CreatePath(m_DebugAVPLs);
		numPaths++;
	}
	
	m_NumPathsDebug = numPaths;
	
	std::cout << "Number of AVPLs: " << m_DebugAVPLs.size() << std::endl;
	m_cpuTimer->Stop("CreatePaths");
	
	RebuildBvh();
}

void Renderer::UpdateBvhDebug()
{
	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> normals;
	for (int i = 0; i < m_DebugAVPLs.size(); ++i)
	{
		if (m_DebugAVPLs[i].GetBounce() > 0) {
			positions.push_back(m_DebugAVPLs[i].GetPosition());
			normals.push_back(m_DebugAVPLs[i].GetOrientation());
		}
	}

	if (positions.size() > 1) {
		m_avplBvh->generateDebugInfo(m_confManager->GetConfVars()->bvhLevel);
	}
	m_pointCloud.reset(new PointCloud(m_avplBvh->getPositions(), m_avplBvh->getColors(), m_ubTransform.get()));
	m_aabbCloud.reset(new AABBCloud(m_avplBvh->getBBMins(), m_avplBvh->getBBMaxs(), m_ubTransform.get()));
}

void Renderer::RebuildBvh()
{
	if (m_DebugAVPLs.size() > 1) 
	{
		m_avplBvh.reset(nullptr);
		m_avplBvh.reset(new AvplBvh(m_DebugAVPLs, m_confManager->GetConfVars()->considerNormals));
		m_avplBvh->generateDebugInfo(m_confManager->GetConfVars()->bvhLevel);
	}

	UpdateBvhDebug();
}

void Renderer::UpdateAreaLights()
{
	m_scene->UpdateAreaLights();
}

float Renderer::GetAntiradFilterNormFactor()
{
	float coneFactor = m_confManager->GetConfVars()->ConeFactor;
	float K = 0.f;
	float a = 1-cos(M_PI/coneFactor);
	if(m_confManager->GetConfVars()->AntiradFilterMode == 1)
	{		
		float b = 2*(coneFactor/M_PI*sin(M_PI/coneFactor)-1);
		K = - a / b;
	}
	else if(m_confManager->GetConfVars()->AntiradFilterMode == 2)
	{
		float b = IntegrateGauss();
		K = a / b;
	}

	m_confManager->GetConfVars()->AntiradFilterK = K;

	return K;
}

float Renderer::IntegrateGauss()
{
	const float coneFactor = float(m_confManager->GetConfVars()->ConeFactor);
	const float M = float(m_confManager->GetConfVars()->AntiradFilterGaussFactor);
	const float s = M_PI / (M*coneFactor);
	const int numSteps = 1000;
	const float stepSize = M_PI / (numSteps * coneFactor);
	
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

void Renderer::RenderPathTracingReference()
{
	static long num = 1;

	bool mis = m_confManager->GetConfVars()->UseMIS == 1 ? true : false;
	m_pathTracingIntegrator->Integrate(m_confManager->GetConfVars()->NumSamples, mis);
	m_textureViewer->DrawTexture(m_imagePlane->GetOGLTexture(), 0, 0, m_scene->GetCamera()->GetWidth(), m_scene->GetCamera()->GetHeight());
		
	if(num % 1000 == 0)
	{
		double time = m_globalTimer->GetTime();
		std::stringstream ss;
		ss << "ref-" << m_CurrentPathAntiradiance << num << "-" <<time << "ms" << ".pfm";
		m_export->ExportPFM(m_imagePlane->GetOGLTexture(), ss.str());
	}

	if(m_confManager->GetConfVars()->DrawReference)
	{
		if(m_scene->GetReferenceImage())
			m_textureViewer->DrawTexture(m_scene->GetReferenceImage()->GetOGLTexture(), 0, 0, m_camera->GetWidth(), m_camera->GetHeight());
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

		AVPL a(position, normal, L, A, w, 10.f, 0, 0, m_scene->GetMaterialBuffer(), m_confManager);
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

void Renderer::shootSceneProbe(int x, int y)
{
	Ray ray = m_camera->GetEyeRay(x, y);
	Intersection intersection;
	float t;
	if (m_scene->IntersectRayScene(ray, &t, &intersection, Triangle::FRONT_FACE)) {
		m_sceneProbe.reset(new Sphere(10));
		glm::mat4 T = glm::translate(glm::mat4(), intersection.getPosition());
		glm::mat4 S = glm::scale(glm::mat4(), glm::vec3(10.f));
		m_sceneProbe->setTransform(T * S);
	}
}

void Renderer::drawSceneProbe()
{
	CRenderTargetLock lock(m_resultRenderTarget.get());
	COGLBindLock lockProgram(m_debugProgram->GetGLProgram(), COGL_PROGRAM_SLOT);
	
	TRANSFORM transform;
	transform.M = m_sceneProbe->getTransform();
	transform.V = m_scene->GetCamera()->GetViewMatrix();
	transform.itM = glm::inverse(transform.M);
	transform.MVP = m_scene->GetCamera()->GetProjectionMatrix() * transform.V * transform.M; 
	m_ubTransform->UpdateData(&transform);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	m_sceneProbe->getMesh()->draw();
	SetTransformToCamera();
	glEnable(GL_CULL_FACE);
}