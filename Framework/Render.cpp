#include "Render.h"

#include <glm/gtc/type_ptr.hpp>

#include "Macros.h"
#include "Structs.h"

#include "CAccumulationBuffer.h"
#include "CGBuffer.h"
#include "CTimer.h"
#include "CProgram.h"
#include "CPointCloud.h"

#include "Scene.h"
#include "Camera.h"
#include "Light.h"
#include "CShadowMap.h"
#include "Postprocessing.h"

#include "CUtils\Util.h"
#include "CUtils\GLErrorUtil.h"
#include "CUtils\ShaderUtil.h"
#include "CUtils\CTextureViewer.h"

#include "CMeshResources\CFullScreenQuad.h"

#include "CGLResources\CGLUniformBuffer.h"
#include "CGLResources\CGLTexture2D.h"
#include "CGLResources\CGLFrameBuffer.h"
#include "CGLResources\CGLProgram.h"
#include "CGLResources\CGLSampler.h"
#include "CGLResources\CGLTextureBuffer.h"

#include <memory>

std::vector<Light*> initialLights;

Renderer::Renderer(Camera* _camera) {
	camera = _camera;

	m_pShadowMap = new CShadowMap();
	scene = 0;
	postProcessing = 0;
	
	m_pDepthBuffer = new CGLTexture2D("Renderer.m_pDepthBuffer");
	
	m_pGBuffer = new CGBuffer();
	m_pAccumulationRadiance = new CAccumulationBuffer();
	m_pAccumulationAntiradiance = new CAccumulationBuffer();
	m_pNormalizedRadiance = new CAccumulationBuffer();
	m_pNormalizedAntiradiance = new CAccumulationBuffer();
	m_pFinalResult = new CAccumulationBuffer();

	m_pTextureViewer = new CTextureViewer();

	m_pUBTransform = new CGLUniformBuffer("Renderer.m_pUBTransform");
	m_pUBMaterial = new CGLUniformBuffer("Renderer.m_pUBMaterial");
	m_pUBLight = new CGLUniformBuffer("Renderer.m_pUBLight");
	m_pUBConfig = new CGLUniformBuffer("Renderer.m_pUBConfig");
	m_pUBCamera = new CGLUniformBuffer("Renderer.m_pUBCamera");
	m_pUBInfo = new CGLUniformBuffer("Renderer.m_pUBInfo");

	m_pGLPointSampler = new CGLSampler("Renderer.m_pGLPointSampler");
	m_pGLShadowMapSampler = new CGLSampler("Renderer.m_pGLShadowMapSampler");

	m_pCreateGBufferProgram = new CProgram("Renderer.m_pCreateGBufferProgram", "Shaders\\CreateGBuffer.vert", "Shaders\\CreateGBuffer.frag");
	m_pCreateSMProgram = new CProgram("Renderer.m_pCreateSMProgram", "Shaders\\CreateSM.vert", "Shaders\\CreateSM.frag");
	m_pGatherRadianceWithSMProgram = new CProgram("Renderer.m_pGatherRadianceWithSMProgram", "Shaders\\Gather.vert", "Shaders\\GatherRadianceWithSM.frag");
	m_pGatherRadianceProgram = new CProgram("Renderer.m_pGatherRadianceProgram", "Shaders\\Gather.vert", "Shaders\\GatherRadiance.frag");
	m_pGatherAntiradianceProgram = new CProgram("Renderer.m_pGatherAntiradianceProgram", "Shaders\\Gather.vert", "Shaders\\GatherAntiradiance.frag");
	m_pFinalGatherProgram = new CProgram("Renderer.m_pFinalGatherProgram", "Shaders\\Gather.vert", "Shaders\\FinalGather.frag");
	m_pNormalizeRadianceProgram = new CProgram("Renderer.m_pNormalizeRadianceProgram", "Shaders\\Gather.vert", "Shaders\\Normalize.frag");
	m_pPointCloudProgram = new CProgram("Renderer.m_pPointCloudProgram", "Shaders\\PointCloud.vert", "Shaders\\PointCloud.frag");

	m_pFullScreenQuad = new CFullScreenQuad();

	m_pPointCloud = new CPointCloud();

	m_pLightBuffer = new CGLTextureBuffer("Renderer.m_pLightBuffer");

	m_BlurSigma = .1f;
	m_RenderBounce = -1;

	m_Frame = 0;
	m_CurrentPath = 0;
	m_N = 50;
	m_NumAdditionalAVPLs = 0;
	m_UseHammersley = false;

	m_NumPaths = 1;
	m_NumPathsPerFrame = 500;

	m_Finished = false;
}

Renderer::~Renderer() {
	SAFE_DELETE(scene);
	SAFE_DELETE(m_pShadowMap);
	SAFE_DELETE(postProcessing);
	SAFE_DELETE(m_pGBuffer);
	SAFE_DELETE(m_pAccumulationRadiance);
	SAFE_DELETE(m_pAccumulationAntiradiance);
	SAFE_DELETE(m_pNormalizedRadiance);
	SAFE_DELETE(m_pNormalizedAntiradiance);
	SAFE_DELETE(m_pFinalResult);
	SAFE_DELETE(m_pTextureViewer);
	SAFE_DELETE(m_pFullScreenQuad);
	SAFE_DELETE(m_pPointCloud);

	SAFE_DELETE(m_pUBTransform);
	SAFE_DELETE(m_pUBMaterial);
	SAFE_DELETE(m_pUBLight);
	SAFE_DELETE(m_pUBConfig);
	SAFE_DELETE(m_pUBCamera);
	SAFE_DELETE(m_pUBInfo);

	SAFE_DELETE(m_pCreateGBufferProgram);
	SAFE_DELETE(m_pCreateSMProgram);
	SAFE_DELETE(m_pGatherRadianceProgram);
	SAFE_DELETE(m_pGatherRadianceWithSMProgram);
	SAFE_DELETE(m_pGatherAntiradianceProgram);
	SAFE_DELETE(m_pFinalGatherProgram);
	SAFE_DELETE(m_pNormalizeRadianceProgram);
	SAFE_DELETE(m_pPointCloudProgram);

	SAFE_DELETE(m_pGLPointSampler);
	SAFE_DELETE(m_pGLShadowMapSampler);

	SAFE_DELETE(m_pDepthBuffer);
	SAFE_DELETE(m_pLightBuffer);
}

bool Renderer::Init() 
{	
	V_RET_FOF(m_pUBTransform->Init(sizeof(TRANSFORM), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBMaterial->Init(sizeof(MATERIAL), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBLight->Init(sizeof(LIGHT), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBConfig->Init(sizeof(CONFIG), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBCamera->Init(sizeof(CAMERA), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBInfo->Init(sizeof(INFO), 0, GL_DYNAMIC_DRAW));

	V_RET_FOF(m_pTextureViewer->Init());
	V_RET_FOF(m_pPointCloud->Init());
	V_RET_FOF(m_pLightBuffer->Init());

	V_RET_FOF(m_pCreateGBufferProgram->Init());
	V_RET_FOF(m_pCreateSMProgram->Init());
	V_RET_FOF(m_pGatherRadianceProgram->Init());
	V_RET_FOF(m_pGatherRadianceWithSMProgram->Init());
	V_RET_FOF(m_pGatherAntiradianceProgram->Init());
	V_RET_FOF(m_pFinalGatherProgram->Init());
	V_RET_FOF(m_pNormalizeRadianceProgram->Init());
	V_RET_FOF(m_pPointCloudProgram->Init());
	
	m_pPointCloudProgram->BindUniformBuffer(m_pUBTransform, "transform");

	m_pCreateGBufferProgram->BindUniformBuffer(m_pUBTransform, "transform");
	m_pCreateGBufferProgram->BindUniformBuffer(m_pUBMaterial, "material");
		
	m_pGatherRadianceWithSMProgram->BindUniformBuffer(m_pUBLight, "light");
	m_pGatherRadianceWithSMProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherRadianceWithSMProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pGatherRadianceProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherRadianceProgram->BindUniformBuffer(m_pUBCamera, "camera");
	m_pGatherRadianceProgram->BindUniformBuffer(m_pUBInfo, "info");

	m_pGatherAntiradianceProgram->BindUniformBuffer(m_pUBInfo, "info");
	m_pGatherAntiradianceProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherAntiradianceProgram->BindUniformBuffer(m_pUBCamera, "camera");
	
	m_pFinalGatherProgram->BindUniformBuffer(m_pUBCamera, "camera");	

	m_pNormalizeRadianceProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pNormalizeRadianceProgram->BindUniformBuffer(m_pUBCamera, "camera");

	V_RET_FOF(m_pGLPointSampler->Init(GL_NEAREST, GL_NEAREST, GL_REPEAT, GL_REPEAT));
	V_RET_FOF(m_pGLShadowMapSampler->Init(GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER));
	
	m_pGatherRadianceProgram->BindSampler(0, m_pGLShadowMapSampler);
	m_pGatherRadianceProgram->BindSampler(1, m_pGLPointSampler);
	m_pGatherRadianceProgram->BindSampler(2, m_pGLPointSampler);
	m_pGatherRadianceProgram->BindSampler(3, m_pGLPointSampler);

	m_pGatherAntiradianceProgram->BindSampler(0, m_pGLPointSampler);
	m_pGatherAntiradianceProgram->BindSampler(1, m_pGLPointSampler);
	m_pGatherAntiradianceProgram->BindSampler(2, m_pGLPointSampler);
	m_pGatherAntiradianceProgram->BindSampler(3, m_pGLPointSampler);

	m_pFinalGatherProgram->BindSampler(0, m_pGLPointSampler);
	m_pFinalGatherProgram->BindSampler(1, m_pGLPointSampler);
	m_pFinalGatherProgram->BindSampler(2, m_pGLPointSampler);

	m_pNormalizeRadianceProgram->BindSampler(0, m_pGLPointSampler);

	V_RET_FOF(m_pFullScreenQuad->Init());

	V_RET_FOF(m_pShadowMap->Init(512));

	V_RET_FOF(m_pDepthBuffer->Init(camera->GetWidth(), camera->GetHeight(), GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, false));

	V_RET_FOF(m_pGBuffer->Init(camera->GetWidth(), camera->GetHeight(), m_pDepthBuffer));

	V_RET_FOF(m_pAccumulationRadiance->Init(camera->GetWidth(), camera->GetHeight(), 0));
	V_RET_FOF(m_pAccumulationAntiradiance->Init(camera->GetWidth(), camera->GetHeight(), 0));
	V_RET_FOF(m_pNormalizedRadiance->Init(camera->GetWidth(), camera->GetHeight(), 0));
	V_RET_FOF(m_pNormalizedAntiradiance->Init(camera->GetWidth(), camera->GetHeight(), 0));
	V_RET_FOF(m_pFinalResult->Init(camera->GetWidth(), camera->GetHeight(), m_pDepthBuffer));

	ClearAccumulationBuffer();

	postProcessing = new Postprocessing();
	postProcessing->Init();

	scene = new Scene(camera);
	scene->Init();
	scene->LoadCornellBox();
	
	scene->CreatePlaneHammersleySamples(m_NumAdditionalAVPLs);

	drawLight = false;
	drawTexture = false;
	m_PrintTimes = false;
	m_DrawOnlyDirectLight = false;
	m_DrawOnlyIndirectLight = false;
	
	m_UseAntiradiance = true;
	m_DrawAntiradiance = false;
	m_GeoTermLimit = 0.05f;
	m_CosBlurFactor = 0.1f;

	ConfigureLighting();

	return true;
}

void Renderer::Release()
{
	CheckGLError("CDSRenderer", "CDSRenderer::Release()");

	m_pDepthBuffer->Release();

	m_pGBuffer->Release();
	m_pAccumulationRadiance->Release();
	m_pAccumulationAntiradiance->Release();
	m_pNormalizedRadiance->Release();
	m_pNormalizedAntiradiance->Release();
	m_pFinalResult->Release();
	m_pTextureViewer->Release();
	m_pFullScreenQuad->Release();
	m_pPointCloud->Release();
	m_pLightBuffer->Release();

	postProcessing->Release();
	scene->Release();

	m_pShadowMap->Release();

	m_pCreateGBufferProgram->Release();
	m_pCreateSMProgram->Release();
	m_pGatherRadianceWithSMProgram->Release();
	m_pGatherRadianceProgram->Release();
	m_pGatherAntiradianceProgram->Release();
	m_pFinalGatherProgram->Release();
	m_pNormalizeRadianceProgram->Release();
	m_pPointCloudProgram->Release();

	m_pGLPointSampler->Release();
	m_pGLShadowMapSampler->Release();

	m_pUBTransform->Release();
	m_pUBMaterial->Release();
	m_pUBLight->Release();
	m_pUBConfig->Release();
	m_pUBCamera->Release();
	m_pUBInfo->Release();
}

void Renderer::Render() 
{	
	SetUpRender();
		
	CreateGBuffer();

	if(m_UseAntiradiance)
	{
		if(m_CurrentPath < m_NumPaths)
		{
			std::vector<Light*> lights;
			int remaining = m_NumPaths - m_CurrentPath;
			if(remaining >= m_NumPathsPerFrame)
			{
				lights = scene->CreatePaths(m_NumPathsPerFrame, m_CurrentPath, m_N, m_NumAdditionalAVPLs, m_UseHammersley);
				m_CurrentPath += m_NumPathsPerFrame;
			}
			else
			{
				 lights = scene->CreatePaths(remaining, m_CurrentPath, m_N, m_NumAdditionalAVPLs, m_UseHammersley);
				 m_CurrentPath += remaining;
			}
					
			GatherRadiance(lights);

			GatherAntiradiance(lights);
	
			if(m_CurrentPath >= m_NumPaths && !m_Finished)
			{
				std::cout << "Finished." << std::endl;
				m_Finished = true;
			}
		}
	}
	else
	{
		if(m_CurrentPath < m_NumPaths)
		{
			std::vector<Light*> path = scene->CreatePath(m_CurrentPath, m_N, m_NumAdditionalAVPLs, m_UseHammersley);

			GatherRadianceWithShadowMap(path);
	
			m_CurrentPath++;
		}
		else{
			if(!m_Finished)
				std::cout << "Finished." << std::endl;

			m_Finished = true;
		}
	}
		
	Normalize();
	
	GLenum buffers[1] = { GL_COLOR_ATTACHMENT0 };
	{
		CGLRenderTargetLock lock(m_pFinalResult->GetRenderTarget(), 1, buffers);
	
		glDisable(GL_DEPTH_TEST);
		glDepthMask(GL_FALSE);
		
		FinalGather();
		
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
		
		DrawAreaLight();
		
		DebugRender();		
	}
	
	m_pTextureViewer->DrawTexture(m_pFinalResult->GetTexture(), 0, 0, camera->GetWidth(), camera->GetHeight());
	
	m_Frame++;
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
	CGLRenderTargetLock lockRenderTarget(m_pGBuffer->GetRenderTarget(), 3, buffers);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	CGLBindLock lockProgram(m_pCreateGBufferProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	scene->DrawScene(m_pUBTransform, m_pUBMaterial);
}

void Renderer::GatherRadiance(std::vector<Light*> path)
{
	if(true)
	{
		GatherRadianceFromLights(path);
	}
	else
	{
		std::vector<Light*>::iterator it;
		for(it = path.begin(); it < path.end(); ++it)
		{
			GatherRadianceFromLight(*it);
		}
	}
}

void Renderer::GatherRadianceWithShadowMap(std::vector<Light*> path)
{
	std::vector<Light*>::iterator it;
	for(it = path.begin(); it < path.end(); ++it)
	{
		GatherRadianceFromLight(*it);
	}
}

void Renderer::Normalize()
{
	ConfigureLighting();
	
	GLenum buffers[1] = { GL_COLOR_ATTACHMENT0 };
	{
		CGLRenderTargetLock lock(m_pNormalizedRadiance->GetRenderTarget(), 1, buffers);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		CGLBindLock lockProgram(m_pNormalizeRadianceProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

		CGLBindLock lock0(m_pAccumulationRadiance->GetTexture(), CGL_TEXTURE0_SLOT);

		m_pFullScreenQuad->Draw();
	}

	{
		CGLRenderTargetLock lock(m_pNormalizedAntiradiance->GetRenderTarget(), 1, buffers);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		CGLBindLock lockProgram(m_pNormalizeRadianceProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

		CGLBindLock lock0(m_pAccumulationAntiradiance->GetTexture(), CGL_TEXTURE0_SLOT);

		m_pFullScreenQuad->Draw();
	}
}

void Renderer::GatherAntiradiance(std::vector<Light*> path)
{
	if(true)
	{
		GatherAntiradianceFromLights(path);
	}
	else
	{
		std::vector<Light*>::iterator it;
		for(it = path.begin(); it < path.end(); ++it)
		{
			GatherAntiradianceFromLight(*it);
		}
	}
}

void Renderer::GatherRadianceFromLight(Light* light)
{
	if(!m_UseAntiradiance) {
		FillShadowMap(light);
	}
	
	if(light->GetContrib().length() == 0.f)
		return;
	
	if(light->GetBounce() != m_RenderBounce && m_RenderBounce != -1)
		return;
	
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	
	GLenum buffers[1] = { GL_COLOR_ATTACHMENT0 };
	CGLRenderTargetLock lock(m_pAccumulationRadiance->GetRenderTarget(), 1, buffers);

	CGLBindLock lockProgram(m_pGatherRadianceWithSMProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	LIGHT light_info;
	light->Fill(light_info);
	m_pUBLight->UpdateData(&light_info);

	CGLBindLock lock0(m_pShadowMap->GetShadowMapTexture(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pGBuffer->GetPositionTextureWS(), CGL_TEXTURE1_SLOT);
	CGLBindLock lock2(m_pGBuffer->GetNormalTexture(), CGL_TEXTURE2_SLOT);
	CGLBindLock lock3(m_pGBuffer->GetMaterialTexture(), CGL_TEXTURE3_SLOT);

	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::GatherRadianceFromLights(std::vector<Light*> lights)
{
	std::vector<Light*> useLights;
	for(uint i = 0; i < lights.size(); ++i)
	{
		Light* light = lights[i];
		if(light->GetContrib().length() == 0.f)
			continue;

		if(light->GetBounce() != m_RenderBounce && m_RenderBounce != -1)
			continue;
	
		useLights.push_back(light);
	}

	LIGHT* lightBuffer = new LIGHT[useLights.size()];
	for(uint i = 0; i < useLights.size(); ++i)
	{
		LIGHT l;
		useLights[i]->Fill(l);
		lightBuffer[i] = l;
	}
	m_pLightBuffer->SetContent(sizeof(LIGHT), useLights.size(), lightBuffer, GL_STATIC_DRAW);
	delete [] lightBuffer;

	INFO i;
	i.numLights = useLights.size();
	m_pUBInfo->UpdateData(&i);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	
	GLenum buffers[1] = { GL_COLOR_ATTACHMENT0 };
	CGLRenderTargetLock lock(m_pAccumulationRadiance->GetRenderTarget(), 1, buffers);

	CGLBindLock lockProgram(m_pGatherRadianceProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	CGLBindLock lock0(m_pShadowMap->GetShadowMapTexture(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pGBuffer->GetPositionTextureWS(), CGL_TEXTURE1_SLOT);
	CGLBindLock lock2(m_pGBuffer->GetNormalTexture(), CGL_TEXTURE2_SLOT);
	CGLBindLock lock3(m_pGBuffer->GetMaterialTexture(), CGL_TEXTURE3_SLOT);
	CGLBindLock lock4(m_pLightBuffer, CGL_TEXTURE4_SLOT);

	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::GatherRadianceFromLightWithShadowMap(Light* light)
{
	FillShadowMap(light);
	
	if(light->GetContrib().length() == 0.f)
		return;
	
	if(light->GetBounce() != m_RenderBounce && m_RenderBounce != -1)
		return;
	
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	
	GLenum buffers[1] = { GL_COLOR_ATTACHMENT0 };
	CGLRenderTargetLock lock(m_pAccumulationRadiance->GetRenderTarget(), 1, buffers);

	CGLBindLock lockProgram(m_pGatherRadianceProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	LIGHT light_info;
	light->Fill(light_info);
	m_pUBLight->UpdateData(&light_info);

	CGLBindLock lock0(m_pShadowMap->GetShadowMapTexture(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pGBuffer->GetPositionTextureWS(), CGL_TEXTURE1_SLOT);
	CGLBindLock lock2(m_pGBuffer->GetNormalTexture(), CGL_TEXTURE2_SLOT);
	CGLBindLock lock3(m_pGBuffer->GetMaterialTexture(), CGL_TEXTURE3_SLOT);

	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::GatherAntiradianceFromLight(Light* light)
{
	if(glm::length(light->GetSrcContrib()) <= 0.01f)
		return;
	
	if(light->GetBounce() != m_RenderBounce + 1 && m_RenderBounce != -1)
		return;

	LIGHT l;
	light->Fill(l);	

	INFO i;
	i.numLights = 5;
	m_pUBInfo->UpdateData(&i);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	
	GLenum buffers[1] = { GL_COLOR_ATTACHMENT0 };
	CGLRenderTargetLock lock(m_pAccumulationAntiradiance->GetRenderTarget(), 1, buffers);

	CGLBindLock lockProgram(m_pGatherAntiradianceProgram->GetGLProgram(), CGL_PROGRAM_SLOT);
	
	CGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pGBuffer->GetNormalTexture(), CGL_TEXTURE1_SLOT);
	CGLBindLock lock2(m_pLightBuffer, CGL_TEXTURE2_SLOT);
	
	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::GatherAntiradianceFromLights(std::vector<Light*> lights)
{
	std::vector<Light*> useLights;
	for(uint i = 0; i < lights.size(); ++i)
	{
		Light* light = lights[i];
		if(glm::length(light->GetSrcContrib()) <= 0.01f)
			continue;

		if(light->GetBounce() != m_RenderBounce + 1 && m_RenderBounce != -1)
			continue;
	
		useLights.push_back(light);
	}

	LIGHT* lightBuffer = new LIGHT[useLights.size()];
	for(uint i = 0; i < useLights.size(); ++i)
	{
		LIGHT l;
		useLights[i]->Fill(l);
		lightBuffer[i] = l;
	}
	m_pLightBuffer->SetContent(sizeof(LIGHT), useLights.size(), lightBuffer, GL_STATIC_DRAW);
	delete [] lightBuffer;

	INFO i;
	i.numLights = useLights.size();
	m_pUBInfo->UpdateData(&i);
	
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	
	GLenum buffers[1] = { GL_COLOR_ATTACHMENT0 };
	CGLRenderTargetLock lock(m_pAccumulationAntiradiance->GetRenderTarget(), 1, buffers);

	CGLBindLock lockProgram(m_pGatherAntiradianceProgram->GetGLProgram(), CGL_PROGRAM_SLOT);
	
	CGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pGBuffer->GetNormalTexture(), CGL_TEXTURE1_SLOT);
	CGLBindLock lock2(m_pLightBuffer, CGL_TEXTURE2_SLOT);
	
	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::FillShadowMap(Light* light)
{
	glEnable(GL_DEPTH_TEST);

	// prevent surface acne
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.1f, 4.0f);
	
	CGLBindLock lockProgram(m_pCreateSMProgram->GetGLProgram(), CGL_PROGRAM_SLOT);
	GLenum buffer[1] = {GL_NONE};
	CGLRenderTargetLock lock(m_pShadowMap->GetRenderTarget(), 1, buffer);

	glViewport(0, 0, m_pShadowMap->GetShadowMapSize(), m_pShadowMap->GetShadowMapSize());
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	scene->DrawScene(light->GetViewMatrix(), light->GetProjectionMatrix(), m_pUBTransform);

	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	glDisable(GL_POLYGON_OFFSET_FILL);
}

void Renderer::FinalGather()
{
	CGLBindLock lockProgram(m_pFinalGatherProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	CGLBindLock lock0(m_pNormalizedRadiance->GetTexture(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pNormalizedAntiradiance->GetTexture(), CGL_TEXTURE1_SLOT);
	CGLBindLock lock2(m_pGBuffer->GetMaterialTexture(), CGL_TEXTURE2_SLOT);

	m_pFullScreenQuad->Draw();
}

void Renderer::DrawAreaLight()
{
	// avoid z-fighting
	glPolygonOffset(-1.0f, 1.0f);
	glDepthFunc(GL_LEQUAL);
	
	scene->DrawAreaLight(m_pUBTransform);
}

void Renderer::DebugRender()
{
	// draw lights for debuging
	if(GetDrawLight())
	{
		std::vector<POINT_CLOUD_POINT> pcp;
		for(int i = 0; i < m_CurrentPath; ++i)
		{
			std::vector<Light*> vPath = scene->GetPath(i);
			std::vector<Light*>::iterator it;

			for ( it=vPath.begin() ; it < vPath.end(); it++ )
			{
				Light* light = *it;
				if(m_RenderBounce == -1 || (light->GetBounce() == m_RenderBounce || light->GetBounce() == m_RenderBounce + 1))
				{
					POINT_CLOUD_POINT p;
					glm::vec3 pos = light->GetPosition() + 0.05f * light->GetOrientation();
					p.position = glm::vec4(pos, 1.0f);
					p.color = ColorForLight(light);
					pcp.push_back(p);
				}
			}
		}
		
		glm::vec4* positionData = new glm::vec4[pcp.size()];
		glm::vec4* colorData = new glm::vec4[pcp.size()];
		for(uint i = 0; i < pcp.size(); ++i)
		{
			positionData[i] = pcp[i].position;
			colorData[i] = pcp[i].color;
		}
		
		{
			CGLBindLock lockProgram(m_pPointCloudProgram->GetGLProgram(), CGL_PROGRAM_SLOT);
				
			TRANSFORM transform;
			transform.M = IdentityMatrix();
			transform.V = camera->GetViewMatrix();
			transform.itM = IdentityMatrix();
			transform.MVP = camera->GetProjectionMatrix() * camera->GetViewMatrix();
			m_pUBTransform->UpdateData(&transform);
			
			m_pPointCloud->Draw(positionData, colorData, pcp.size());		
		}
		delete [] positionData;
		delete [] colorData;
	}	
	
	// draw gbuffer info for debuging
	if(GetDrawTexture()) 
	{
		m_pTextureViewer->DrawTexture(m_pGBuffer->GetMaterialTexture(),  640, 360, 620, 340);
		m_pTextureViewer->DrawTexture(m_pNormalizedRadiance->GetTexture(),  10, 10, 620, 340);
		m_pTextureViewer->DrawTexture(m_pNormalizedAntiradiance->GetTexture(), 640, 10, 630, 340);
	}
}

void Renderer::WindowChanged()
{
	m_pDepthBuffer->Release();
	m_pDepthBuffer->Init(camera->GetWidth(), camera->GetHeight(), GL_DEPTH_COMPONENT32F, 
		GL_DEPTH, GL_FLOAT, 1, false);

	m_pGBuffer->Release();
	m_pGBuffer->Init(camera->GetWidth(), camera->GetHeight(), m_pDepthBuffer);
	
	m_pAccumulationRadiance->Release();
	m_pAccumulationAntiradiance->Release();
	m_pNormalizedRadiance->Release();
	m_pAccumulationRadiance->Init(camera->GetWidth(), camera->GetHeight(), 0);
	m_pAccumulationAntiradiance->Init(camera->GetWidth(), camera->GetHeight(), 0);
	m_pNormalizedRadiance->Init(camera->GetWidth(), camera->GetHeight(), 0);
	m_pFinalResult->Init(camera->GetWidth(), camera->GetHeight(), m_pDepthBuffer);

	ClearAccumulationBuffer();
}

void Renderer::ClearAccumulationBuffer()
{
	GLenum buffers[1] = { GL_COLOR_ATTACHMENT0 };
	glClearColor(0, 0, 0, 0);
	{
		CGLRenderTargetLock lock(m_pAccumulationRadiance->GetRenderTarget(), 1, buffers);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	}
	{
		CGLRenderTargetLock lock(m_pAccumulationAntiradiance->GetRenderTarget(), 1, buffers);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	}
	{
		CGLRenderTargetLock lock(m_pNormalizedRadiance->GetRenderTarget(), 1, buffers);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	}
	{
		CGLRenderTargetLock lock(m_pNormalizedAntiradiance->GetRenderTarget(), 1, buffers);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	}
	{
		CGLRenderTargetLock lock(m_pFinalResult->GetRenderTarget(), 1, buffers);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
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
}

void Renderer::Stats() 
{
	scene->Stats();
}

void Renderer::ConfigureLighting()
{
	CONFIG conf;
	conf.GeoTermLimit = m_GeoTermLimit;
	conf.BlurSigma = m_BlurSigma;
	conf.BlurK = CalcBlurNormalizationFactor(m_BlurSigma);
	conf.DrawAntiradiance =  m_DrawAntiradiance ? 1 : 0;
	conf.UseAntiradiance = m_UseAntiradiance ? 1 : 0;
	conf.nPaths = std::min(m_CurrentPath, m_NumPaths);
	conf.N = m_N;
	conf.Bias = m_Bias;
	m_pUBConfig->UpdateData(&conf);
}

void Renderer::PrintCameraConfig()
{
	std::cout << "Camera: " << std::endl;
	std::cout << "position: (" << scene->GetCamera()->GetPosition().x << ", " 
		<< scene->GetCamera()->GetPosition().y << ", " 
		<< scene->GetCamera()->GetPosition().z << ")" << std::endl;
}

float Renderer::CalcBlurNormalizationFactor(float sigma)
{
	int numSteps = 1000;
	float delta = PI / float(numSteps);
	float integralVal = 0.f;

	for(int i = 0; i < numSteps; ++i)
	{
		float sample = i * delta + delta / 2.f;
		float val = 1 / (sqrt(2*PI)*sigma) * exp(-(pow(sample,2)/(2*pow(sigma,2))));
		integralVal += val * delta;
	}

	return 1/(2*PI*integralVal);
}

void Renderer::Debug(std::vector<Light*> path)
{
	std::cout << "Path:" << std::endl;

	for(uint i = 0; i < path.size(); ++i)
	{
		Light* light = path[i];
		std::cout << "Light:" << i << std::endl;
		Debug(light);
	}
}

void Renderer::Debug(Light* light)
{
	std::cout << "Position: " << AsString(light->GetPosition()) << std::endl;
	std::cout << "Orientation: " << AsString(light->GetOrientation()) << std::endl;
	std::cout << "Contribution: " << AsString(light->GetContrib()) << std::endl;
	std::cout << "Src-Position: " << AsString(light->GetSrcPosition()) << std::endl;
	std::cout << "Src-Orientation: " << AsString(light->GetSrcOrientation()) << std::endl;
	std::cout << "Src-Contribution: " << AsString(light->GetSrcContrib()) << std::endl;
}

std::string Renderer::Debug(glm::vec3 v)
{
	std::stringstream ss("");
	ss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	return ss.str();
}

std::string Renderer::Debug(glm::vec4 v)
{
	std::stringstream ss("");
	ss << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
	return ss.str();
}

glm::vec4 Renderer::ColorForLight(Light* light)
{
	glm::vec4 color;
	switch(light->GetBounce()){
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

void Renderer::CreatePlaneHammersleySamples(int i)
{
	scene->CreatePlaneHammersleySamples(i);
}
