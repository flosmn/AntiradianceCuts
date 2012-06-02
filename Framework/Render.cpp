#include "Render.h"

#include <glm/gtc/type_ptr.hpp>

#include "Macros.h"
#include "Structs.h"

#include "CGBuffer.h"
#include "CTimer.h"
#include "CProgram.h"
#include "CPointCloud.h"
#include "CExport.h"

#include "AVPL.h"
#include "Scene.h"
#include "Camera.h"
#include "CShadowMap.h"
#include "CPostprocess.h"
#include "CRenderTarget.h"

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
#include <string>
#include <sstream>
#include <time.h>

std::vector<Light*> initialLights;

Renderer::Renderer(Camera* _camera) {
	camera = _camera;

	m_pShadowMap = new CShadowMap();
	scene = 0;

	m_PartialSum = true;

	m_Export = new CExport();
	m_Timer = new CTimer(1000);

	m_pDepthBuffer = new CGLTexture2D("Renderer.m_pDepthBuffer");
	
	m_pGBuffer = new CGBuffer();
		
	m_pLightDebugRenderTarget = new CRenderTarget();
	m_pPostProcessRenderTarget = new CRenderTarget();
	m_pPostProcess = new CPostprocess();
	
	m_pTextureViewer = new CTextureViewer();

	m_pUBTransform = new CGLUniformBuffer("Renderer.m_pUBTransform");
	m_pUBMaterial = new CGLUniformBuffer("Renderer.m_pUBMaterial");
	m_pUBLight = new CGLUniformBuffer("Renderer.m_pUBLight");
	m_pUBConfig = new CGLUniformBuffer("Renderer.m_pUBConfig");
	m_pUBCamera = new CGLUniformBuffer("Renderer.m_pUBCamera");
	m_pUBInfo = new CGLUniformBuffer("Renderer.m_pUBInfo");
	m_pUBAreaLight = new CGLUniformBuffer("Renderer.m_pUBAreaLight");

	m_pGLPointSampler = new CGLSampler("Renderer.m_pGLPointSampler");
	m_pGLShadowMapSampler = new CGLSampler("Renderer.m_pGLShadowMapSampler");

	m_pGatherRenderTarget = new CRenderTarget();
	m_pNormalizeRenderTarget = new CRenderTarget();
	m_pShadeRenderTarget = new CRenderTarget();

	m_pGatherProgram = new CProgram("Renderer.m_pGatherProgram", "Shaders\\Gather.vert", "Shaders\\Gather.frag");
	m_pNormalizeProgram = new CProgram("Renderer.m_pNormalizeProgram", "Shaders\\Gather.vert", "Shaders\\Normalize.frag");
	m_pShadeProgram = new CProgram("Renderer.m_pShadeProgram", "Shaders\\Gather.vert", "Shaders\\Shade.frag");

	m_pCreateGBufferProgram = new CProgram("Renderer.m_pCreateGBufferProgram", "Shaders\\CreateGBuffer.vert", "Shaders\\CreateGBuffer.frag");
	m_pCreateSMProgram = new CProgram("Renderer.m_pCreateSMProgram", "Shaders\\CreateSM.vert", "Shaders\\CreateSM.frag");
	m_pGatherRadianceWithSMProgram = new CProgram("Renderer.m_pGatherRadianceWithSMProgram", "Shaders\\Gather.vert", "Shaders\\GatherRadianceWithSM.frag");
	m_pPointCloudProgram = new CProgram("Renderer.m_pPointCloudProgram", "Shaders\\PointCloud.vert", "Shaders\\PointCloud.frag");
	m_pAreaLightProgram = new CProgram("Renderer.m_pAreaLightProgram", "Shaders\\DrawAreaLight.vert", "Shaders\\DrawAreaLight.frag");

	m_pFullScreenQuad = new CFullScreenQuad();

	m_pPointCloud = new CPointCloud();

	m_pLightBuffer = new CGLTextureBuffer("Renderer.m_pLightBuffer");

	m_RenderBounce = -1;
	m_Frame = 0;
	m_CurrentPath = 0;
	m_N = 50;
	m_NumAdditionalAVPLs = 128;
	m_UseHammersley = false;

	m_NumPaths = 1;
	m_NumPathsPerFrame = 1;

	m_Finished = false;
}

Renderer::~Renderer() {
	SAFE_DELETE(scene);
	SAFE_DELETE(m_Export);
	SAFE_DELETE(m_pShadowMap);
	SAFE_DELETE(m_pGBuffer);
	
	SAFE_DELETE(m_pGatherRenderTarget);
	SAFE_DELETE(m_pNormalizeRenderTarget);
	SAFE_DELETE(m_pShadeProgram);
		
	SAFE_DELETE(m_pGatherProgram);
	SAFE_DELETE(m_pNormalizeProgram);
	SAFE_DELETE(m_pShadeProgram);

	SAFE_DELETE(m_pLightDebugRenderTarget);
	SAFE_DELETE(m_pPostProcessRenderTarget);
	SAFE_DELETE(m_pPostProcess);
	
	SAFE_DELETE(m_pTextureViewer);
	SAFE_DELETE(m_pFullScreenQuad);
	SAFE_DELETE(m_pPointCloud);

	SAFE_DELETE(m_pUBTransform);
	SAFE_DELETE(m_pUBMaterial);
	SAFE_DELETE(m_pUBLight);
	SAFE_DELETE(m_pUBConfig);
	SAFE_DELETE(m_pUBCamera);
	SAFE_DELETE(m_pUBInfo);
	SAFE_DELETE(m_pUBAreaLight);

	SAFE_DELETE(m_pCreateGBufferProgram);
	SAFE_DELETE(m_pCreateSMProgram);
	SAFE_DELETE(m_pGatherRadianceWithSMProgram);
	SAFE_DELETE(m_pPointCloudProgram);
	SAFE_DELETE(m_pAreaLightProgram);

	SAFE_DELETE(m_pGLPointSampler);
	SAFE_DELETE(m_pGLShadowMapSampler);

	SAFE_DELETE(m_pDepthBuffer);
	SAFE_DELETE(m_pLightBuffer);

	SAFE_DELETE(m_Timer);
}

bool Renderer::Init() 
{	
	V_RET_FOF(m_pDepthBuffer->Init(camera->GetWidth(), camera->GetHeight(), GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, false));

	V_RET_FOF(m_pUBTransform->Init(sizeof(TRANSFORM), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBMaterial->Init(sizeof(MATERIAL), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBLight->Init(sizeof(AVPL), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBConfig->Init(sizeof(CONFIG), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBCamera->Init(sizeof(CAMERA), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBInfo->Init(sizeof(INFO), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBAreaLight->Init(sizeof(AREA_LIGHT), 0, GL_DYNAMIC_DRAW));

	V_RET_FOF(m_pTextureViewer->Init());
	V_RET_FOF(m_pPointCloud->Init());
	V_RET_FOF(m_pLightBuffer->Init());
	
	V_RET_FOF(m_pCreateGBufferProgram->Init());
	V_RET_FOF(m_pCreateSMProgram->Init());
	V_RET_FOF(m_pGatherRadianceWithSMProgram->Init());
	V_RET_FOF(m_pPointCloudProgram->Init());

	V_RET_FOF(m_pGatherProgram->Init());
	V_RET_FOF(m_pNormalizeProgram->Init());
	V_RET_FOF(m_pShadeProgram->Init());
	V_RET_FOF(m_pAreaLightProgram->Init());

	V_RET_FOF(m_pGatherRenderTarget->Init(camera->GetWidth(), camera->GetHeight(), 3, 0));
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

	V_RET_FOF(m_pGLPointSampler->Init(GL_NEAREST, GL_NEAREST, GL_REPEAT, GL_REPEAT));
	V_RET_FOF(m_pGLShadowMapSampler->Init(GL_NEAREST, GL_NEAREST, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_BORDER));
	
	m_pGatherRadianceWithSMProgram->BindSampler(0, m_pGLShadowMapSampler);
	m_pGatherRadianceWithSMProgram->BindSampler(1, m_pGLPointSampler);
	m_pGatherRadianceWithSMProgram->BindSampler(2, m_pGLPointSampler);

	m_pGatherProgram->BindSampler(0, m_pGLPointSampler);
	m_pGatherProgram->BindSampler(1, m_pGLPointSampler);
	m_pGatherProgram->BindSampler(2, m_pGLPointSampler);

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

	V_RET_FOF(m_pFullScreenQuad->Init());

	V_RET_FOF(m_pShadowMap->Init(512));
	
	V_RET_FOF(m_pGBuffer->Init(camera->GetWidth(), camera->GetHeight(), m_pDepthBuffer));

	ClearAccumulationBuffer();

	scene = new Scene(camera);
	scene->Init();
	scene->LoadCornellBox();
	
	scene->CreatePlaneHammersleySamples(m_NumAdditionalAVPLs);

	drawLight = false;
	drawTexture = false;
	
	m_UseAntiradiance = true;
	m_GeoTermLimit = 0.05f;
	m_ToneMapping = true;

	ConfigureLighting();

	time(&m_StartTime);

	m_Timer->Init();
	m_Timer->RegisterEvent("Create Paths", CTimer::CPU);

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
	m_pLightBuffer->Release();

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

	m_pGatherRenderTarget->Release();
	m_pNormalizeRenderTarget->Release();
	m_pShadeRenderTarget->Release();

	m_pNormalizeProgram->Release();
	m_pGatherProgram->Release();
	m_pShadeProgram->Release();
	
	m_pGLPointSampler->Release();
	m_pGLShadowMapSampler->Release();

	m_pUBTransform->Release();
	m_pUBMaterial->Release();
	m_pUBLight->Release();
	m_pUBConfig->Release();
	m_pUBCamera->Release();
	m_pUBInfo->Release();
	m_pUBAreaLight->Release();
}

void Renderer::Render() 
{	
	SetUpRender();
		
	CreateGBuffer();
	
	if(m_UseAntiradiance)
	{
		if(m_CurrentPath < m_NumPaths)
		{
			m_Timer->StartEvent("Create Paths");
			std::vector<AVPL*> avpls;
			int remaining = m_NumPaths - m_CurrentPath;
			if(remaining >= m_NumPathsPerFrame)
			{
				avpls = scene->CreatePaths(m_NumPathsPerFrame, m_N, m_NumAdditionalAVPLs, m_UseHammersley);
				m_CurrentPath += m_NumPathsPerFrame;
			}
			else
			{
				 avpls = scene->CreatePaths(remaining, m_N, m_NumAdditionalAVPLs, m_UseHammersley);
				 m_CurrentPath += remaining;
			}
			m_Timer->StopEvent("Create Paths");

			Gather(avpls);

			for(uint i = 0; i < avpls.size(); ++i)
			{
				delete avpls[i];
			}

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
			std::vector<AVPL*> path = scene->CreatePath(m_N, m_NumAdditionalAVPLs, m_UseHammersley);

			GatherRadianceWithShadowMap(path);
	
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
	
	if(m_ToneMapping)
	{
		m_pPostProcess->Postprocess(m_pShadeRenderTarget->GetBuffer(0), m_pPostProcessRenderTarget);
		m_pTextureViewer->DrawTexture(m_pPostProcessRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	else
	{
		m_pTextureViewer->DrawTexture(m_pShadeRenderTarget->GetBuffer(0), 0, 0, camera->GetWidth(), camera->GetHeight());
	}
	
	m_Frame++;
	
	if(m_Frame % 10 == 0)
	{
		m_Timer->PrintStats();
		m_Timer->Reset();
	}

	if(m_Frame % 1000 == 0)
	{
		ExportPartialResult();
		std::cout << "#Paths: " << m_CurrentPath << std::endl;
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
	CGLRenderTargetLock lockRenderTarget(m_pGBuffer->GetRenderTarget(), 3, buffers);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	CGLBindLock lockProgram(m_pCreateGBufferProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

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

		CGLBindLock lockProgram(m_pNormalizeProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

		CGLBindLock lock0(m_pGatherRenderTarget->GetBuffer(0), CGL_TEXTURE0_SLOT);
		CGLBindLock lock1(m_pGatherRenderTarget->GetBuffer(1), CGL_TEXTURE1_SLOT);
		CGLBindLock lock2(m_pGatherRenderTarget->GetBuffer(2), CGL_TEXTURE2_SLOT);

		m_pFullScreenQuad->Draw();
	}
}

void Renderer::GatherRadianceFromLightWithShadowMap(AVPL* avpl)
{
	FillShadowMap(avpl);
	
	if(avpl->GetIntensity().length() == 0.f)
		return;
	
	if(avpl->GetBounce() != m_RenderBounce && m_RenderBounce != -1)
		return;
	
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	
	CRenderTargetLock lock(m_pGatherRenderTarget);

	CGLBindLock lockProgram(m_pGatherRadianceWithSMProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	AVPL_STRUCT light_info;
	avpl->Fill(light_info);
	m_pUBLight->UpdateData(&light_info);

	CGLBindLock lock0(m_pShadowMap->GetShadowMapTexture(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pGBuffer->GetPositionTextureWS(), CGL_TEXTURE1_SLOT);
	CGLBindLock lock2(m_pGBuffer->GetNormalTexture(), CGL_TEXTURE2_SLOT);
	
	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::Gather(std::vector<AVPL*> avpls)
{
	std::vector<AVPL*> useAVPLs;
	if(m_RenderBounce == -1)
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
			if(avpl->GetBounce() != m_RenderBounce + 1) avpl->SetAntiintensity(glm::vec3(0.f));
			if(avpl->GetBounce() != m_RenderBounce) avpl->SetIntensity(glm::vec3(0.f));
			useAVPLs.push_back(avpl);
		}
	}

	AVPL_BUFFER* avplBuffer = new AVPL_BUFFER[useAVPLs.size() + 1];
	memset(avplBuffer, 0, sizeof(AVPL_BUFFER) * useAVPLs.size() + 1);
	for(uint i = 0; i < useAVPLs.size(); ++i)
	{
		AVPL_BUFFER buffer;
		useAVPLs[i]->Fill(buffer);
		avplBuffer[i] = buffer;
	}
	m_pLightBuffer->SetContent(sizeof(AVPL_BUFFER), useAVPLs.size() + 1, avplBuffer, GL_STATIC_DRAW);
	delete [] avplBuffer;

	INFO i;
	i.numLights = useAVPLs.size();
	m_pUBInfo->UpdateData(&i);

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	
	CRenderTargetLock lockRenderTarget(m_pGatherRenderTarget);

	CGLBindLock lockProgram(m_pGatherProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	CGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pGBuffer->GetNormalTexture(), CGL_TEXTURE1_SLOT);
	CGLBindLock lock2(m_pLightBuffer, CGL_TEXTURE2_SLOT);

	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::FillShadowMap(AVPL* avpl)
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

		CGLBindLock lockProgram(m_pShadeProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

		CGLBindLock lock0(m_pNormalizeRenderTarget->GetBuffer(0), CGL_TEXTURE0_SLOT);
		CGLBindLock lock2(m_pGBuffer->GetMaterialTexture(), CGL_TEXTURE1_SLOT);

		m_pFullScreenQuad->Draw();
		
		glEnable(GL_DEPTH_TEST);
		glDepthMask(GL_TRUE);
	}

	DrawAreaLight();
		
	DebugRender();
}

void Renderer::DrawAreaLight()
{
	CGLBindLock lock(m_pAreaLightProgram->GetGLProgram(), CGL_PROGRAM_SLOT);
	
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
		if(m_RenderBounce == -1 || (avpl->GetBounce() == m_RenderBounce || avpl->GetBounce() == m_RenderBounce + 1))
		{
			POINT_CLOUD_POINT p;
			glm::vec3 pos = avpl->GetPosition() + 0.05f * avpl->GetOrientation();
			p.position = glm::vec4(pos, 1.0f);
			p.color = ColorForLight(avpl);
			pcp.push_back(p);
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
		CRenderTargetLock lock(m_pLightDebugRenderTarget);
		
		CGLBindLock lockProgram(m_pPointCloudProgram->GetGLProgram(), CGL_PROGRAM_SLOT);
		
		TRANSFORM transform;
		transform.M = IdentityMatrix();
		transform.V = camera->GetViewMatrix();
		transform.itM = IdentityMatrix();
		transform.MVP = camera->GetProjectionMatrix() * camera->GetViewMatrix();
		m_pUBTransform->UpdateData(&transform);
	
		glDepthMask(GL_FALSE);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);	
		glBlendFunc(GL_ONE, GL_ONE);
		m_pPointCloud->Draw(positionData, colorData, pcp.size());		
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);
	}
	
	pcp.clear();
	delete [] positionData;
	delete [] colorData;
}

void Renderer::DebugRender()
{
	// draw gbuffer info for debuging
	if(GetDrawTexture()) 
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
}

void Renderer::Stats() 
{
}

void Renderer::ConfigureLighting()
{
	CONFIG conf;
	conf.GeoTermLimit = m_GeoTermLimit;
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

void Renderer::CreatePlaneHammersleySamples(int i)
{
	scene->CreatePlaneHammersleySamples(i);
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

void Renderer::SetGamma(float gamma)
{
	m_pPostProcess->SetGamma(gamma);
}

void Renderer::SetExposure(float exposure)
{
	m_pPostProcess->SetExposure(exposure);
}
