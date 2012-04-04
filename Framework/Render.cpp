#include "Render.h"

#include <glm/gtc/type_ptr.hpp>

#include "Macros.h"
#include "Structs.h"

#include "CAccumulationBuffer.h"
#include "CGBuffer.h"
#include "CTimer.h"
#include "CProgram.h"

#include "Scene.h"
#include "Camera.h"
#include "Light.h"
#include "CShadowMap.h"
#include "Postprocessing.h"

#include "CUtils\Util.h"
#include "CUtils\GLErrorUtil.h"
#include "CUtils\ShaderUtil.h"
#include "CUtils\CLightViewer.h"
#include "CUtils\CTextureViewer.h"

#include "CMeshResources\CFullScreenQuad.h"

#include "CGLResources\CGLUniformBuffer.h"
#include "CGLResources\CGLTexture2D.h"
#include "CGLResources\CGLFrameBuffer.h"
#include "CGLResources\CGLProgram.h"
#include "CGLResources\CGLSampler.h"

#include <memory>

std::vector<Light*> initialLights;

Renderer::Renderer(Camera* _camera) {
	camera = _camera;

	m_pShadowMap = new CShadowMap();
	scene = 0;
	postProcessing = 0;
	m_pGBuffer = new CGBuffer();
	m_pAccumulationRadiance = new CAccumulationBuffer();
	m_pAccumulationAntiradiance = new CAccumulationBuffer();
	m_pNormalizedRadiance = new CAccumulationBuffer();
	m_pNormalizedAntiradiance = new CAccumulationBuffer();

	m_pLightViewer = new CLightViewer();
	m_pTextureViewer = new CTextureViewer();

	m_pUBTransform = new CGLUniformBuffer("Renderer.m_pUBTransform");
	m_pUBMaterial = new CGLUniformBuffer("Renderer.m_pUBMaterial");
	m_pUBLight = new CGLUniformBuffer("Renderer.m_pUBLight");
	m_pUBConfig = new CGLUniformBuffer("Renderer.m_pUBConfig");
	m_pUBCamera = new CGLUniformBuffer("Renderer.m_pUBCamera");

	m_pGLPointSampler = new CGLSampler("Renderer.m_pGLPointSampler");

	m_pCreateGBufferProgram = new CProgram("Renderer.m_pCreateGBufferProgram", 
		"Shaders\\CreateGBuffer.vert", "Shaders\\CreateGBuffer.frag");

	m_pCreateSMProgram = new CProgram("Renderer.m_pCreateSMProgram", 
		"Shaders\\CreateSM.vert", "Shaders\\CreateSM.frag");

	m_pGatherRadianceProgram = new CProgram("Renderer.m_pGatherRadianceProgram", "Shaders\\Gather.vert", "Shaders\\GatherRadiance.frag");
	m_pGatherAntiradianceProgram = new CProgram("Renderer.m_pGatherAntiradianceProgram", "Shaders\\Gather.vert", "Shaders\\GatherAntiradiance.frag");
	m_pFinalGatherProgram = new CProgram("Renderer.m_pFinalGatherProgram", "Shaders\\Gather.vert", "Shaders\\FinalGather.frag");
	m_pNormalizeRadianceProgram = new CProgram("Renderer.m_pNormalizeRadianceProgram", "Shaders\\Gather.vert", "Shaders\\Normalize.frag");

	m_pFullScreenQuad = new CFullScreenQuad();

	m_BlurSigma = .1f;
	
	m_Frame = 0;
	m_CurrentPath = 0;

	m_NumPaths = 1;

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
	SAFE_DELETE(m_pLightViewer);
	SAFE_DELETE(m_pTextureViewer);
	SAFE_DELETE(m_pFullScreenQuad);

	SAFE_DELETE(m_pUBTransform);
	SAFE_DELETE(m_pUBMaterial);
	SAFE_DELETE(m_pUBLight);
	SAFE_DELETE(m_pUBConfig);
	SAFE_DELETE(m_pUBCamera);

	SAFE_DELETE(m_pCreateGBufferProgram);
	SAFE_DELETE(m_pCreateSMProgram);
	SAFE_DELETE(m_pGatherRadianceProgram);
	SAFE_DELETE(m_pGatherAntiradianceProgram);
	SAFE_DELETE(m_pFinalGatherProgram);
	SAFE_DELETE(m_pNormalizeRadianceProgram);

	SAFE_DELETE(m_pGLPointSampler);
}

bool Renderer::Init() 
{	
	V_RET_FOF(m_pUBTransform->Init(sizeof(TRANSFORM), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBMaterial->Init(sizeof(MATERIAL), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBLight->Init(sizeof(LIGHT), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBConfig->Init(sizeof(CONFIG), 0, GL_DYNAMIC_DRAW));
	V_RET_FOF(m_pUBCamera->Init(sizeof(CAMERA), 0, GL_DYNAMIC_DRAW));

	V_RET_FOF(m_pLightViewer->Init());
	m_pLightViewer->BindUniformBuffer(m_pUBTransform, "transform");

	V_RET_FOF(m_pTextureViewer->Init());

	V_RET_FOF(m_pCreateGBufferProgram->Init());
	V_RET_FOF(m_pCreateSMProgram->Init());
	V_RET_FOF(m_pGatherRadianceProgram->Init());
	V_RET_FOF(m_pGatherAntiradianceProgram->Init());
	V_RET_FOF(m_pFinalGatherProgram->Init());
	V_RET_FOF(m_pNormalizeRadianceProgram->Init());

	m_pCreateGBufferProgram->BindUniformBuffer(m_pUBTransform, "transform");
	m_pCreateGBufferProgram->BindUniformBuffer(m_pUBMaterial, "material");

	m_pGatherRadianceProgram->BindUniformBuffer(m_pUBLight, "light");
	m_pGatherRadianceProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherRadianceProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pGatherAntiradianceProgram->BindUniformBuffer(m_pUBLight, "light");
	m_pGatherAntiradianceProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pGatherAntiradianceProgram->BindUniformBuffer(m_pUBCamera, "camera");

	m_pFinalGatherProgram->BindUniformBuffer(m_pUBCamera, "camera");
	m_pFinalGatherProgram->BindUniformBuffer(m_pUBConfig, "config");

	m_pNormalizeRadianceProgram->BindUniformBuffer(m_pUBConfig, "config");
	m_pNormalizeRadianceProgram->BindUniformBuffer(m_pUBCamera, "camera");

	V_RET_FOF(m_pGLPointSampler->Init(GL_NEAREST, GL_NEAREST));

	m_pGatherRadianceProgram->BindSampler(0, m_pGLPointSampler);
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

	V_RET_FOF(m_pGBuffer->Init(camera->GetWidth(), camera->GetHeight()));

	V_RET_FOF(m_pAccumulationRadiance->Init(camera->GetWidth(), camera->GetHeight()));
	V_RET_FOF(m_pAccumulationAntiradiance->Init(camera->GetWidth(), camera->GetHeight()));
	V_RET_FOF(m_pNormalizedRadiance->Init(camera->GetWidth(), camera->GetHeight()));
	V_RET_FOF(m_pNormalizedAntiradiance->Init(camera->GetWidth(), camera->GetHeight()));

	ClearAccumulationBuffer();

	postProcessing = new Postprocessing();
	postProcessing->Init();

	m_Timer = new CTimer(20);
	m_Timer->Init();
	m_Timer->RegisterEvent("Frame GPU", CTimer::GPU);
	m_Timer->RegisterEvent("Set State", CTimer::GPU);
	m_Timer->RegisterEvent("Fill GBuffer", CTimer::GPU);
	m_Timer->RegisterEvent("Create Path", CTimer::GPU);
	m_Timer->RegisterEvent("Render Path", CTimer::GPU);
	m_Timer->RegisterEvent("Fill Shadow Map", CTimer::GPU);
	m_Timer->RegisterEvent("Render VPL", CTimer::GPU);
	m_Timer->RegisterEvent("Post-Process", CTimer::GPU);
	m_Timer->RegisterEvent("Draw Area Light", CTimer::GPU);

	scene = new Scene(camera);
	scene->Init();
	scene->LoadSimpleScene();
			
	drawLight = false;
	drawTexture = false;
	m_PrintTimes = false;
	m_DrawOnlyDirectLight = false;
	m_DrawOnlyIndirectLight = false;
	
	m_UseAntiradiance = true;
	m_DrawAntiradiance = false;
	m_GeoTermLimit = 0.1f;
	m_CosBlurFactor = 20.f;

	ConfigureLighting();
	
	return true;
}

void Renderer::Release()
{
	CheckGLError("CDSRenderer", "CDSRenderer::Release()");
	m_pGBuffer->Release();
	m_pAccumulationRadiance->Release();
	m_pAccumulationAntiradiance->Release();
	m_pNormalizedRadiance->Release();
	m_pNormalizedAntiradiance->Release();
	m_pLightViewer->Release();
	m_pTextureViewer->Release();
	m_pFullScreenQuad->Release();

	postProcessing->Release();
	scene->Release();

	m_pShadowMap->Release();

	m_pCreateGBufferProgram->Release();
	m_pCreateSMProgram->Release();
	m_pGatherRadianceProgram->Release();
	m_pGatherAntiradianceProgram->Release();
	m_pFinalGatherProgram->Release();
	m_pNormalizeRadianceProgram->Release();

	m_pGLPointSampler->Release();

	m_pUBTransform->Release();
	m_pUBMaterial->Release();
	m_pUBLight->Release();
	m_pUBConfig->Release();
	m_pUBCamera->Release();
}

void Renderer::Render() 
{
	m_Timer->StartEvent("Frame GPU");
		
	SetUpRender();
	
	CreateGBuffer();
	
	if(m_CurrentPath < m_NumPaths)
	{
		std::vector<Light*> path = scene->CreatePath();

		GatherRadiance();
	
		GatherAntiradiance();

		m_CurrentPath++;
	}
	else{
		if(!m_Finished)
			std::cout << "Finished." << std::endl;

		m_Finished = true;
	}

	Normalize();
	
	FinalGather();
		
	DrawAreaLight();

	DebugRender();
	
	m_Timer->StopEvent("Frame GPU");
	
	if(m_Frame > 0 && m_Frame % 49 == 0 && m_Frame < 200) {
		m_Timer->PrintStats();
		m_Timer->Reset();
	}
	m_Frame++;
}

void Renderer::SetUpRender()
{
	m_Timer->StartEvent("Set State");

	glViewport(0, 0, (GLsizei)scene->GetCamera()->GetWidth(), (GLsizei)scene->GetCamera()->GetHeight());

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CW);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_LEQUAL);
	glDepthRange(0.0f, 1.0f);
	
	m_Timer->StopEvent("Set State");

	CAMERA camera;
	camera.positionWS = scene->GetCamera()->GetPosition();
	camera.width = (int)scene->GetCamera()->GetWidth();
	camera.height = (int)scene->GetCamera()->GetHeight();
	m_pUBCamera->UpdateData(&camera);
}

void Renderer::CreateGBuffer()
{
	m_Timer->StartEvent("Fill GBuffer");

	GLenum buffers [3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2};
	CGLRenderTargetLock lockRenderTarget(m_pGBuffer->GetRenderTarget(), 3, buffers);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	CGLBindLock lockProgram(m_pCreateGBufferProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	scene->DrawScene(m_pUBTransform, m_pUBMaterial);

	m_Timer->StopEvent("Fill GBuffer");
}

void Renderer::GatherRadiance()
{
	std::vector<Light*> path = scene->GetCurrentPath();
	std::vector<Light*>::iterator it;
	for(it = path.begin(); it < path.end(); ++it)
	{
		Light* light = *it;
		GatherRadianceFromLight(light);
	}
}

void Renderer::Normalize()
{
	CONFIG conf;
	conf.uGeoTermLimit = m_GeoTermLimit;
	conf.BlurSigma = m_BlurSigma;
	conf.BlurK = CalcBlurNormalizationFactor(m_BlurSigma);
	conf.uDrawAntiradiance =  m_DrawAntiradiance ? 1 : 0;
	conf.uUseAntiradiance = m_UseAntiradiance ? 1 : 0;
	conf.nPaths = std::min(m_CurrentPath, m_NumPaths);
	m_pUBConfig->UpdateData(&conf);
	
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

void Renderer::GatherAntiradiance()
{
	if(!m_UseAntiradiance){
		return;
	}

	std::vector<Light*> path = scene->GetCurrentPath();
	std::vector<Light*>::iterator it;
	for(it = path.begin(); it < path.end(); ++it)
	{
		Light* light = *it;
		GatherAntiradianceFromLight(light);
	}
}

void Renderer::GatherRadianceFromLight(Light* light)
{
	if(!m_UseAntiradiance) {
		FillShadowMap(light);
	}
	
	if(light->GetFlux().r == 0.f && light->GetFlux().g == 0.f && light->GetFlux().b == 0.f)
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
	if(light->GetSrcFlux().r == 0.f && light->GetSrcFlux().g == 0.f && light->GetSrcFlux().b == 0.f)
		return;

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);	
	glBlendFunc(GL_ONE, GL_ONE);
	glViewport(0, 0, camera->GetWidth(), camera->GetHeight());
	
	GLenum buffers[1] = { GL_COLOR_ATTACHMENT0 };
	CGLRenderTargetLock lock(m_pAccumulationAntiradiance->GetRenderTarget(), 1, buffers);

	CGLBindLock lockProgram(m_pGatherAntiradianceProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	LIGHT light_info;
	light->Fill(light_info);
	m_pUBLight->UpdateData(&light_info);

	CGLBindLock lock0(m_pGBuffer->GetPositionTextureWS(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pGBuffer->GetNormalTexture(), CGL_TEXTURE1_SLOT);
	
	m_pFullScreenQuad->Draw();
	
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void Renderer::FillShadowMap(Light* light)
{
	m_Timer->StartEvent("Fill Shadow Map"); 
	{
		glEnable(GL_DEPTH_TEST);

		// prevent surface acne
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1.1f, 4.0f);
		glViewport(0, 0, m_pShadowMap->GetShadowMapSize(), m_pShadowMap->GetShadowMapSize());

		CGLBindLock lockProgram(m_pCreateSMProgram->GetGLProgram(), CGL_PROGRAM_SLOT);
		GLenum buffer[1] = {GL_NONE};
		CGLRenderTargetLock lock(m_pShadowMap->GetRenderTarget(), 1, buffer);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		scene->DrawScene(light->GetViewMatrix(), light->GetProjectionMatrix(), m_pUBTransform);
	}
	m_Timer->StopEvent("Fill Shadow Map");
}

void Renderer::FinalGather()
{
	CGLBindLock lockProgram(m_pFinalGatherProgram->GetGLProgram(), CGL_PROGRAM_SLOT);

	CGLBindLock lock0(m_pNormalizedRadiance->GetTexture(), CGL_TEXTURE0_SLOT);
	CGLBindLock lock1(m_pNormalizedAntiradiance->GetTexture(), CGL_TEXTURE1_SLOT);
	CGLBindLock lock2(m_pGBuffer->GetMaterialTexture(), CGL_TEXTURE2_SLOT);

	m_pFullScreenQuad->Draw();
}

void Renderer::PostProcess()
{
	// do post processing
	m_Timer->StartEvent("Post-Process");
	postProcessing->Postprocess(
		m_pAccumulationRadiance->GetTexture()->GetResourceIdentifier(), 
		camera->GetWidth(), camera->GetHeight(), m_CurrentPath);
	m_Timer->StopEvent("Post-Process");
}

void Renderer::DrawAreaLight()
{
	GLenum err;
	
	m_Timer->StartEvent("Draw Area Light");
	
	// draw the area lights of the scene
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// regain depth information
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	scene->DrawScene(camera->GetViewMatrix(), camera->GetProjectionMatrix(), m_pUBTransform);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	
	err = glGetError();
	if(err != GL_NO_ERROR)
	{
		std::cout << "OpenGL error: " << gluErrorString(err) << std::endl;
	}

	// avoid z-fighting
	glPolygonOffset(-1.0f, 1.0f);
	glDepthFunc(GL_LEQUAL);

	scene->DrawAreaLight(m_pUBTransform);
	
	err = glGetError();
	if(err != GL_NO_ERROR)
	{
		std::cout << "OpenGL error: " << gluErrorString(err) << std::endl;
	}

	m_Timer->StopEvent("Draw Area Light");
}

void Renderer::DebugRender()
{
	GLenum err;

	// draw lights for debuging
	if(GetDrawLight())
	{
		for(int i = 0; i < m_CurrentPath; ++i)
		{
			std::vector<Light*> vPath = scene->GetPath(i);
			std::vector<Light*>::iterator it;

			for ( it=vPath.begin() ; it < vPath.end(); it++ )
			{
				m_pLightViewer->DrawLight(*it, camera, m_pUBTransform);
			}
		}
	}	
	
	// draw gbuffer info for debuging
	if(GetDrawTexture()) 
	{
		m_pTextureViewer->DrawTexture(m_pShadowMap->GetShadowMapTexture(), 10, 360, 620, 340);
		m_pTextureViewer->DrawTexture(m_pGBuffer->GetMaterialTexture(),  640, 360, 620, 340);
		m_pTextureViewer->DrawTexture(m_pNormalizedRadiance->GetTexture(),  10, 10, 620, 340);
		m_pTextureViewer->DrawTexture(m_pNormalizedAntiradiance->GetTexture(), 640, 10, 630, 340);
	}
	
	err = glGetError();
	if(err != GL_NO_ERROR)
	{
		std::cout << "OpenGL error: " << gluErrorString(err) << std::endl;
	}
}

void Renderer::WindowChanged()
{
	m_pGBuffer->Release();
	m_pGBuffer->Init(camera->GetWidth(), camera->GetHeight());
	
	m_pAccumulationRadiance->Release();
	m_pAccumulationAntiradiance->Release();
	m_pNormalizedRadiance->Release();
	m_pAccumulationRadiance->Init(camera->GetWidth(), camera->GetHeight());
	m_pAccumulationAntiradiance->Init(camera->GetWidth(), camera->GetHeight());
	m_pNormalizedRadiance->Init(camera->GetWidth(), camera->GetHeight());

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
	conf.uGeoTermLimit = m_GeoTermLimit;
	conf.BlurSigma = m_BlurSigma;
	conf.BlurK = CalcBlurNormalizationFactor(m_BlurSigma);
	conf.uDrawAntiradiance =  m_DrawAntiradiance ? 1 : 0;
	conf.uUseAntiradiance = m_UseAntiradiance ? 1 : 0;
	conf.nPaths = m_NumPaths;
	m_pUBConfig->UpdateData(&conf);

	//std::cout << "Blur Factor: " << m_BlurSigma << std::endl;
	//std::cout << "Blur Norm: " << conf.BlurK << std::endl;
	//std::cout << "Use Antiradiance: " << conf.uUseAntiradiance << std::endl;
	//std::cout << "Draw Antiradiance: " << conf.uDrawAntiradiance << std::endl;
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
