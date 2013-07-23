#include "CConfigManager.h"

#include "Render.h"

#include "Defines.h"

#include <iostream>

CConfigManager::CConfigManager()
{
	m_pConfVars = new CONF_VARS[1];
	m_pConfVarsGUI = new CONF_VARS[1];

	m_pConfVars->gatherWithCuda = m_pConfVarsGUI->gatherWithCuda = 1;
	m_pConfVars->useLightCuts = m_pConfVarsGUI->useLightCuts = 0;
	m_pConfVars->useClusteredDeferred = m_pConfVarsGUI->useClusteredDeferred = 1;
	m_pConfVars->bvhLevel = m_pConfVarsGUI->bvhLevel = 0;
	m_pConfVars->DrawClusterAABBs = m_pConfVarsGUI->DrawClusterAABBs = 0;
	m_pConfVars->DrawClusterLights = m_pConfVarsGUI->DrawClusterLights = 0;
	m_pConfVars->DrawLights = m_pConfVarsGUI->DrawLights = 0;
	m_pConfVars->considerNormals = m_pConfVarsGUI->considerNormals = 1;
	m_pConfVars->gatherClusterLightsSimple = m_pConfVarsGUI->gatherClusterLightsSimple = 0;
	
	m_pConfVars->lightRadiusScale = m_pConfVarsGUI->lightRadiusScale = 0.5f;
	m_pConfVars->photonRadiusScale = m_pConfVarsGUI->photonRadiusScale = 1.0f;

	m_pConfVars->drawRadianceTextures = m_pConfVarsGUI->drawRadianceTextures = 0;
	m_pConfVars->drawGBufferTextures = m_pConfVarsGUI->drawGBufferTextures = 0;

	m_pConfVars->UseAntiradiance = m_pConfVarsGUI->UseAntiradiance = 1;
	m_pConfVars->SeparateDirectIndirectLighting = m_pConfVarsGUI->SeparateDirectIndirectLighting = 0;
	m_pConfVars->LightingMode = m_pConfVarsGUI->LightingMode = 0;
	m_pConfVars->GatherWithAVPLAtlas = m_pConfVarsGUI->GatherWithAVPLAtlas = 0;
	m_pConfVars->GatherWithAVPLClustering = m_pConfVarsGUI->GatherWithAVPLClustering = 0;
	m_pConfVars->NoAntiradiance = m_pConfVarsGUI->NoAntiradiance = 0;
	m_pConfVars->GeoTermLimitRadiance = m_pConfVarsGUI->GeoTermLimitRadiance = 0.01f;
	m_pConfVars->GeoTermLimitAntiradiance = m_pConfVarsGUI->GeoTermLimitAntiradiance = 0.0001f;
	m_pConfVars->ClampGeoTerm = m_pConfVarsGUI->ClampGeoTerm = 0;
	m_pConfVars->ClampConeMode = m_pConfVarsGUI->ClampConeMode = 0;
	m_pConfVars->NumAVPLsPerFrame = m_pConfVarsGUI->NumAVPLsPerFrame = 100;
	m_pConfVars->NumAdditionalAVPLs = m_pConfVarsGUI->NumAdditionalAVPLs = 0;
	m_pConfVars->NumVPLsDirectLight = m_pConfVarsGUI->NumVPLsDirectLight = 500;
	m_pConfVars->NumVPLsDirectLightPerFrame = m_pConfVarsGUI->NumVPLsDirectLightPerFrame = 5;
	m_pConfVars->ConeFactor = m_pConfVarsGUI->ConeFactor = 10.f;
	m_pConfVars->ClusterRefinementThreshold = m_pConfVarsGUI->ClusterRefinementThreshold = M_PI / 10.f;
	m_pConfVars->ClusterRefinementWeight = m_pConfVarsGUI->ClusterRefinementWeight = 0.8f;
	m_pConfVars->ClusterRefinementMaxRadiance = m_pConfVarsGUI->ClusterRefinementMaxRadiance = 20.f;

	m_pConfVars->DrawError = m_pConfVarsGUI->DrawError = 0;
	m_pConfVars->DrawReference = m_pConfVarsGUI->DrawReference = 0;
	m_pConfVars->UsePathTracing = m_pConfVarsGUI->UsePathTracing = 0;
	m_pConfVars->UseMIS = m_pConfVarsGUI->UseMIS = 0;
	m_pConfVars->GaussianBlur = m_pConfVarsGUI->GaussianBlur = 0;
	m_pConfVars->NumSamples = m_pConfVarsGUI->NumSamples = 128;

	m_pConfVars->UseIBL = m_pConfVarsGUI->UseIBL = 0;
	
	m_pConfVars->UseDebugMode = m_pConfVarsGUI->UseDebugMode = 0;
	m_pConfVars->DrawAVPLAtlas = m_pConfVarsGUI->DrawAVPLAtlas = 0;
	m_pConfVars->DrawAVPLClusterAtlas = m_pConfVarsGUI->DrawAVPLClusterAtlas = 0;

	m_pConfVars->DrawLights = m_pConfVarsGUI->DrawLights = 0;
	m_pConfVars->DrawCutSizes = m_pConfVarsGUI->DrawCutSizes = 0;
	m_pConfVars->FilterAvplAtlasLinear = m_pConfVarsGUI->FilterAvplAtlasLinear = 1;
	m_pConfVars->FillAvplAltasOnGPU = m_pConfVarsGUI->FillAvplAltasOnGPU = 1;
	m_pConfVars->LimitBounces = m_pConfVarsGUI->LimitBounces = -1;
	
	m_pConfVars->UseToneMapping = m_pConfVarsGUI->UseToneMapping = 0;
	m_pConfVars->UseGammaCorrection = m_pConfVarsGUI->UseGammaCorrection = 1;
	m_pConfVars->Gamma = m_pConfVarsGUI->Gamma = 2.2f;
	m_pConfVars->Exposure = m_pConfVarsGUI->Exposure = 1.f;
	m_pConfVars->Intersection_BFC = m_pConfVarsGUI->Intersection_BFC = 1;
			
	m_pConfVars->NumAVPLsDebug = m_pConfVarsGUI->NumAVPLsDebug = 100;
	m_pConfVars->AntiradFilterMode = m_pConfVarsGUI->AntiradFilterMode = 0;
	m_pConfVars->AntiradFilterGaussFactor = m_pConfVarsGUI->AntiradFilterGaussFactor = 2.5f;
	m_pConfVars->RenderBounce = m_pConfVarsGUI->RenderBounce = -1;
	m_pConfVars->NumSqrtAtlasSamples = m_pConfVarsGUI->NumSqrtAtlasSamples = 2;
	m_pConfVars->DisplacePCP = m_pConfVarsGUI->DisplacePCP = 1.f;

	m_pConfVars->LightTreeCutDepth = m_pConfVarsGUI->LightTreeCutDepth = -1;
	m_pConfVars->ClusterDepth = m_pConfVarsGUI->ClusterDepth = 0;
	m_pConfVars->ClusterMethod = m_pConfVarsGUI->ClusterMethod = 0;
			
	m_pConfVars->DrawDirectLighting = m_pConfVarsGUI->DrawDirectLighting = 0;
	m_pConfVars->DrawIndirectLighting = m_pConfVarsGUI->DrawIndirectLighting = 0;
	
	m_pConfVars->AreaLightFrontDirection[0] = m_pConfVarsGUI->AreaLightFrontDirection[0] = 0.f;
	m_pConfVars->AreaLightFrontDirection[1] = m_pConfVarsGUI->AreaLightFrontDirection[1] = 0.f;
	m_pConfVars->AreaLightFrontDirection[2] = m_pConfVarsGUI->AreaLightFrontDirection[2] = 0.f;
	
	m_pConfVars->AreaLightPosX = m_pConfVarsGUI->AreaLightPosX = 0.f;
	m_pConfVars->AreaLightPosY = m_pConfVarsGUI->AreaLightPosY = 0.f;
	m_pConfVars->AreaLightPosZ = m_pConfVarsGUI->AreaLightPosZ = 0.f;

	m_pConfVars->AreaLightRadianceScale = m_pConfVarsGUI->AreaLightRadianceScale = 1.f;

	m_pConfVars->DrawCubeMapFace = m_pConfVarsGUI->DrawCubeMapFace = -1;
}

CConfigManager::~CConfigManager()
{
	delete [] m_pConfVars;
	delete [] m_pConfVarsGUI;
}

void CConfigManager::Update()
{
	bool configureLighting = false;
	bool clearLighting = false;
	bool clearAccumBuffer = false;
	bool updateAreaLight = false;

	if(m_pConfVarsGUI->lightRadiusScale != m_pConfVars->lightRadiusScale)
	{
		m_pConfVars->lightRadiusScale = m_pConfVarsGUI->lightRadiusScale;
		bool clearLighting = false;
		bool clearAccumBuffer = false;
	}

	if(m_pConfVarsGUI->photonRadiusScale != m_pConfVars->photonRadiusScale)
	{
		m_pConfVars->photonRadiusScale = m_pConfVarsGUI->photonRadiusScale;
		bool clearLighting = false;
		bool clearAccumBuffer = false;
	}

	if(m_pConfVarsGUI->bvhLevel != m_pConfVars->bvhLevel)
	{
		m_pConfVars->bvhLevel = m_pConfVarsGUI->bvhLevel;
		m_renderer->UpdateBvhDebug();
		bool clearLighting = false;
		bool clearAccumBuffer = false;
	}
	
	if(m_pConfVarsGUI->considerNormals != m_pConfVars->considerNormals)
	{
		m_pConfVars->considerNormals = m_pConfVarsGUI->considerNormals;
		m_renderer->RebuildBvh();
		bool clearLighting = false;
		bool clearAccumBuffer = false;
	}
	
	if(m_pConfVarsGUI->gatherClusterLightsSimple != m_pConfVars->gatherClusterLightsSimple)
	{
		m_pConfVars->gatherClusterLightsSimple = m_pConfVarsGUI->gatherClusterLightsSimple;
	}

	if(m_pConfVarsGUI->DrawClusterAABBs != m_pConfVars->DrawClusterAABBs)
	{
		m_pConfVars->DrawClusterAABBs = m_pConfVarsGUI->DrawClusterAABBs;
		bool clearLighting = false;
		bool clearAccumBuffer = false;
	}
	
	if(m_pConfVarsGUI->DrawClusterLights != m_pConfVars->DrawClusterLights)
	{
		m_pConfVars->DrawClusterLights = m_pConfVarsGUI->DrawClusterLights;
		bool clearLighting = false;
		bool clearAccumBuffer = false;
	}

	if(m_pConfVarsGUI->gatherWithCuda != m_pConfVars->gatherWithCuda)
	{
		m_pConfVars->gatherWithCuda = m_pConfVarsGUI->gatherWithCuda;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->useLightCuts != m_pConfVars->useLightCuts)
	{
		m_pConfVars->useLightCuts = m_pConfVarsGUI->useLightCuts;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->useClusteredDeferred != m_pConfVars->useClusteredDeferred)
	{
		m_pConfVars->useClusteredDeferred = m_pConfVarsGUI->useClusteredDeferred;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}
	
	if(m_pConfVarsGUI->GeoTermLimitRadiance != m_pConfVars->GeoTermLimitRadiance)
	{
		m_pConfVars->GeoTermLimitRadiance = m_pConfVarsGUI->GeoTermLimitRadiance;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->GeoTermLimitAntiradiance != m_pConfVars->GeoTermLimitAntiradiance)
	{
		m_pConfVars->GeoTermLimitAntiradiance = m_pConfVarsGUI->GeoTermLimitAntiradiance;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->ClampGeoTerm != m_pConfVars->ClampGeoTerm)
	{
		m_pConfVars->ClampGeoTerm = m_pConfVarsGUI->ClampGeoTerm;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->ClampConeMode != m_pConfVars->ClampConeMode)
	{
		m_pConfVars->ClampConeMode = m_pConfVarsGUI->ClampConeMode;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->Gamma != m_pConfVars->Gamma)
	{
		m_pConfVars->Gamma = m_pConfVarsGUI->Gamma;
		configureLighting = true;
	}

	if(m_pConfVarsGUI->NoAntiradiance != m_pConfVars->NoAntiradiance)
	{
		m_pConfVars->NoAntiradiance = m_pConfVarsGUI->NoAntiradiance;
	}

	if(m_pConfVarsGUI->DrawError != m_pConfVars->DrawError)
	{
		m_pConfVars->DrawError = m_pConfVarsGUI->DrawError;
	}

	if(m_pConfVarsGUI->DrawReference != m_pConfVars->DrawReference)
	{
		m_pConfVars->DrawReference = m_pConfVarsGUI->DrawReference;
	}

	if(m_pConfVarsGUI->NumSamples != m_pConfVars->NumSamples)
	{
		m_pConfVars->NumSamples = m_pConfVarsGUI->NumSamples;
	}

	if(m_pConfVarsGUI->GaussianBlur != m_pConfVars->GaussianBlur)
	{
		m_pConfVars->GaussianBlur = m_pConfVarsGUI->GaussianBlur;
	}

	if(m_pConfVarsGUI->Exposure != m_pConfVars->Exposure)
	{
		m_pConfVars->Exposure = m_pConfVarsGUI->Exposure;
		configureLighting = true;
	}

	if(m_pConfVarsGUI->UseAntiradiance != m_pConfVars->UseAntiradiance)
	{
		m_pConfVars->UseAntiradiance = m_pConfVarsGUI->UseAntiradiance;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->UseToneMapping != m_pConfVars->UseToneMapping)
	{
		m_pConfVars->UseToneMapping = m_pConfVarsGUI->UseToneMapping;
		configureLighting = true;
	}

	if(m_pConfVarsGUI->UseGammaCorrection != m_pConfVars->UseGammaCorrection)
	{
		m_pConfVars->UseGammaCorrection = m_pConfVarsGUI->UseGammaCorrection;
		configureLighting = true;
	}

	if(m_pConfVarsGUI->UseDebugMode != m_pConfVars->UseDebugMode)
	{
		m_pConfVars->UseDebugMode = m_pConfVarsGUI->UseDebugMode;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}
	
	if(m_pConfVarsGUI->LimitBounces != m_pConfVars->LimitBounces)
	{
		m_pConfVars->LimitBounces = m_pConfVarsGUI->LimitBounces;
		clearAccumBuffer = true;
	}

	if(m_pConfVarsGUI->SeparateDirectIndirectLighting != m_pConfVars->SeparateDirectIndirectLighting)
	{
		m_pConfVars->SeparateDirectIndirectLighting = m_pConfVarsGUI->SeparateDirectIndirectLighting;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->LightingMode != m_pConfVars->LightingMode)
	{
		m_pConfVars->LightingMode = m_pConfVarsGUI->LightingMode;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->DrawDirectLighting != m_pConfVars->DrawDirectLighting)
	{
		m_pConfVars->DrawDirectLighting = m_pConfVarsGUI->DrawDirectLighting;
		configureLighting = true;
	}

	if(m_pConfVarsGUI->DrawIndirectLighting != m_pConfVars->DrawIndirectLighting)
	{
		m_pConfVars->DrawIndirectLighting = m_pConfVarsGUI->DrawIndirectLighting;
		configureLighting = true;
	}
	
	if(m_pConfVarsGUI->DrawAVPLAtlas != m_pConfVars->DrawAVPLAtlas)
	{
		m_pConfVars->DrawAVPLAtlas = m_pConfVarsGUI->DrawAVPLAtlas;
		configureLighting = true;
	}

	if(m_pConfVarsGUI->DrawAVPLClusterAtlas != m_pConfVars->DrawAVPLClusterAtlas)
	{
		m_pConfVars->DrawAVPLClusterAtlas = m_pConfVarsGUI->DrawAVPLClusterAtlas;
		configureLighting = true;
	}

	if(m_pConfVarsGUI->drawRadianceTextures != m_pConfVars->drawRadianceTextures)
	{
		m_pConfVars->drawRadianceTextures = m_pConfVarsGUI->drawRadianceTextures;
	}

	if(m_pConfVarsGUI->drawGBufferTextures != m_pConfVars->drawGBufferTextures)
	{
		m_pConfVars->drawGBufferTextures = m_pConfVarsGUI->drawGBufferTextures;
	}

	if(m_pConfVarsGUI->DrawCutSizes != m_pConfVars->DrawCutSizes)
	{
		m_pConfVars->DrawCutSizes = m_pConfVarsGUI->DrawCutSizes;
	}

	if(m_pConfVarsGUI->DrawLights != m_pConfVars->DrawLights)
	{
		m_pConfVars->DrawLights = m_pConfVarsGUI->DrawLights;
		configureLighting = true;
		clearAccumBuffer = true;
	}

	if(m_pConfVarsGUI->UsePathTracing != m_pConfVars->UsePathTracing)
	{
		m_pConfVars->UsePathTracing = m_pConfVarsGUI->UsePathTracing;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->UseMIS != m_pConfVars->UseMIS)
	{
		m_pConfVars->UseMIS = m_pConfVarsGUI->UseMIS;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->FilterAvplAtlasLinear != m_pConfVars->FilterAvplAtlasLinear)
	{
		m_pConfVars->FilterAvplAtlasLinear = m_pConfVarsGUI->FilterAvplAtlasLinear;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->FillAvplAltasOnGPU != m_pConfVars->FillAvplAltasOnGPU)
	{
		m_pConfVars->FillAvplAltasOnGPU = m_pConfVarsGUI->FillAvplAltasOnGPU;
		clearAccumBuffer = true;
	}

	if(m_pConfVarsGUI->GatherWithAVPLAtlas != m_pConfVars->GatherWithAVPLAtlas)
	{
		m_pConfVars->GatherWithAVPLAtlas = m_pConfVarsGUI->GatherWithAVPLAtlas;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->GatherWithAVPLClustering != m_pConfVars->GatherWithAVPLClustering)
	{
		m_pConfVars->GatherWithAVPLClustering = m_pConfVarsGUI->GatherWithAVPLClustering;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->ConeFactor != m_pConfVars->ConeFactor)
	{
		m_pConfVars->ConeFactor = m_pConfVarsGUI->ConeFactor;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->AntiradFilterMode != m_pConfVars->AntiradFilterMode)
	{
		m_pConfVars->AntiradFilterMode = m_pConfVarsGUI->AntiradFilterMode;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->AntiradFilterGaussFactor != m_pConfVars->AntiradFilterGaussFactor)
	{
		m_pConfVars->AntiradFilterGaussFactor = m_pConfVarsGUI->AntiradFilterGaussFactor;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->NumVPLsDirectLight != m_pConfVars->NumVPLsDirectLight)
	{
		m_pConfVars->NumVPLsDirectLight = m_pConfVarsGUI->NumVPLsDirectLight;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->NumVPLsDirectLightPerFrame != m_pConfVars->NumVPLsDirectLightPerFrame)
	{
		m_pConfVars->NumVPLsDirectLightPerFrame = m_pConfVarsGUI->NumVPLsDirectLightPerFrame;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}
	
	if(m_pConfVarsGUI->NumAVPLsPerFrame != m_pConfVars->NumAVPLsPerFrame)
	{
		m_pConfVars->NumAVPLsPerFrame = m_pConfVarsGUI->NumAVPLsPerFrame;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->NumAVPLsDebug != m_pConfVars->NumAVPLsDebug)
	{
		m_pConfVars->NumAVPLsDebug = m_pConfVarsGUI->NumAVPLsDebug;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}


	if(m_pConfVarsGUI->NumAdditionalAVPLs != m_pConfVars->NumAdditionalAVPLs)
	{
		m_pConfVars->NumAdditionalAVPLs = m_pConfVarsGUI->NumAdditionalAVPLs;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->RenderBounce != m_pConfVars->RenderBounce)
	{
		m_pConfVars->RenderBounce = m_pConfVarsGUI->RenderBounce;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->NumSqrtAtlasSamples != m_pConfVars->NumSqrtAtlasSamples)
	{
		m_pConfVars->NumSqrtAtlasSamples = m_pConfVarsGUI->NumSqrtAtlasSamples;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->DisplacePCP != m_pConfVars->DisplacePCP)
	{
		m_pConfVars->DisplacePCP = m_pConfVarsGUI->DisplacePCP;
	}

	if(m_pConfVarsGUI->LightTreeCutDepth != m_pConfVars->LightTreeCutDepth)
	{
		m_pConfVars->LightTreeCutDepth = m_pConfVarsGUI->LightTreeCutDepth;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->ClusterDepth != m_pConfVars->ClusterDepth)
	{
		m_pConfVars->ClusterDepth = m_pConfVarsGUI->ClusterDepth;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->ClusterMethod != m_pConfVars->ClusterMethod)
	{
		m_pConfVars->ClusterMethod = m_pConfVarsGUI->ClusterMethod;
		clearLighting = true;
	}
	
	if(m_pConfVarsGUI->ClusterRefinementThreshold != m_pConfVars->ClusterRefinementThreshold)
	{
		m_pConfVars->ClusterRefinementThreshold = m_pConfVarsGUI->ClusterRefinementThreshold;
		configureLighting = true;
		clearAccumBuffer = true;
	}

	if(m_pConfVarsGUI->ClusterRefinementMaxRadiance != m_pConfVars->ClusterRefinementMaxRadiance)
	{
		m_pConfVars->ClusterRefinementMaxRadiance = m_pConfVarsGUI->ClusterRefinementMaxRadiance;
		configureLighting = true;
		clearAccumBuffer = true;
	}

	if(m_pConfVarsGUI->ClusterRefinementWeight != m_pConfVars->ClusterRefinementWeight)
	{
		m_pConfVars->ClusterRefinementWeight = m_pConfVarsGUI->ClusterRefinementWeight;
		configureLighting = true;
		clearAccumBuffer = true;
	}

	if(m_pConfVarsGUI->AreaLightFrontDirection[0] != m_pConfVars->AreaLightFrontDirection[0] 
		|| m_pConfVarsGUI->AreaLightFrontDirection[1] != m_pConfVars->AreaLightFrontDirection[1]
		|| m_pConfVarsGUI->AreaLightFrontDirection[2] != m_pConfVars->AreaLightFrontDirection[2])
	{
		m_pConfVars->AreaLightFrontDirection[0] = m_pConfVarsGUI->AreaLightFrontDirection[0];
		m_pConfVars->AreaLightFrontDirection[1] = m_pConfVarsGUI->AreaLightFrontDirection[1];
		m_pConfVars->AreaLightFrontDirection[2] = m_pConfVarsGUI->AreaLightFrontDirection[2];
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
		updateAreaLight = true;
	}

	if(m_pConfVarsGUI->AreaLightPosX != m_pConfVars->AreaLightPosX)
	{
		m_pConfVars->AreaLightPosX = m_pConfVarsGUI->AreaLightPosX;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
		updateAreaLight = true;
	}

	if(m_pConfVarsGUI->AreaLightPosY != m_pConfVars->AreaLightPosY)
	{
		m_pConfVars->AreaLightPosY = m_pConfVarsGUI->AreaLightPosY;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
		updateAreaLight = true;
	}

	if(m_pConfVarsGUI->AreaLightPosZ != m_pConfVars->AreaLightPosZ)
	{
		m_pConfVars->AreaLightPosZ = m_pConfVarsGUI->AreaLightPosZ;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
		updateAreaLight = true;
	}

	if(m_pConfVarsGUI->AreaLightRadianceScale != m_pConfVars->AreaLightRadianceScale)
	{
		m_pConfVars->AreaLightRadianceScale = m_pConfVarsGUI->AreaLightRadianceScale;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
		updateAreaLight = true;
	}

	if(m_pConfVarsGUI->Intersection_BFC != m_pConfVars->Intersection_BFC)
	{
		m_pConfVars->Intersection_BFC = m_pConfVarsGUI->Intersection_BFC;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->DrawCubeMapFace != m_pConfVars->DrawCubeMapFace)
	{
		m_pConfVars->DrawCubeMapFace = m_pConfVarsGUI->DrawCubeMapFace;
	}
		
	if(updateAreaLight) m_renderer->UpdateAreaLights();
	if(clearLighting) m_renderer->IssueClearLighting();
	if(clearAccumBuffer) m_renderer->IssueClearAccumulationBuffer();
	if(configureLighting) m_renderer->UpdateUniformBuffers();
}
