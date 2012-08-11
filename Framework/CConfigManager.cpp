#include "CConfigManager.h"

#include "Render.h"

#include <iostream>

CConfigManager::CConfigManager(Renderer* pRenderer)
{
	m_pRenderer = pRenderer;

	m_pConfVars = new CONF_VARS[1];
	m_pConfVarsGUI = new CONF_VARS[1];

	m_pConfVars->UseAntiradiance = m_pConfVarsGUI->UseAntiradiance = 1;
	m_pConfVars->UseToneMapping = m_pConfVarsGUI->UseToneMapping = 0;
	m_pConfVars->UseDebugMode = m_pConfVarsGUI->UseDebugMode = 0;
	m_pConfVars->DrawAVPLAtlas = m_pConfVarsGUI->DrawAVPLAtlas = 0;
	m_pConfVars->DrawAVPLClusterAtlas = m_pConfVarsGUI->DrawAVPLClusterAtlas = 0;
	m_pConfVars->GatherWithAVPLAtlas = m_pConfVarsGUI->GatherWithAVPLAtlas = 0;
	m_pConfVars->GatherWithAVPLClustering = m_pConfVarsGUI->GatherWithAVPLClustering = 0;
	m_pConfVars->DrawDebugTextures = m_pConfVarsGUI->DrawDebugTextures = 0;
	m_pConfVars->DrawLights = m_pConfVarsGUI->DrawLights = 0;
	m_pConfVars->DrawCutSizes = m_pConfVarsGUI->DrawCutSizes = 0;
	m_pConfVars->FilterAvplAtlasLinear = m_pConfVarsGUI->FilterAvplAtlasLinear = 0;
	m_pConfVars->FillAvplAltasOnGPU = m_pConfVarsGUI->FillAvplAltasOnGPU = 1;
	m_pConfVars->UseLightTree = m_pConfVarsGUI->UseLightTree = 0;
	m_pConfVars->LimitBounces = m_pConfVarsGUI->LimitBounces = -1;
	
	m_pConfVars->GeoTermLimitRadiance = m_pConfVarsGUI->GeoTermLimitRadiance = 1.0f;
	m_pConfVars->GeoTermLimitAntiradiance = m_pConfVarsGUI->GeoTermLimitAntiradiance = 1.0f;
	m_pConfVars->ClampGeoTerm = m_pConfVarsGUI->ClampGeoTerm = 1;
	m_pConfVars->ClampCone = m_pConfVarsGUI->ClampCone = 1;
	m_pConfVars->Gamma = m_pConfVarsGUI->Gamma = 2.2f;
	m_pConfVars->Exposure = m_pConfVarsGUI->Exposure = 1.f;
	m_pConfVars->Intersection_BFC = m_pConfVarsGUI->Intersection_BFC = 1;

	m_pConfVars->NumVPLsDirectLight = m_pConfVarsGUI->NumVPLsDirectLight = 50;
	m_pConfVars->NumVPLsDirectLightPerFrame = m_pConfVarsGUI->NumVPLsDirectLightPerFrame = 1;

	m_pConfVars->NumPaths = m_pConfVarsGUI->NumPaths = 1000;
	m_pConfVars->NumPathsPerFrame = m_pConfVarsGUI->NumPathsPerFrame = 100;
	m_pConfVars->NumAdditionalAVPLs = m_pConfVarsGUI->NumAdditionalAVPLs = 0;
	m_pConfVars->ConeFactor = m_pConfVarsGUI->ConeFactor = 30.f;
	m_pConfVars->AntiradFilterMode = m_pConfVarsGUI->AntiradFilterMode = 0;
	m_pConfVars->AntiradFilterGaussFactor = m_pConfVarsGUI->AntiradFilterGaussFactor = 2.5f;
	m_pConfVars->RenderBounce = m_pConfVarsGUI->RenderBounce = -1;
	m_pConfVars->DrawLightingOfLight = m_pConfVarsGUI->DrawLightingOfLight = -1;
	m_pConfVars->NumSqrtAtlasSamples = m_pConfVarsGUI->NumSqrtAtlasSamples = 4;
	m_pConfVars->TexelOffsetX = m_pConfVarsGUI->TexelOffsetX = 0.f;
	m_pConfVars->TexelOffsetY = m_pConfVarsGUI->TexelOffsetY = 0.f;
	m_pConfVars->DisplacePCP = m_pConfVarsGUI->DisplacePCP = 1.f;

	m_pConfVars->LightTreeCutDepth = m_pConfVarsGUI->LightTreeCutDepth = -1;
	m_pConfVars->ClusterDepth = m_pConfVarsGUI->ClusterDepth = 0;
	m_pConfVars->ClusterMethod = m_pConfVarsGUI->ClusterMethod = 0;
	m_pConfVars->ClusterWeightNormals = m_pConfVarsGUI->ClusterWeightNormals = 0.5f;
	m_pConfVars->ClusterRefinementThreshold = m_pConfVarsGUI->ClusterRefinementThreshold = 0.01f;

	m_pConfVars->SeparateDirectIndirectLighting = m_pConfVarsGUI->SeparateDirectIndirectLighting = 0;
	m_pConfVars->DrawDirectLight = m_pConfVarsGUI->DrawDirectLight = 0;
	m_pConfVars->DrawIndirectLight = m_pConfVarsGUI->DrawIndirectLight = 0;

	m_pConfVars->AreaLightFrontDirection[0] = m_pConfVarsGUI->AreaLightFrontDirection[0] = 0.f;
	m_pConfVars->AreaLightFrontDirection[1] = m_pConfVarsGUI->AreaLightFrontDirection[1] = 0.f;
	m_pConfVars->AreaLightFrontDirection[2] = m_pConfVarsGUI->AreaLightFrontDirection[2] = 0.f;
	
	m_pConfVars->AreaLightPosX = m_pConfVarsGUI->AreaLightPosX = 0.f;
	m_pConfVars->AreaLightPosY = m_pConfVarsGUI->AreaLightPosY = 0.f;
	m_pConfVars->AreaLightPosZ = m_pConfVarsGUI->AreaLightPosZ = 0.f;

	m_pConfVars->AreaLightRadianceScale = m_pConfVarsGUI->AreaLightRadianceScale = 1.f;

	m_pConfVars->UseAVPLImportanceSampling = m_pConfVarsGUI->UseAVPLImportanceSampling = 1;
	m_pConfVars->ISMode = m_pConfVarsGUI->ISMode = 0;
	m_pConfVars->ConeFactorIS = m_pConfVarsGUI->ConeFactorIS = 4;
	m_pConfVars->NumSceneSamples = m_pConfVarsGUI->NumSceneSamples = 100;
	m_pConfVars->DrawSceneSamples = m_pConfVarsGUI->DrawSceneSamples = 0;
	m_pConfVars->DrawCollectedAVPLs = m_pConfVarsGUI->DrawCollectedAVPLs = 0;
	m_pConfVars->DrawCollectedISAVPLs = m_pConfVarsGUI->DrawCollectedISAVPLs = 0;
	m_pConfVars->CollectAVPLs = m_pConfVarsGUI->CollectAVPLs = 0;
	m_pConfVars->CollectISAVPLs = m_pConfVarsGUI->CollectISAVPLs = 0;
	m_pConfVars->IrradAntiirradWeight = m_pConfVarsGUI->IrradAntiirradWeight = 0.5f;
	m_pConfVars->AcceptProbabEpsilon = m_pConfVarsGUI->AcceptProbabEpsilon = 0.05f;

	m_pConfVars->UseBIDIR = m_pConfVarsGUI->UseBIDIR = 0;
	m_pConfVars->DrawBIDIRSceneSamples = m_pConfVarsGUI->DrawBIDIRSceneSamples = 0;
	m_pConfVars->NumEyeRays = m_pConfVarsGUI->NumEyeRays = 100;
	m_pConfVars->NumSamplesForPE = m_pConfVarsGUI->NumSamplesForPE = 64;
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

	if(m_pConfVarsGUI->ClampCone != m_pConfVars->ClampCone)
	{
		m_pConfVars->ClampCone = m_pConfVarsGUI->ClampCone;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->Gamma != m_pConfVars->Gamma)
	{
		m_pConfVars->Gamma = m_pConfVarsGUI->Gamma;
		configureLighting = true;
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

	if(m_pConfVarsGUI->UseDebugMode != m_pConfVars->UseDebugMode)
	{
		m_pConfVars->UseDebugMode = m_pConfVarsGUI->UseDebugMode;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->UseLightTree != m_pConfVars->UseLightTree)
	{
		m_pConfVars->UseLightTree = m_pConfVarsGUI->UseLightTree;
		clearAccumBuffer = true;
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

	if(m_pConfVarsGUI->DrawDirectLight != m_pConfVars->DrawDirectLight)
	{
		m_pConfVars->DrawDirectLight = m_pConfVarsGUI->DrawDirectLight;
		configureLighting = true;
	}

	if(m_pConfVarsGUI->DrawIndirectLight != m_pConfVars->DrawIndirectLight)
	{
		m_pConfVars->DrawIndirectLight = m_pConfVarsGUI->DrawIndirectLight;
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

	if(m_pConfVarsGUI->DrawDebugTextures != m_pConfVars->DrawDebugTextures)
	{
		m_pConfVars->DrawDebugTextures = m_pConfVarsGUI->DrawDebugTextures;
		configureLighting = true;
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

	if(m_pConfVarsGUI->NumPaths != m_pConfVars->NumPaths)
	{
		m_pConfVars->NumPaths = m_pConfVarsGUI->NumPaths;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->NumPathsPerFrame != m_pConfVars->NumPathsPerFrame)
	{
		m_pConfVars->NumPathsPerFrame = m_pConfVarsGUI->NumPathsPerFrame;
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

	if(m_pConfVarsGUI->DrawLightingOfLight != m_pConfVars->DrawLightingOfLight)
	{
		m_pConfVars->DrawLightingOfLight = m_pConfVarsGUI->DrawLightingOfLight;
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

	if(m_pConfVarsGUI->TexelOffsetX != m_pConfVars->TexelOffsetX)
	{
		m_pConfVars->TexelOffsetX = m_pConfVarsGUI->TexelOffsetX;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->TexelOffsetY != m_pConfVars->TexelOffsetY)
	{
		m_pConfVars->TexelOffsetY = m_pConfVarsGUI->TexelOffsetY;
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

	if(m_pConfVarsGUI->ClusterWeightNormals != m_pConfVars->ClusterWeightNormals)
	{
		m_pConfVars->ClusterWeightNormals = m_pConfVarsGUI->ClusterWeightNormals;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->ClusterRefinementThreshold != m_pConfVars->ClusterRefinementThreshold)
	{
		m_pConfVars->ClusterRefinementThreshold = m_pConfVarsGUI->ClusterRefinementThreshold;
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
			
	if(m_pConfVarsGUI->UseAVPLImportanceSampling != m_pConfVars->UseAVPLImportanceSampling)
	{
		m_pConfVars->UseAVPLImportanceSampling = m_pConfVarsGUI->UseAVPLImportanceSampling;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->ISMode != m_pConfVars->ISMode)
	{
		m_pConfVars->ISMode = m_pConfVarsGUI->ISMode;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}
	
	if(m_pConfVarsGUI->ConeFactorIS != m_pConfVars->ConeFactorIS)
	{
		m_pConfVars->ConeFactorIS = m_pConfVarsGUI->ConeFactorIS;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->DrawSceneSamples != m_pConfVars->DrawSceneSamples)
	{
		m_pConfVars->DrawSceneSamples = m_pConfVarsGUI->DrawSceneSamples;
	}

	if(m_pConfVarsGUI->DrawCollectedAVPLs != m_pConfVars->DrawCollectedAVPLs)
	{
		m_pConfVars->DrawCollectedAVPLs = m_pConfVarsGUI->DrawCollectedAVPLs;
	}

	if(m_pConfVarsGUI->DrawCollectedISAVPLs != m_pConfVars->DrawCollectedISAVPLs)
	{
		m_pConfVars->DrawCollectedISAVPLs = m_pConfVarsGUI->DrawCollectedISAVPLs;
	}

	if(m_pConfVarsGUI->CollectAVPLs != m_pConfVars->CollectAVPLs)
	{
		m_pConfVars->CollectAVPLs = m_pConfVarsGUI->CollectAVPLs;
	}

	if(m_pConfVarsGUI->CollectISAVPLs != m_pConfVars->CollectISAVPLs)
	{
		m_pConfVars->CollectISAVPLs = m_pConfVarsGUI->CollectISAVPLs;
	}

	if(m_pConfVarsGUI->NumSceneSamples != m_pConfVars->NumSceneSamples)
	{
		m_pConfVars->NumSceneSamples = m_pConfVarsGUI->NumSceneSamples;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->IrradAntiirradWeight != m_pConfVars->IrradAntiirradWeight)
	{
		m_pConfVars->IrradAntiirradWeight = m_pConfVarsGUI->IrradAntiirradWeight;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->AcceptProbabEpsilon != m_pConfVars->AcceptProbabEpsilon)
	{
		m_pConfVars->AcceptProbabEpsilon = m_pConfVarsGUI->AcceptProbabEpsilon;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->UseBIDIR != m_pConfVars->UseBIDIR)
	{
		m_pConfVars->UseBIDIR = m_pConfVarsGUI->UseBIDIR;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->NumEyeRays != m_pConfVars->NumEyeRays)
	{
		m_pConfVars->NumEyeRays = m_pConfVarsGUI->NumEyeRays;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->NumSamplesForPE != m_pConfVars->NumSamplesForPE)
	{
		m_pConfVars->NumSamplesForPE = m_pConfVarsGUI->NumSamplesForPE;
		configureLighting = true;
		clearAccumBuffer = true;
		clearLighting = true;
	}

	if(m_pConfVarsGUI->DrawBIDIRSceneSamples != m_pConfVars->DrawBIDIRSceneSamples)
	{
		m_pConfVars->DrawBIDIRSceneSamples = m_pConfVarsGUI->DrawBIDIRSceneSamples;
	}
		
	if(updateAreaLight) m_pRenderer->UpdateAreaLights();
	if(clearLighting) m_pRenderer->ClearLighting();
	if(clearAccumBuffer) m_pRenderer->ClearAccumulationBuffer();
	if(configureLighting) m_pRenderer->ConfigureLighting();
}