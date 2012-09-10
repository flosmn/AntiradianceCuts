#ifndef _C_CONFIG_MANAGER_H_
#define _C_CONFIG_MANAGER_H_

class Renderer;

struct CONF_VARS
{
	int DrawError;
	int DrawReference;

	int UseAntiradiance;
	int UseToneMapping;
	int UseDebugMode;
	int DrawAVPLAtlas;
	int DrawAVPLClusterAtlas;
	int GatherWithAVPLAtlas;
	int GatherWithAVPLClustering;
	int DrawDebugTextures;
	int DrawLights;
	int DrawCutSizes;
	int FilterAvplAtlasLinear;
	int FillAvplAltasOnGPU;
	int SeparateDirectIndirectLighting;
	int LightingMode;
	int DrawDirectLighting;
	int DrawIndirectLighting;
	int NumVPLsDirectLight;
	int NumVPLsDirectLightPerFrame;
	int AntiradFilterMode;
	float AntiradFilterGaussFactor;
	float AntiradFilterK;
	int LimitBounces;
	int NoAntiradiance;

	int UsePathTracing;
	int UseMIS;
	int GaussianBlur;
	int NumSamples;

	int UseIBL;

	float GeoTermLimitRadiance;
	float GeoTermLimitAntiradiance;
	int ClampGeoTerm;
	int ClampCone;
	float Gamma;
	float Exposure;
	int Intersection_BFC;
	
	int NumAVPLsPerFrame;
	int NumAVPLsDebug;
	int NumAdditionalAVPLs;
	int RenderBounce;
	int NumSqrtAtlasSamples;
	float ConeFactor;
	float DisplacePCP;

	int LightTreeCutDepth;
	int ClusterDepth;
	int ClusterMethod;
	float ClusterWeightNormals;
	float ClusterRefinementThreshold;

	float AreaLightFrontDirection[3];
	float AreaLightPosX;
	float AreaLightPosY;
	float AreaLightPosZ;
	float AreaLightRadianceScale;

	int UseAVPLImportanceSampling;
	int ISMode;
	int ConeFactorIS;
	int NumSceneSamples;
	int DrawSceneSamples;
	int DrawCollectedAVPLs;
	int DrawCollectedISAVPLs;
	int CollectAVPLs;
	int CollectISAVPLs;
	float IrradAntiirradWeight;
	float AcceptProbabEpsilon;

	int UseBIDIR;
	int UseStratification;
	int DrawBIDIRSamples;
	int DrawBIDIRSamplesMode;
	int NumEyeRaysSS;	
	int NumSamplesForPESS;	
	int NumEyeRaysASS;	
	int NumSamplesForPEASS;

	int DrawCubeMapFace;
};

class CConfigManager
{
public:
	CConfigManager(Renderer* renderer);
	~CConfigManager();

	CONF_VARS* GetConfVarsGUI() { return m_pConfVarsGUI; }
	CONF_VARS* GetConfVars() { return m_pConfVars; }

	void Update();

private:
	CONF_VARS* m_pConfVarsGUI;
	CONF_VARS* m_pConfVars;

	Renderer* m_pRenderer;
};

#endif // _C_CONFIG_MANAGER_H_