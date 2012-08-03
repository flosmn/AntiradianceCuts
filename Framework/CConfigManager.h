#ifndef _C_CONFIG_MANAGER_H_
#define _C_CONFIG_MANAGER_H_

class Renderer;

struct CONF_VARS
{
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
	int UseLightTree;
	int SeparateDirectIndirectLighting;
	int DrawDirectLight;
	int DrawIndirectLight;
	int NumVPLsDirectLight;
	int NumVPLsDirectLightPerFrame;
	int AntiradFilterMode;
	float AntiradFilterGaussFactor;
	float AntiradFilterK;

	float GeoTermLimit;
	int ClampGeoTerm;
	float Gamma;
	float Exposure;
	int Intersection_BFC;
	
	int ConeFactor;
	int NumPaths;
	int NumPathsPerFrame;
	int NumAdditionalAVPLs;
	int RenderBounce;
	int DrawLightingOfLight;
	int NumSqrtAtlasSamples;
	float TexelOffsetX;
	float TexelOffsetY;
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