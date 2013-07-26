#ifndef _C_CONFIG_MANAGER_H_
#define _C_CONFIG_MANAGER_H_

class Renderer;

struct CONF_VARS
{
	int gatherWithCuda;
	int useLightCuts;
	int useClusteredDeferred;
	int gatherClusterLightsSimple;
	int bvhLevel;
	int considerNormals;
	int DrawLights;
	int DrawClusterLights;
	int DrawClusterAABBs;

	int explicitDirectIllum;

	float lightRadiusScale;
	float photonRadiusScale;

	int drawRadianceTextures;
	int drawGBufferTextures;

	int DrawError;
	int DrawReference;

	int UseAntiradiance;
	int UseDebugMode;
	
	int SeparateDirectIndirectLighting;
	int LightingMode;
	
	int NumVPLsDirectLight;
	int NumVPLsDirectLightPerFrame;
		
	int UseGammaCorrection;
	float Gamma;
	float Exposure;
	
	int NumAVPLsPerFrame;
	int NumAVPLsDebug;
	
	float ClusterRefinementThreshold;

	float AreaLightFrontDirection[3];
	float AreaLightPosX;
	float AreaLightPosY;
	float AreaLightPosZ;
	float AreaLightRadianceScale;
};

class CConfigManager
{
public:
	CConfigManager();
	~CConfigManager();

	void setRenderer(Renderer* renderer) { m_renderer = renderer; }
	CONF_VARS* GetConfVarsGUI() { return m_pConfVarsGUI; }
	CONF_VARS* GetConfVars() { return m_pConfVars; }

	void Update();

private:
	CONF_VARS* m_pConfVarsGUI;
	CONF_VARS* m_pConfVars;

	Renderer* m_renderer;
};

#endif // _C_CONFIG_MANAGER_H_
