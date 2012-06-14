#ifndef _C_CONFIG_MANAGER_H_
#define _C_CONFIG_MANAGER_H_

class Renderer;

struct CONF_VARS
{
	int UseAntiradiance;
	int UseToneMapping;
	int UseDebugMode;
	int DrawAVPLAtlas;
	int GatherWithAVPLAtlas;
	int DrawDebugTextures;
	int DrawLights;
	int FilterAvplAtlasLinear;

	float GeoTermLimit;
	float Gamma;
	float Exposure;
	
	int ConeFactor;
	int NumPaths;
	int NumPathsPerFrame;
	int NumAdditionalAVPLs;
	int RenderBounce;
	int DrawLightingOfLight;
	int NumSqrtAtlasSamples;
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