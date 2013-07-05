#include "CGUI.h"

#include "CConfigManager.h"

CGUI::CGUI(CConfigManager* pConfigManager)
	: m_pConfigManager(pConfigManager), m_fps(0.f)
{
}

CGUI::~CGUI()
{
}

bool CGUI::Init(uint window_width, uint window_height)
{
	if(!TwInit(TW_OPENGL, NULL))
	{
		return false;
	}

	TwWindowSize(window_width, window_height);

	m_pTwBar = TwNewBar("GUI");
		
	TwAddVarRO(m_pTwBar, "fps", TW_TYPE_FLOAT, &m_fps, "");

	TwAddSeparator(m_pTwBar, "", "");
		
	TwAddVarRW(m_pTwBar, "Gather with cuda", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->gatherWithCuda), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Use Antiradiance", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseAntiradiance), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Gather With AVPL Clustering", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->GatherWithAVPLClustering), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Sep D/I Lighting", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->SeparateDirectIndirectLighting), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Lighting Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->LightingMode), " min=0 max=2 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Direct Lighting", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawDirectLighting), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Indirect Lighting", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawIndirectLighting), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "#AVPLs per frame", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumAVPLsPerFrame), " min=1 max=200000 step=1 ");
	TwAddVarRW(m_pTwBar, "#add. AVPL", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumAdditionalAVPLs), " min=0 max=2048 step=1 ");
	TwAddVarRW(m_pTwBar, "#VPLs DL", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumVPLsDirectLight), " min=0 max=10000 step=1 ");
	TwAddVarRW(m_pTwBar, "#VPLs DL Per Frame", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumVPLsDirectLightPerFrame), " min=0 max=10000 step=1 ");
	TwAddVarRW(m_pTwBar, "Cluster Refinement Threshold", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ClusterRefinementThreshold), "min=0.0 max=7.0 step=0.01");
	TwAddVarRW(m_pTwBar, "Cluster Refinement Max Radiance", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ClusterRefinementMaxRadiance), "min=0.0 max=10000.0 step=1.f");
	TwAddVarRW(m_pTwBar, "Cluster Refinement Weight", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ClusterRefinementWeight), "min=0.0 max=1.f step=0.05f");
	TwAddVarRW(m_pTwBar, "Draw Error", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawError), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Reference", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawReference), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Cone Factor", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ConeFactor), " min=0.1 max=1000.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "No Antiradiance", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NoAntiradiance), " min=0 max=1 step=1 ");
	
	TwAddSeparator(m_pTwBar, "", "");

	TwAddVarRW(m_pTwBar, "Clamp Radiance", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->GeoTermLimitRadiance), "min=0 max=100000 step=0.00001");
	TwAddVarRW(m_pTwBar, "Clamp Antiradiance", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->GeoTermLimitAntiradiance), "min=0 max=100000 step=0.00001");
	TwAddVarRW(m_pTwBar, "Clamp Geo-Term", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ClampGeoTerm), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Clamp Cone", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ClampConeMode), "min=0 max=2 step=1");

	TwAddSeparator(m_pTwBar, "", "");
		
	TwAddVarRW(m_pTwBar, "#AVPLs debug", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumAVPLsDebug), " min=1 max=100000 step=1 ");
	TwAddVarRW(m_pTwBar, "Use Debug Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseDebugMode), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Use Debug Textures", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawDebugTextures), "min=0 max=1 step=1");
	
	TwAddSeparator(m_pTwBar, "", "");

	TwAddVarRW(m_pTwBar, "Antirad Filter Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->AntiradFilterMode), " min=0 max=2 step=1 ");
	TwAddVarRW(m_pTwBar, "Antirad Filter Gauss Factor", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AntiradFilterGaussFactor), " min=1.0 max=4.0 step=0.1 ");
		
	TwAddSeparator(m_pTwBar, "", "");
	
	TwAddVarRW(m_pTwBar, "Draw cut sizes", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawCutSizes), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw AVPL atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawAVPLAtlas), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw AVPL cluster atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawAVPLClusterAtlas), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw Lights", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawLights), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw AABBs", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawAABBs), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "BVH level", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->bvhLevel), " min=0 max=100 step=1 ");
	TwAddVarRW(m_pTwBar, "Consider Normals", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->considerNormals), " min=0 max=1 step=1 ");
		
	TwAddSeparator(m_pTwBar, "", "");

	TwAddVarRW(m_pTwBar, "Use Gamma Correction", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseGammaCorrection), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Gamma", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->Gamma), " min=0.0 max=100.0 step=0.1 ");
	
	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "AreaLight Front", TW_TYPE_DIR3F, &(m_pConfigManager->GetConfVarsGUI()->AreaLightFrontDirection), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Pos X", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightPosX), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Pos Y", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightPosY), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Pos Z", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightPosZ), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Radiance Scale", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightRadianceScale), "min=0.0 max=10000.0 step=1.0");

	return true;
}

void CGUI::Release()
{
}

void CGUI::Render(float fps)
{
	m_fps = fps;

	TwDraw();
}

bool CGUI::HandleEvent(void* wnd, uint msg, WPARAM wParam, LPARAM lParam)
{
	if(TwEventWin(wnd, msg, wParam, lParam))
		return true;

	return false;
}
