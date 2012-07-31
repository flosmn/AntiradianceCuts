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
	
	TwAddVarRW(m_pTwBar, "Use Antiradiance", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseAntiradiance), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Gather With AVPL Atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->GatherWithAVPLAtlas), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Gather With AVPL Clustering", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->GatherWithAVPLClustering), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Filter AVPL Atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->FilterAvplAtlasLinear), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Fill AVPL Atlas On GPU", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->FillAvplAltasOnGPU), " min=0 max=1 step=1 ");
	
	TwAddSeparator(m_pTwBar, "", "");
	
	TwAddVarRW(m_pTwBar, "#Paths", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumPaths), " min=1 max=10000000000 step=1 ");
	TwAddVarRW(m_pTwBar, "#Paths per frame", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumPathsPerFrame), " min=1 max=16000 step=1 ");
	TwAddVarRW(m_pTwBar, "Cone Factor", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ConeFactor), " min=1 max=1000 step=1 ");
	TwAddVarRW(m_pTwBar, "Antirad Filter Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->AntiradFilterMode), " min=0 max=2 step=1 ");
	TwAddVarRW(m_pTwBar, "Antirad Filter Gauss Factor", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AntiradFilterGaussFactor), " min=1.0 max=4.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "#add. AVPL", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumAdditionalAVPLs), " min=0 max=2048 step=1 ");
	
	TwAddSeparator(m_pTwBar, "", "");
	
	TwAddVarRW(m_pTwBar, "Draw cut sizes", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawCutSizes), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw AVPL atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawAVPLAtlas), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw AVPL cluster atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawAVPLClusterAtlas), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Use Debug Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseDebugMode), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw LOL", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawLightingOfLight), " min=-1 max=100 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Lights", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawLights), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "RenderBounce", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->RenderBounce), " min=-1 max=100 step=1 ");
	TwAddVarRW(m_pTwBar, "Geo-Term Limit", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->GeoTermLimit), "min=0 max=100000 step=0.00001");
	TwAddVarRW(m_pTwBar, "Clamp Geo-Term", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ClampGeoTerm), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Sqrt # Atlas Samles", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumSqrtAtlasSamples), " min=1 max=100 step=1 ");

	TwAddSeparator(m_pTwBar, "", "");

	TwAddVarRW(m_pTwBar, "Use Tone Mapping", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseToneMapping), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Gamma", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->Gamma), " min=0.0 max=100.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "Exposure", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->Exposure), " min=0.0 max=100.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "Isect. BFC", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->Intersection_BFC), " min=0 max=1 step=1 ");

	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "Texel Offset X", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->TexelOffsetX), " min=0.0 max=10.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "Texel Offset Y", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->TexelOffsetY), " min=0.0 max=10.0 step=0.1 ");

	TwAddVarRW(m_pTwBar, "Cluster/Light tree", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseLightTree), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Light Tree Cut Depth", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->LightTreeCutDepth), " min=-1 max=64 step=1 ");
	TwAddVarRW(m_pTwBar, "Vis. Cluster Depth", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ClusterDepth), " min=0 max=1000 step=1 ");
	TwAddVarRW(m_pTwBar, "Cluster Method", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ClusterMethod), " min=0 max=10 step=1 ");
	TwAddVarRW(m_pTwBar, "Cluster Weight Normals", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ClusterWeightNormals), "min=0 max=1000 step=1");
	TwAddVarRW(m_pTwBar, "Cluster Refinement Threshold", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ClusterRefinementThreshold), "min=0.0 max=1.0 step=0.00000001");

	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "Sep D/I Light", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->SeparateDirectIndirectLighting), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Direct Light", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawDirectLight), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Indirect Light", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawIndirectLight), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "#VPLs DL", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumVPLsDirectLight), " min=0 max=10000 step=1 ");
	TwAddVarRW(m_pTwBar, "#VPLs DL Per Frame", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumVPLsDirectLightPerFrame), " min=0 max=10000 step=1 ");

	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "AreaLight Front", TW_TYPE_DIR3F, &(m_pConfigManager->GetConfVarsGUI()->AreaLightFrontDirection), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Pos X", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightPosX), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Pos Y", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightPosY), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Pos Z", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightPosZ), " ");
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