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
	TwAddVarRW(m_pTwBar, "Filter AVPL Atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->FilterAvplAtlasLinear), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Fill AVPL Atlas On GPU", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->FillAvplAltasOnGPU), " min=0 max=1 step=1 ");
	
	TwAddSeparator(m_pTwBar, "", "");
	
	TwAddVarRW(m_pTwBar, "#Paths", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumPaths), " min=1 max=10000000000 step=1 ");
	TwAddVarRW(m_pTwBar, "#Paths per frame", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumPathsPerFrame), " min=1 max=5000 step=1 ");
	TwAddVarRW(m_pTwBar, "Cone Factor", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ConeFactor), " min=1 max=1000 step=1 ");
	TwAddVarRW(m_pTwBar, "#add. AVPL", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumAdditionalAVPLs), " min=0 max=512 step=1 ");
	
	TwAddSeparator(m_pTwBar, "", "");
	
	TwAddVarRW(m_pTwBar, "Draw AVPL atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawAVPLAtlas), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Use Debug Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseDebugMode), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw LOL", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawLightingOfLight), " min=-1 max=100 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Lights", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawLights), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "RenderBounce", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->RenderBounce), " min=-1 max=100 step=1 ");
	TwAddVarRW(m_pTwBar, "Geo-Term Limit", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->GeoTermLimit), "min=0 max=100000 step=0.00001");
	TwAddVarRW(m_pTwBar, "Sqrt # Atlas Samles", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumSqrtAtlasSamples), " min=1 max=100 step=1 ");

	TwAddSeparator(m_pTwBar, "", "");

	TwAddVarRW(m_pTwBar, "Use Tone Mapping", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseToneMapping), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Gamma", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->Gamma), " min=0.0 max=10.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "Exposure", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->Exposure), " min=0.0 max=10.0 step=0.1 ");

	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "Texel Offset X", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->TexelOffsetX), " min=0.0 max=10.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "Texel Offset Y", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->TexelOffsetY), " min=0.0 max=10.0 step=0.1 ");

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

bool CGUI::HandleEvent(void* wnd, uint msg, uint wParam, uint lParam)
{
	if(TwEventWin(wnd, msg, wParam, lParam))
		return true;

	return false;
}