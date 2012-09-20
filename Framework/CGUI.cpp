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
		
	TwAddVarRW(m_pTwBar, "Draw Error", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawError), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Reference", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawReference), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Use Antiradiance", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseAntiradiance), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Gather With AVPL Atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->GatherWithAVPLAtlas), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Gather With AVPL Clustering", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->GatherWithAVPLClustering), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Filter AVPL Atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->FilterAvplAtlasLinear), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Fill AVPL Atlas On GPU", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->FillAvplAltasOnGPU), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Sep D/I Lighting", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->SeparateDirectIndirectLighting), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Lighting Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->LightingMode), " min=0 max=2 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Direct Lighting", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawDirectLighting), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Draw Indirect Lighting", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawIndirectLighting), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "#AVPLs per frame", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumAVPLsPerFrame), " min=1 max=100000 step=1 ");
	TwAddVarRW(m_pTwBar, "#VPLs DL", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumVPLsDirectLight), " min=0 max=10000 step=1 ");
	TwAddVarRW(m_pTwBar, "#VPLs DL Per Frame", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumVPLsDirectLightPerFrame), " min=0 max=10000 step=1 ");
	TwAddVarRW(m_pTwBar, "#NoAntiradiance", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NoAntiradiance), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Cluster Refinement Threshold", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ClusterRefinementThreshold), "min=0.0 max=10.0 step=0.05");
	TwAddVarRW(m_pTwBar, "Cluster Refinement Max Radiance", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ClusterRefinementMaxRadiance), "min=0.0 max=10000.0 step=1.f");
	TwAddVarRW(m_pTwBar, "Cluster Refinement Weight", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ClusterRefinementWeight), "min=0.0 max=1.f step=0.05f");

	TwAddSeparator(m_pTwBar, "", "");
	
	
	TwAddVarRW(m_pTwBar, "#AVPLs debug", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumAVPLsDebug), " min=1 max=10000 step=1 ");
	TwAddVarRW(m_pTwBar, "Cone Factor", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->ConeFactor), " min=0.1 max=1000.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "Antirad Filter Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->AntiradFilterMode), " min=0 max=2 step=1 ");
	TwAddVarRW(m_pTwBar, "Antirad Filter Gauss Factor", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AntiradFilterGaussFactor), " min=1.0 max=4.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "#add. AVPL", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumAdditionalAVPLs), " min=0 max=2048 step=1 ");
	
	TwAddSeparator(m_pTwBar, "", "");
	
	TwAddVarRW(m_pTwBar, "Draw cut sizes", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawCutSizes), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw AVPL atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawAVPLAtlas), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw AVPL cluster atlas", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawAVPLClusterAtlas), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Use Debug Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseDebugMode), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Draw Lights", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawLights), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Limit Bounces", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->LimitBounces), " min=-1 max=10 step=1 ");
	TwAddVarRW(m_pTwBar, "RenderBounce", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->RenderBounce), " min=-1 max=100 step=1 ");
	TwAddVarRW(m_pTwBar, "Clamp Radiance", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->GeoTermLimitRadiance), "min=0 max=100000 step=0.00001");
	TwAddVarRW(m_pTwBar, "Clamp Antiradiance", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->GeoTermLimitAntiradiance), "min=0 max=100000 step=0.00001");
	TwAddVarRW(m_pTwBar, "Clamp Geo-Term", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ClampGeoTerm), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Clamp Cone Mode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ClampConeMode), "min=0 max=2 step=1");
	TwAddVarRW(m_pTwBar, "Sqrt # Atlas Samles", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumSqrtAtlasSamples), " min=1 max=100 step=1 ");

	TwAddSeparator(m_pTwBar, "", "");

	TwAddVarRW(m_pTwBar, "Use Tone Mapping", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseToneMapping), "min=0 max=1 step=1");
	TwAddVarRW(m_pTwBar, "Gamma", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->Gamma), " min=0.0 max=100.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "Exposure", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->Exposure), " min=0.0 max=100.0 step=0.1 ");
	TwAddVarRW(m_pTwBar, "Isect. BFC", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->Intersection_BFC), " min=0 max=1 step=1 ");

	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "DisplacePCP", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->DisplacePCP), " min=0.0 max=100.0 step=0.01 ");

	TwAddVarRW(m_pTwBar, "Light Tree Cut Depth", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->LightTreeCutDepth), " min=-1 max=64 step=1 ");
	TwAddVarRW(m_pTwBar, "Vis. Cluster Depth", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ClusterDepth), " min=0 max=1000 step=1 ");
	TwAddVarRW(m_pTwBar, "Cluster Method", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ClusterMethod), " min=0 max=10 step=1 ");
	
	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "AreaLight Front", TW_TYPE_DIR3F, &(m_pConfigManager->GetConfVarsGUI()->AreaLightFrontDirection), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Pos X", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightPosX), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Pos Y", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightPosY), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Pos Z", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightPosZ), " ");
	TwAddVarRW(m_pTwBar, "AreaLight Radiance Scale", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AreaLightRadianceScale), "min=0.0 max=10000.0 step=1.0");

	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "UseAVPLImpSampling", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseAVPLImportanceSampling), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "ImportanceSamplingMode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ISMode), " min=0 max=2 step=1 ");
	TwAddVarRW(m_pTwBar, "ConeFactorIS", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->ConeFactorIS), " min=1 max=100 step=1 ");
	TwAddVarRW(m_pTwBar, "NumSceneSamples", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumSceneSamples), " min=1 max=10000 step=1 ");
	TwAddVarRW(m_pTwBar, "DrawSceneSamples", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawSceneSamples), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "DrawCollectedAVPLs", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawCollectedAVPLs), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "DrawCollectedISAVPLs", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawCollectedISAVPLs), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "AcceptProbabEpsilon", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->AcceptProbabEpsilon), " min=0.00 max=1.00 step=0.01");
	TwAddVarRW(m_pTwBar, "WeightIrradAntiirrad", TW_TYPE_FLOAT, &(m_pConfigManager->GetConfVarsGUI()->IrradAntiirradWeight), " min=0.0 max=1.0 step=0.1");
	
	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "UseBIDIR", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseBIDIR), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "UseStratification", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseStratification), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "DrawBIDIRSamples", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawBIDIRSamples), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "DrawBIDIRSamplesMode", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->DrawBIDIRSamplesMode), " min=0 max=3 step=1 ");
	TwAddVarRW(m_pTwBar, "NumEyeRaysSS", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumEyeRaysSS), " min=1 max=20000 step=1 ");
	TwAddVarRW(m_pTwBar, "NumSamplesForPESS", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumSamplesForPESS), " min=1 max=1000 step=1 ");
	TwAddVarRW(m_pTwBar, "NumEyeRaysASS", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumEyeRaysASS), " min=1 max=20000 step=1 ");
	TwAddVarRW(m_pTwBar, "NumSamplesForPEASS", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumSamplesForPEASS), " min=1 max=1000 step=1 ");
	
	TwAddSeparator(m_pTwBar, "", "");
	TwAddVarRW(m_pTwBar, "Use Path-Tracing", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UsePathTracing), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Use MIS", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->UseMIS), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Gaussian Blur", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->GaussianBlur), " min=0 max=1 step=1 ");
	TwAddVarRW(m_pTwBar, "Num Samples", TW_TYPE_INT32, &(m_pConfigManager->GetConfVarsGUI()->NumSamples), " min=0 max=50000 step=1 ");

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