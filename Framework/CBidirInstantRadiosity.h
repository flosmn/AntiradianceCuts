#ifndef _C_BI_DIR_INSTANT_RADIOSIY_H_
#define _C_BI_DIR_INSTANT_RADIOSIY_H_

#include "Intersection.h"
#include "AVPL.h"

#include <vector>

class CConfigManager;
class Scene;

class CBidirInstantRadiosity
{
public:
	CBidirInstantRadiosity(Scene* pScene, CConfigManager* pConfManager);
	~CBidirInstantRadiosity();

	void CreateSceneSamples(bool profile);
	void CreateAntiSceneSamples(bool profile);
	void CreatePaths(std::vector<AVPL>& avpls, int& numPaths, bool profile);
	void CreatePath(std::vector<AVPL>& avpls, bool profile);

	std::vector<SceneSample>& GetSceneSamples() { return m_SceneSamples; }
	std::vector<SceneSample>& GetAntiSceneSamples() { return m_AntiSceneSamples; }
	std::vector<SceneSample>& GetVisiblesSS() { return m_VisiblesSS; }
	std::vector<SceneSample>& GetVisiblesASS() { return m_VisiblesASS; }

private:
	void ConnectToSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale);
	void ConnectToAntiSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale);
	bool CreateAVPLAtSceneSample(const SceneSample& ss, const AVPL& pred, AVPL* newAVPL);
	bool CreateAVPLAtAntiSceneSample(const SceneSample& ss, const AVPL& pred, AVPL* newAVPL);
	bool Visible(const SceneSample& ss, const AVPL& avpl, CPrimitive::IsectMode isect_mode);
	bool Visible(const SceneSample& ss1, const SceneSample& ss2, CPrimitive::IsectMode isect_mode);
	void CreateVisibleSceneSamples(std::vector<SceneSample>& ss, int numSS);

	void CreateVisibles(std::vector<SceneSample>& sceneSamples, int numVisibles);

	std::vector<SceneSample> m_VisiblesSS;
	std::vector<SceneSample> m_VisiblesASS;
	std::vector<SceneSample> m_SceneSamples;		// scene samples to create imporant radiance carrying paths
	std::vector<SceneSample> m_AntiSceneSamples;	// scene samples to create imporant antiradiance carrying paths

	std::vector<SceneSample> m_Visibles;
	int m_NumVisibles;

	CConfigManager* m_pConfManager;
	Scene* m_pScene;
};

#endif // _C_BI_DIR_INSTANT_RADIOSIY_H_