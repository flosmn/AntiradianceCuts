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
	void CreatePaths(std::vector<AVPL>& avpls, int& numCreatedPaths);
	void CreatePath(std::vector<AVPL>& avpls, int& numCreatedPaths);

	std::vector<SceneSample>& GetSceneSamples() { return m_SceneSamples; }
	std::vector<SceneSample>& GetAntiSceneSamples() { return m_AntiSceneSamples; }
	std::vector<SceneSample>& GetVisibles() { return m_Visibles; }

private:
	void ConnectToSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale);
	void ConnectToAntiSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale);
	bool CreateAVPLAtSceneSample(const SceneSample& ss, const AVPL& pred, AVPL* newAVPL);
	bool CreateAVPLAtAntiSceneSample(const SceneSample& ss, const AVPL& pred, AVPL* newAVPL);
	bool Visible(const SceneSample& ss, const AVPL& avpl, CPrimitive::IsectMode isect_mode);
	bool Visible(const SceneSample& ss1, const SceneSample& ss2, CPrimitive::IsectMode isect_mode);

	std::vector<SceneSample> m_Visibles;
	std::vector<SceneSample> m_SceneSamples;		// scene samples to create imporant radiance carrying paths
	std::vector<SceneSample> m_AntiSceneSamples;	// scene samples to create imporant antiradiance carrying paths

	CConfigManager* m_pConfManager;
	Scene* m_pScene;
};

#endif // _C_BI_DIR_INSTANT_RADIOSIY_H_