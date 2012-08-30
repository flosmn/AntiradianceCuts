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
	enum SceneSampleType { SS, ASS };

	CBidirInstantRadiosity(Scene* pScene, CConfigManager* pConfManager);
	~CBidirInstantRadiosity();

	void CreateSceneSamples(bool profile, SceneSampleType ss_type);
	void CreatePaths(std::vector<AVPL>& avpls, int& numPaths, bool profile);
	void CreatePath(std::vector<AVPL>& avpls, bool profile);

	std::vector<SceneSample>& GetSceneSamples() { return m_SceneSamples; }
	std::vector<SceneSample>& GetAntiSceneSamples() { return m_AntiSceneSamples; }
	std::vector<SceneSample>& GetVisibles() { return m_Visibles; }
	
private:
	void ConnectToSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale);
	void ConnectToAntiSceneSamples(AVPL& avpl, std::vector<AVPL>& avpls, float scale);
	bool CreateAVPLAtSceneSample(const SceneSample& ss, const AVPL& pred, AVPL* newAVPL, SceneSampleType ss_type);
	bool Visible(const AVPL& from, const SceneSample& to, SceneSampleType ss_type);
	bool Visible(const SceneSample& from, const SceneSample& to_param, SceneSampleType ss_type);

	void CreateVisibles(std::vector<SceneSample>& sceneSamples, int numVisibles);
	float Probability(const SceneSample& from, const SceneSample& to, SceneSampleType ss_type);

	std::vector<SceneSample> m_SceneSamples;		// scene samples to create imporant radiance carrying paths
	std::vector<SceneSample> m_AntiSceneSamples;	// scene samples to create imporant antiradiance carrying paths

	std::vector<SceneSample> m_Visibles;
	int m_NumVisibles;

	CConfigManager* m_pConfManager;
	Scene* m_pScene;
};

#endif // _C_BI_DIR_INSTANT_RADIOSIY_H_