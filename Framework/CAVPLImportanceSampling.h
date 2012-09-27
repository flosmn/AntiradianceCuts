#ifndef _C_AVPL_IMPORTANCE_SAMPLING_H_
#define _C_AVPL_IMPORTANCE_SAMPLING_H_

#include <glm/glm.hpp>

class Scene;
class COGLTexture2D;
class CConfigManager;
class CMaterialBuffer;
class AVPL;

#include "Intersection.h"

#include <vector>

typedef unsigned int uint;

class CAVPLImportanceSampling
{
public:
	CAVPLImportanceSampling(Scene* pScene, CConfigManager* pConfManager);
	~CAVPLImportanceSampling();

	void SetNumberOfSceneSamples(uint num);

	void UpdateCurrentIrradiance(COGLTexture2D* pTexture);
	void UpdateCurrentAntiirradiance(COGLTexture2D* pTexture);
	void CreateSceneSamples();

	bool EvaluateAVPLImportance0(AVPL& avpl, float* scale);
	bool EvaluateAVPLImportance1(AVPL& avpl, float* scale);	
	bool EvaluateAVPLImportance(AVPL& avpl, float* scale);	
	
	const std::vector<SceneSample>& GetSceneSamples() const { return m_SceneSamples; }

	void ImportanceSampling(const std::vector<AVPL>& avpls, std::vector<AVPL>& result);

	bool HasAntiradianceContribution(const AVPL& avpl);

private:

	glm::vec3 f(const AVPL& avpl, const SceneSample& ss);
	glm::vec3 f_light(const AVPL& avpl, const SceneSample& ss);
	
	uint m_NumSceneSamples;
	float m_OneOverNumSceneSamples;
	std::vector<SceneSample> m_SceneSamples;

	Scene* m_pScene;
	CConfigManager* m_pConfManager;

	float m_AvgIrradiance;		// average luminance irradiance per pixel
	float m_AvgAntiirradiance;  // average luminance antiiradiance per pixel

	float m_RadianceContrib;
	int m_NumContribSamples;

	float m_Alpha;
	float m_Epsilon;
	float m_ConeFactor;
};

#endif // _C_AVPL_IMPORTANCE_SAMPLING_H_