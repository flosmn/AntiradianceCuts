#ifndef _C_AVPL_IMPORTANCE_SAMPLING_H_
#define _C_AVPL_IMPORTANCE_SAMPLING_H_

#include <glm/glm.hpp>

class Scene;
class COGLTexture2D;
class CConfigManager;
class AVPL;

#include "Structs.h"

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

	bool EvaluateAVPLImportance(const AVPL& avpl, float* scale);
	bool EvaluateAVPLAntiintensityImportance(const AVPL& avpl, float* scale);

	const std::vector<SceneSample>& GetSceneSamples() const { return m_SceneSamples; }

private:
	
	uint m_NumSceneSamples;
	float m_OneOverNumSceneSamples;
	std::vector<SceneSample> m_SceneSamples;

	Scene* m_pScene;
	CConfigManager* m_pConfManager;

	float m_AvgIrradiance;		// average luminance irradiance per pixel
	float m_AvgAntiirradiance;  // average luminance antiiradiance per pixel

	float m_Alpha;
	float m_Epsilon;
	float m_ConeFactor;
};

#endif // _C_AVPL_IMPORTANCE_SAMPLING_H_