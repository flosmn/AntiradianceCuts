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

	bool EvaluateAVPLImportance(AVPL* avpl, float* scale);
	bool EvaluateAVPLAntiintensityImportance(AVPL* avpl, float* scale);

private:
	
	uint m_NumSceneSamples;
	std::vector<SceneSample> m_SceneSamples;

	Scene* m_pScene;
	CConfigManager* m_pConfManager;

	glm::vec3 m_AvgIrradiance;		// average irradiance per pixel
	glm::vec3 m_AvgAntiirradiance;  // average antiiradiance per pixel
};

#endif // _C_AVPL_IMPORTANCE_SAMPLING_H_