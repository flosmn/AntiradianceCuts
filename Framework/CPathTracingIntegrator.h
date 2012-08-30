#ifndef _C_PATH_TRACING_INTEGRATOR_H_
#define _C_PATH_TRACING_INTEGRATOR_H_

typedef unsigned int uint;

#include "Material.h"
#include "Intersection.h"

class Scene;
class CImagePlane;

class CPathTracingIntegrator
{
public:
	CPathTracingIntegrator();
	~CPathTracingIntegrator();

	bool Init(Scene* pScene, CImagePlane* pImagePlane);

	void Integrate(uint numPaths, bool MIS);

private:
	Scene* m_pScene;
	CImagePlane* m_pImagePlane;
};

#endif // _C_PATH_TRACING_INTEGRATOR_H_