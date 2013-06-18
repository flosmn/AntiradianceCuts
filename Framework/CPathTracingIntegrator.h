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
	CPathTracingIntegrator(Scene* pScene, CImagePlane* pImagePlane);
	~CPathTracingIntegrator();

	void Integrate(uint numPaths, bool MIS);

private:
	Scene* m_scene;
	CImagePlane* m_imagePlane;
};

#endif // _C_PATH_TRACING_INTEGRATOR_H_
