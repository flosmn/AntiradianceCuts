#ifndef SCENEPROBE_H_
#define SCENEPROBE_H_

#include "OGLResources/COGLUniformBuffer.h"
#include "OGLResources/COGLProgram.h"
#include "OGLResources/COGLBindLock.h"

#include "CProgram.h"
#include "CRenderTarget.h"
#include "CCamera.h"

#include "structs.h"

#include "Intersection.h"
#include "SimpleObjects.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <memory>
#include <iostream>

class SceneProbe
{
public:
	SceneProbe(glm::uvec2 const& pixel, Intersection const& intersection)
		: m_pixel(pixel), m_intersection(intersection)
	{
		m_sphere.reset(new Sphere(10));
		glm::mat4 T = glm::translate(glm::mat4(), m_intersection.getPosition());
		glm::mat4 S = glm::scale(glm::mat4(), glm::vec3(10.f));
		m_sphere->setTransform(T * S);

		std::cout << "place scene probe for pixel (" << pixel.x << ", " << pixel.y << ")" << std::endl;
	}

	void draw(CRenderTarget* target, CProgram* program, COGLUniformBuffer* ub, CCamera* camera)
	{
		CRenderTargetLock lock(target);
		COGLBindLock lockProgram(program->GetGLProgram(), COGL_PROGRAM_SLOT);
		
		TRANSFORM transform;
		transform.M = m_sphere->getTransform();
		transform.V = camera->GetViewMatrix();
		transform.itM = glm::inverse(transform.M);
		transform.MVP = camera->GetProjectionMatrix() * transform.V * transform.M; 
		ub->UpdateData(&transform);

		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		m_sphere->getMesh()->draw();
		glEnable(GL_CULL_FACE);
	}

	glm::uvec2 const& getPixel() { return m_pixel; }
private:
	glm::uvec2 m_pixel;
	Intersection m_intersection;
	std::unique_ptr<Sphere> m_sphere;
};

#endif // SCENEPROBE_H_
