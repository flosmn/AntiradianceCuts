#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include "GL/glew.h"

class CFullScreenQuad;

class Postprocessing
{
public:
	Postprocessing();
	~Postprocessing();

	bool Init();
	void Release();

	void Postprocess(GLuint textureSource, GLuint width, GLuint height, int numberOfLightPaths);

	void SetExposure(float _exposure) { exposure = _exposure; }
	void SetGamma(float _gamma) { gamma = _gamma; } 
private:
	GLuint postProcessingProgram;
	GLuint uniformNumberOfLightPaths;
	GLuint uniformExposure;
	GLuint uniformGamma;
	GLuint uniformSampler;
	GLuint uniformWidth;
	GLuint uniformHeight;
	
	GLuint m_Sampler;

	CFullScreenQuad* m_pFullScreenQuad;

	float gamma;
	float exposure;
};

#endif