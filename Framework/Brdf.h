#ifndef BRDF_H_
#define BRDF_H_

#include "Util.h"
#include "Material.h"
#include "Defines.h"
#include "Sampler.h"

glm::vec3 phong_eval(glm::vec3 const& w_i, glm::vec3 const& w_o, glm::vec3 const& n, MATERIAL* mat) {
	if (glm::dot(w_i, n) <= 0.f) return glm::vec3(0.f);
	if (glm::dot(w_o, n) <= 0.f) return glm::vec3(0.f);

	glm::vec3 res(0.f);
	if (glm::length(mat->specular) > 0.f) {
		const glm::vec3 r = reflect(w_i, n);
		const float cos_theta = glm::max(0.f, glm::dot(r, w_o));
		res += ONE_OVER_TWO_PI * (mat->exponent+2.f) * powf(cos_theta, mat->exponent) * mat->specular;
	}
	res += ONE_OVER_PI * mat->diffuse;
	return res;
}

float phong_pdf(glm::vec3 const& w_i, glm::vec3 const& w_o, glm::vec3 const& n, MATERIAL* mat) {
	if (glm::dot(w_i, n) <= 0.f) return 0.f;
	if (glm::dot(w_o, n) <= 0.f) return 0.f;
	
	const bool hasDiffuse = glm::length(mat->diffuse) > 0.f;
	const bool hasSpecular = glm::length(mat->specular) > 0.f;

	float diffuseProb = 0.f;
	float specularProb = 0.f;

	if (hasDiffuse) {
		diffuseProb += ONE_OVER_PI * glm::dot(n, w_o);
	}
	if (hasSpecular) {
		const glm::vec3 r = reflect(w_i, n);
		const float cos_theta = glm::max(0.f, glm::dot(r, w_o));
		specularProb += ONE_OVER_TWO_PI * (mat->exponent+1.f) * powf(cos_theta, mat->exponent);
	}

	if (hasDiffuse && hasSpecular) {
		const float alpha = luminance(mat->specular) / (luminance(mat->specular) + luminance(mat->diffuse));
		return alpha * specularProb + (1.f-alpha) * diffuseProb;
	} else if (hasDiffuse) {
		return diffuseProb;
	} else if (hasSpecular) {
		return specularProb;
	}

	return 0.f;
}

glm::vec3 phong_sample(glm::vec3 const& w_i, glm::vec3 const& n, MATERIAL* mat, glm::vec2 const& sample_, float& pdf) {
	glm::vec2 sample = sample_;

	const bool hasDiffuse = glm::length(mat->diffuse) > 0.f;
	const bool hasSpecular = glm::length(mat->specular) > 0.f;

	bool choseSpecular = hasSpecular;

	if (hasDiffuse && hasSpecular) {
		const float alpha = luminance(mat->specular) / (luminance(mat->specular) + luminance(mat->diffuse));
		if (sample.x <= alpha) {
			sample.x /= alpha;
		} else {
			sample.x = (sample.x - alpha) / (1.f - alpha);
			choseSpecular = false;
		}
	}

	if (choseSpecular) {
		const glm::vec3 r = reflect(w_i, n);
		const glm::vec3 dir = sampleCosCone(w_i, sample, pdf, mat->exponent);
		if (glm::dot(dir, n) <= 0.f) {
			return glm::vec3(0.f);
		}
	} 
	
	return sampleCosCone(n, sample, pdf, 1);
}

#endif BRDF_H_