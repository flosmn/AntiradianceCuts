#ifndef _OCTAHEDRON_UTIL_H_
#define _OCTAHEDRON_UTIL_H_

#include <glm/glm.hpp>

glm::vec2 GetTexCoordForDir(glm::vec3 dir) {
	// Project from sphere onto octahedron
	dir /= glm::dot(glm::vec3(1.0f), glm::abs(dir));
	
	// If on lower hemisphere...
	if (dir.y < 0.0f) {
		// ...unfold
		float x = (1.0f - abs(dir.z)) * glm::sign(dir.x);
		float z = (1.0f - abs(dir.x)) * glm::sign(dir.z);
		dir.x = x;
		dir.z = z;
	}

	// Map from [0,1]^2 to [-1,1]^2
	dir.x = dir.x * 0.5f + 0.5f;
	dir.z = dir.z * 0.5f + 0.5f;
	
	return glm::vec2(dir.x, dir.z);
}

glm::vec3 GetDirForTexCoord(glm::vec2 texCoord) {
	glm::vec3 dir;

	dir.x = texCoord.x;
	dir.z = texCoord.y;
	
	// [-1,1]^2 -> [0,1]^2
	dir.x *= 2.0f;
	dir.z *= 2.0f;
	dir.x -= 1.0f;
	dir.z -= 1.0f;

	glm::vec3 vAbs = glm::abs(dir);
	// If on lower hemisphere...
	if (vAbs.x + vAbs.z > 1.0f) {
		// ...fold
		float x = glm::sign(dir.x) * (1.0f - vAbs.z);
		float z = glm::sign(dir.z) * (1.0f - vAbs.x);
		dir.x = x;
		dir.z = z;
	}

	// Elevate height
	dir.y = 1.0f - vAbs.x - vAbs.z;

	// Project onto sphere
	dir = glm::normalize(dir);

	return dir;
}

#endif
