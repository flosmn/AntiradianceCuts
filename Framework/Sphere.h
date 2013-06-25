#ifndef SPHERE_H_
#define SPHERE_H_

#include <glm/glm.hpp>

#include "Defines.h"
#include "Mesh.h"

#include <vector>
#include <memory>

class Sphere 
{
	public:
		// resolution: number of vertices on equator
		// number of vertices total will be resolution^2
		Sphere(const int resolution)
		{
			// just the positions. not ordered as trianglmes
			std::vector<glm::vec3> positionData;

			std::vector<glm::vec3> positions;
			std::vector<glm::vec3> normals;
			std::vector<glm::uvec3> indices;

			for (int j = 0; j < resolution; ++j)
			{
				const float theta = M_PI * float(j) / resolution;
				for (int i = 0; i < resolution; ++i)
				{
					const float phi = 2.f * M_PI * float(i) / resolution;
					positionData.push_back(glm::vec3(
						std::sin(theta)*std::cos(phi),
						std::cos(theta),
						std::sin(theta)*std::sin(phi)));
				}
			}

			int indicesCounter = 0;
			for (int j = 0; j < resolution; ++j)
			{
				for (int i = 0; i < resolution; ++i)
				{
					const int next = (i == resolution - 1) ? 0 : i + 1;
					const bool southpole = (j == resolution - 1);

					glm::vec3 v1, v2, v3, v4;
					v1 = positionData[j * resolution + i];
					v2 = positionData[j * resolution + next];
					
					if (southpole || (j+1) * resolution + next >= positionData.size()) {
						v3 = glm::vec3(0.0f, -1.0f, 0.0f);
					} else {
						v3 = positionData[(j+1) * resolution + next];
						v4 = positionData[(j+1) * resolution + i];
					}

					positions.push_back(v1);
					positions.push_back(v2);
					positions.push_back(v3);
					normals.push_back(glm::normalize(v1));
					normals.push_back(glm::normalize(v2));
					normals.push_back(glm::normalize(v3));
					indices.push_back(glm::uvec3(
						indicesCounter + 0,
						indicesCounter + 1,
						indicesCounter + 2));
					indicesCounter += 3;

					if (!southpole)
					{
						positions.push_back(v3);
						positions.push_back(v4);
						positions.push_back(v1);
						normals.push_back(glm::normalize(v3));
						normals.push_back(glm::normalize(v4));
						normals.push_back(glm::normalize(v1));
						indices.push_back(glm::uvec3(
							indicesCounter + 0,
							indicesCounter + 1,
							indicesCounter + 2));
						indicesCounter += 3;
					}
				}
			}

			m_mesh.reset(new Mesh(positions, normals, indices));
		}

		Mesh* getMesh() { return m_mesh.get(); }
	private:
		std::unique_ptr<Mesh> m_mesh;
};

#endif //  SPHERE_H_
