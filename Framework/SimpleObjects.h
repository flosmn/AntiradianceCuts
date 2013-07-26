#ifndef SIMPLEOBJECTS_H_
#define SIMPLEOBJECTS_H_

#include <glm/glm.hpp>

#include "Mesh.hpp"
#include "Defines.h"

#include <vector>
#include <memory>

class SimpleObject
{
public:
	SimpleObject() {}
	virtual ~SimpleObject() {}

	Mesh* getMesh() { return m_mesh.get(); }
	void setTransform(glm::mat4 const& mat) { m_transform = mat; }
	glm::mat4 const& getTransform() const { return m_transform; }
protected:
	std::unique_ptr<Mesh> m_mesh;
	glm::mat4 m_transform;
};

class Cube : public SimpleObject
{
public:
	Cube() : SimpleObject()
	{
		std::vector<glm::vec3> positions;
		// face 1
		positions.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
		positions.push_back(glm::vec3(0.0f, 1.0f, 1.0f));
		positions.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
		positions.push_back(glm::vec3(1.0f, 0.0f, 1.0f));
		// face 2 
		positions.push_back(glm::vec3(1.0f, 0.0f, 1.0f));
		positions.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
		positions.push_back(glm::vec3(1.0f, 1.0f, 0.0f));
		positions.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
		// face 3
		positions.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
		positions.push_back(glm::vec3(1.0f, 1.0f, 0.0f));
		positions.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
		positions.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
		// face 4
		positions.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
		positions.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
		positions.push_back(glm::vec3(0.0f, 1.0f, 1.0f));
		positions.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
		// face 5
		positions.push_back(glm::vec3(0.0f, 1.0f, 1.0f));
		positions.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
		positions.push_back(glm::vec3(1.0f, 1.0f, 0.0f));
		positions.push_back(glm::vec3(1.0f, 1.0f, 1.0f));
		// face 6
		positions.push_back(glm::vec3(1.0f, 0.0f, 1.0f));
		positions.push_back(glm::vec3(1.0f, 0.0f, 0.0f));
		positions.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
		positions.push_back(glm::vec3(0.0f, 0.0f, 1.0f));

		std::vector<glm::vec3> normals;
		// normals face 1
		normals.push_back(glm::vec3( 0.0f,  0.0f,  1.0f));
		normals.push_back(glm::vec3( 0.0f,  0.0f,  1.0f));
		normals.push_back(glm::vec3( 0.0f,  0.0f,  1.0f));
		normals.push_back(glm::vec3( 0.0f,  0.0f,  1.0f));
		// normals face 2
		normals.push_back(glm::vec3( 1.0f,  0.0f,  0.0f));
		normals.push_back(glm::vec3( 1.0f,  0.0f,  0.0f));
		normals.push_back(glm::vec3( 1.0f,  0.0f,  0.0f));
		normals.push_back(glm::vec3( 1.0f,  0.0f,  0.0f));
		// normals face 3
		normals.push_back(glm::vec3( 0.0f,  0.0f, -1.0f));
		normals.push_back(glm::vec3( 0.0f,  0.0f, -1.0f));
		normals.push_back(glm::vec3( 0.0f,  0.0f, -1.0f));
		normals.push_back(glm::vec3( 0.0f,  0.0f, -1.0f));
		// normals face 4
		normals.push_back(glm::vec3(-1.0f,  0.0f,  0.0f));
		normals.push_back(glm::vec3(-1.0f,  0.0f,  0.0f));
		normals.push_back(glm::vec3(-1.0f,  0.0f,  0.0f));
		normals.push_back(glm::vec3(-1.0f,  0.0f,  0.0f));
		// normals face 5
		normals.push_back(glm::vec3( 0.0f,  1.0f,  0.0f));
		normals.push_back(glm::vec3( 0.0f,  1.0f,  0.0f));
		normals.push_back(glm::vec3( 0.0f,  1.0f,  0.0f));
		normals.push_back(glm::vec3( 0.0f,  1.0f,  0.0f));
		// normals face 6
		normals.push_back(glm::vec3( 0.0f, -1.0f,  0.0f));
		normals.push_back(glm::vec3( 0.0f, -1.0f,  0.0f));
		normals.push_back(glm::vec3( 0.0f, -1.0f,  0.0f));
		normals.push_back(glm::vec3( 0.0f, -1.0f,  0.0f));

		std::vector<glm::uvec3> indices;
		for (int i = 0, k = 0; i < 6; ++i, k+=4)
		{
			// face i
			indices.push_back(glm::uvec3(k, k+1, k+2));
			indices.push_back(glm::uvec3(k, k+2, k+3));
		}

		m_mesh.reset(new Mesh(positions, normals, indices));
	}

	~Cube()
	{
	}
};

class AABBox : public SimpleObject
{
public:
	AABBox(glm::vec3 min, glm::vec3 max) : SimpleObject()
	{
		Cube cube;

		std::vector<glm::vec3> positions;
		for (int i = 0; i < cube.getMesh()->getPositions().size(); ++i) {
			positions.push_back((max - min) * cube.getMesh()->getPositions()[i] + min);
		}
	}
};

class Sphere : public SimpleObject
{
public:
	// resolution: number of vertices on equator
	// number of vertices total will be resolution^2
	Sphere(const int resolution) : SimpleObject()
	{
		// just the positions. not ordered as trianglmes
		std::vector<glm::vec3> positionData;

		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> normals;
		std::vector<glm::uvec3> indices;

		for (int j = 0; j < resolution; ++j)
		{
			const float theta = PI * float(j) / resolution;
			for (int i = 0; i < resolution; ++i)
			{
				const float phi = 2.f * PI * float(i) / resolution;
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
};

#endif // SIMPLEOBJECTS_H_
