#ifndef MODEL_HPP_
#define MODEL_HPP_

class Model
{
public:
	Model(Mesh* mesh, int material, glm::mat4 const& worldTransform) :
		m_mesh(mesh), m_material(material), m_worldTranform(worldTransform)
	{ }

	Mesh* getMesh() const { return m_mesh; }
	int getMaterial() const { return m_material; }
	glm::mat4 const& getWorldTransform() const { return m_worldTranform; }
private:
	Mesh* m_mesh;
	int m_material;
	glm::mat4 m_worldTranform;
};

#endif // MODEL_HPP_
