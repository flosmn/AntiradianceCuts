#include "CClusterTree.h"

#include <iostream>
#include <map>
#include <algorithm>

#include "AVPL.h"
#include "CTimer.h"

#include "Utils\Rand.h"

const float SQRT_3 = std::sqrtf(3.f);



struct C
{
	float sim;
	int index;
};

CClusterTree::CClusterTree()
{
	m_Head = 0;
	m_pDataPoints = 0;

	m_NumColors = 200;
	InitColors();
}

CClusterTree::~CClusterTree()
{
	delete [] m_pColors;
}

void CClusterTree::BuildTree(std::vector<AVPL*>& avpls)
{
	CTimer timer(CTimer::CPU);
	timer.Start();

	m_NumDataPoints = (int)avpls.size();
	m_pDataPoints = new DATA_POINT[m_NumDataPoints];

	for(uint i = 0; i < avpls.size(); ++i)
	{
		DATA_POINT data_point;
		data_point.key = i;
		data_point.position = avpls[i]->GetPosition();
		data_point.normal = avpls[i]->GetOrientation();
		m_pDataPoints[i] = data_point;
	}

	//Get BB of and normalize
	glm::vec3 min = glm::vec3(std::numeric_limits<float>::max());
	glm::vec3 max = glm::vec3(std::numeric_limits<float>::min());
	for(int i = 0; i < m_NumDataPoints; ++i)
	{
		min = glm::min(m_pDataPoints[i].position, min);
		max = glm::max(m_pDataPoints[i].position, max);
	}
	glm::vec3 norm = glm::vec3(
		min.x == max.x ? 1.f : 1.f / (max.x - min.x),
		min.y == max.y ? 1.f : 1.f / (max.y - min.y),
		min.z == max.z ? 1.f : 1.f / (max.z - min.z));
	for(int i = 0; i < m_NumDataPoints; ++i)
	{
		m_pDataPoints[i].position = norm * (m_pDataPoints[i].position - min);
	}

	BuildTree(m_pDataPoints, m_NumDataPoints);

	timer.Stop();
	std::cout << "Clustering " << m_NumDataPoints << " avpls took " << timer.GetTime() << "ms." << std::endl;
}

void CClusterTree::Release()
{
	if(m_pDataPoints)
		delete [] m_pDataPoints;

	Release(GetHead());
}

void CClusterTree::Release(Node* node)
{
	if(!node) return;

	if(IsLeaf(node))
	{
		delete node;
		node = 0;
	}
	else
	{
		Release(node->leftChild);
		Release(node->rightChild);
	}
}

void CClusterTree::BuildTree(DATA_POINT* data_points,
	int num_data_points)
{
	std::map<int, Node*> nodes;

	C dummy;
	dummy.index = -1;
	dummy.sim = -1.f;

	C** c = new C*[num_data_points];
	
	for(int i = 0; i <num_data_points; ++i)
	{
		c[i] = new C[num_data_points];
	}
	
	int* I = new int[num_data_points];
	C* NBM = new C[num_data_points];

	int nodeId = 0;
	for(int n = 0; n < num_data_points; ++n)
	{
		Node* node = new Node;
		node->kvp_index = n;
		node->leftChild = 0;
		node->rightChild = 0;
		node->id = nodeId++;
		nodes[n] = node;
	}
	
	for(int n = 0; n < num_data_points; ++n)
	{
		for(int i = 0; i < num_data_points; ++i)
		{
			c[n][i].sim = SIM(data_points[n], data_points[i]);
			c[n][i].index = i;
		}
		I[n] = n;
		
		C argmax;
		float maxval = -1.f;
		bool initialized = false;
		for(int i = 0; i < num_data_points; ++i)
		{
			if(i==n) continue;
			if(c[n][i].sim > maxval)
			{
				maxval = c[n][i].sim;
				argmax = c[n][i];
				initialized = true;
			}
		}
		NBM[n] = initialized ? argmax : dummy;
	}

	for(int n = 0; n < num_data_points-1; ++n)
	{
		int i1;
		{
			int argmax;
			float maxval = -1.f;
			bool initialized = false;
			for(int i = 0; i < num_data_points; ++i)
			{
				if(I[i]!=i) continue;
				if(NBM[i].sim > maxval)
				{
					maxval = NBM[i].sim;
					argmax = i;
					initialized = true;
				}
			}

			i1 = initialized ? argmax : 0;
		}		

		int i2 = I[NBM[i1].index];

		m_Head = new Node;
		m_Head->leftChild = nodes[i1];
		m_Head->rightChild = nodes[i2];
		m_Head->kvp_index = -1;
		m_Head->id = nodeId++;
		nodes[i2] = 0;
		nodes[i1] = m_Head;

		for(int i = 0; i < num_data_points; ++i)
		{
			if(I[i] == i && i != i1 && i != i2)
			{
				const float max = std::max(c[i1][i].sim, c[i2][i].sim);
				c[i1][i].sim = max;
				c[i][i1].sim = max;
			}
			
			if(I[i] == i2) 
				I[i] = i1;

			{
				C argmax;
				float maxval = -1.f;
				bool initialized = false;
				for(int i = 0; i < num_data_points; ++i)
				{
					if(i == i1 || I[i]!=i) continue;
					if(c[i1][i].sim > maxval)
					{
						maxval = c[i1][i].sim;
						argmax = c[i1][i];
						initialized = true;
					}
				}
				
				NBM[i1] = initialized? argmax : dummy; 
			}	
		}
	}
}

void CClusterTree::Traverse(Node* node)
{
	bool leafNode = (node->leftChild==0 && node->rightChild == 0);
	std::cout << "Node id: " << node->id << " is leaf: " << leafNode << std::endl;
	if(leafNode)
	{
		std::cout << "key-val-pair-index: " << node->kvp_index << std::endl;
	}
	else
	{
		std::cout << "leftNode id: " << node->leftChild->id << std::endl;
		std::cout << "rightNode id: " << node->rightChild->id << std::endl;

		Traverse(node->leftChild);
		Traverse(node->rightChild);
	}
}
	
Node* CClusterTree::GetHead()
{
	return m_Head;
}

__inline float map01(float x, float min, float max)
{
	return (x - min) / (max - min);
}

float CClusterTree::SIM(const DATA_POINT& p1, const DATA_POINT& p2)
{
	const float dist = glm::length(p1.position - p2.position);
	const float w1 = map01(1.f / (1.f + dist), 1.f / (1.f + SQRT_3) , 1.f);
	const float w2 = 0.5f * (glm::dot(p1.normal, p2.normal) + 1.f);

	return 0.5f * (w1 + w2);
}

void CClusterTree::ColorAVPLs(std::vector<AVPL*>& avpls, int depth)
{
	ColorNodes(avpls, 0, GetHead(), depth, 0);
}

void CClusterTree::ColorNodes(std::vector<AVPL*>& avpls, int level, Node* node, int depth, int colorIndex)
{
	if (level == depth)
	{
		std::vector<Node*> leafs;
		GetAllLeafs(node, leafs);
		for (size_t i = 0; i < leafs.size(); ++i)
		{
			avpls[leafs[i]->kvp_index]->SetColor(m_pColors[colorIndex]);
		}
		leafs.clear();
	}
	else
	{
		if (IsLeaf(node)) return;

		ColorNodes(avpls, level + 1, node->leftChild, depth, int(Rand01() * (m_NumColors - 1)) );
		ColorNodes(avpls, level + 1, node->rightChild, depth, int(Rand01() * (m_NumColors - 1)) );
	}
}

 void CClusterTree::GetAllLeafs(Node* node, std::vector<Node*>& leafs)
{
	if(IsLeaf(node))
	{
		leafs.push_back(node);
	}
	else
	{
		GetAllLeafs(node->leftChild, leafs);
		GetAllLeafs(node->rightChild, leafs);
	}
}

 bool CClusterTree::IsLeaf(Node* node)
 {
	 if(!node) return false;

	 if(node->leftChild != 0 || node->rightChild != 0)
		 return false;
	 return true;
 }

 glm::vec3 CClusterTree::GetRandomColor()
 {
	 glm::vec3 color = glm::vec3(Rand01(), Rand01(), Rand01());
	 return color;
 }

void CClusterTree::InitColors()
{
	m_pColors = new glm::vec3[m_NumColors];

	for(int i = 0; i < m_NumColors; ++i)
	{
		m_pColors[i] = GetRandomColor();
	}
}
