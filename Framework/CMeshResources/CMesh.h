#ifndef _C_MESH_H_
#define _C_MESH_H_

typedef unsigned int uint;
typedef unsigned short ushort;

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#include "..\CUtils\Util.h"

#include "..\Triangle.h"

#include <vector>


class CMesh 
{
public:
	CMesh() {}
	virtual ~CMesh() {};
		
	uint GetNumberOfVertices() { return numberOfVertices; }
	uint GetNumberOfTriangles() { return numberOfTriangles; }
	const glm::vec4* GetVertexPositions() { return m_pVertexPositions; }
	const glm::vec3* GetVertexNormals() { return m_pVertexNormals; }
	const glm::vec3* GetVertexTexCoords() { return m_pVertexTexCoords; }
	const ushort* GetIndexData() { return m_pIndexData; }

	void CreateTriangleData(std::vector<Triangle*>& triangles)
	{
		std::vector<Triangle*>::iterator it;
		for ( it=triangles.begin() ; it < triangles.end(); it++ )
		{
			SAFE_DELETE(*it);
		}
		triangles.clear();
		
		for(uint i = 0; i < numberOfTriangles; i++)
		{
			uint i1 = m_pIndexData[i * 3 + 0];
			uint i2 = m_pIndexData[i * 3 + 1];
			uint i3 = m_pIndexData[i * 3 + 2];
			glm::vec3 p1 = glm::vec3(m_pVertexPositions[i1]);
			glm::vec3 p2 = glm::vec3(m_pVertexPositions[i2]);
			glm::vec3 p3 = glm::vec3(m_pVertexPositions[i3]);

			glm::vec3 normal = glm::normalize(glm::cross(p3-p1, p2-p1));

			Triangle* triangle = new Triangle(p1, p2, p3, normal);
			triangles.push_back(triangle);																	 
		}
	}
	
protected:
	glm::vec4* m_pVertexPositions;
	glm::vec3* m_pVertexNormals;
	glm::vec3* m_pVertexTexCoords;
	ushort* m_pIndexData;
	uint numberOfVertices;
	uint numberOfTriangles;

	std::vector<Triangle*> triangles;
};

class CFullScreenQuadMesh : public CMesh 
{
public:
	CFullScreenQuadMesh() 
	{
		numberOfVertices = 4;
		m_pVertexPositions = new glm::vec4[numberOfVertices];
		m_pVertexTexCoords = new glm::vec3[numberOfVertices];
		memset(m_pVertexPositions, 0 , numberOfVertices * sizeof(glm::vec4));
		memset(m_pVertexTexCoords, 0 , numberOfVertices * sizeof(glm::vec3));

		// positions
		m_pVertexPositions[0] = glm::vec4(-1.0f, -1.0f, 0.0f, 1.0f);
		m_pVertexPositions[1] = glm::vec4(-1.0f,  1.0f, 0.0f, 1.0f);
		m_pVertexPositions[2] = glm::vec4( 1.0f,  1.0f, 0.0f, 1.0f);
		m_pVertexPositions[3] = glm::vec4( 1.0f, -1.0f, 0.0f, 1.0f);

		// texture coordinates
		m_pVertexTexCoords[0] = glm::vec3(0.0f, 0.0f, 0.0f);
		m_pVertexTexCoords[1] = glm::vec3(0.0f, 1.0f, 0.0f);
		m_pVertexTexCoords[2] = glm::vec3(1.0f, 1.0f, 0.0f);
		m_pVertexTexCoords[3] = glm::vec3(1.0f, 0.0f, 0.0f);

		numberOfTriangles = 2;
		m_pIndexData = new ushort[3 * numberOfTriangles];
		m_pIndexData[0] = 0; m_pIndexData[1] = 1; m_pIndexData[2] = 2;
		m_pIndexData[3] = 0; m_pIndexData[4] = 2; m_pIndexData[5] = 3;
	}

	~CFullScreenQuadMesh() 
	{
		SAFE_DELETE_ARRAY(m_pIndexData);
		SAFE_DELETE_ARRAY(m_pVertexPositions); 
		SAFE_DELETE_ARRAY(m_pVertexTexCoords);
		
		std::vector<Triangle*>::iterator it;
		for(it=triangles.begin(); it < triangles.end(); ++it)
		{
			SAFE_DELETE(*it);
		}
		triangles.clear();
	}
};

class CQuadMesh : public CMesh 
{
public:
	CQuadMesh() 
	{
		numberOfVertices = 4;
		m_pVertexPositions = new glm::vec4[numberOfVertices];
		m_pVertexNormals = new glm::vec3[numberOfVertices];
		memset(m_pVertexPositions, 0 , numberOfVertices * sizeof(glm::vec4));
		memset(m_pVertexNormals, 0 , numberOfVertices * sizeof(glm::vec3));
		
		// positions
		m_pVertexPositions[0] = glm::vec4(-1.0f, -1.0f, 0.0f, 1.0f);
		m_pVertexPositions[1] = glm::vec4(-1.0f,  1.0f, 0.0f, 1.0f);
		m_pVertexPositions[2] = glm::vec4( 1.0f,  1.0f, 0.0f, 1.0f);
		m_pVertexPositions[3] = glm::vec4( 1.0f, -1.0f, 0.0f, 1.0f);
	
		// normals
		m_pVertexNormals[0] = glm::vec3( 0.0f,	0.0f,	1.0f);
		m_pVertexNormals[1] = glm::vec3( 0.0f,	0.0f,	1.0f);
		m_pVertexNormals[2] = glm::vec3( 0.0f,	0.0f,	1.0f);
		m_pVertexNormals[3] = glm::vec3( 0.0f,	0.0f,	1.0f);

		numberOfTriangles = 2;
		m_pIndexData = new ushort[3 * numberOfTriangles];
		m_pIndexData[0] = 0; m_pIndexData[1] = 1; m_pIndexData[2] = 2;
		m_pIndexData[3] = 0; m_pIndexData[4] = 2; m_pIndexData[5] = 3;
	}

	~CQuadMesh() 
	{
		SAFE_DELETE_ARRAY(m_pIndexData);
		SAFE_DELETE_ARRAY(m_pVertexPositions); 
		SAFE_DELETE_ARRAY(m_pVertexNormals); 

		std::vector<Triangle*>::iterator it;
		for(it=triangles.begin(); it < triangles.end(); ++it)
		{
			SAFE_DELETE(*it);
		}
		triangles.clear();
	}
};

class CCubeMesh : public CMesh 
{
public:
	CCubeMesh() 
	{
		numberOfVertices = 24;
		m_pVertexPositions = new glm::vec4[numberOfVertices];
		m_pVertexNormals = new glm::vec3[numberOfVertices];
		memset(m_pVertexPositions, 0 ,numberOfVertices * sizeof(glm::vec4));
		memset(m_pVertexNormals, 0 ,numberOfVertices * sizeof(glm::vec3));

		// face 1
		m_pVertexPositions[0] = glm::vec4(-1.0f, -1.0f, 1.0f, 1.0f);
		m_pVertexPositions[1] = glm::vec4(-1.0f,  1.0f, 1.0f, 1.0f);
		m_pVertexPositions[2] = glm::vec4( 1.0f,  1.0f, 1.0f, 1.0f);
		m_pVertexPositions[3] = glm::vec4( 1.0f, -1.0f, 1.0f, 1.0f);
		// face 2						
		m_pVertexPositions[4] = glm::vec4( 1.0f, -1.0f,	1.0f, 1.0f);
		m_pVertexPositions[5] = glm::vec4( 1.0f,  1.0f,	1.0f, 1.0f);
		m_pVertexPositions[6] = glm::vec4( 1.0f,  1.0f, -1.0f, 1.0f);
		m_pVertexPositions[7] = glm::vec4( 1.0f, -1.0f, -1.0f, 1.0f);
		// face 3
		m_pVertexPositions[8]  = glm::vec4( 1.0f, -1.0f, -1.0f, 1.0f);
		m_pVertexPositions[9]  = glm::vec4( 1.0f,  1.0f, -1.0f, 1.0f);
		m_pVertexPositions[10] = glm::vec4(-1.0f,  1.0f, -1.0f, 1.0f);
		m_pVertexPositions[11] = glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f);
		// face 4
		m_pVertexPositions[12] = glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f);
		m_pVertexPositions[13] = glm::vec4(-1.0f,  1.0f, -1.0f, 1.0f);
		m_pVertexPositions[14] = glm::vec4(-1.0f,  1.0f,  1.0f, 1.0f);
		m_pVertexPositions[15] = glm::vec4(-1.0f, -1.0f,  1.0f, 1.0f);
		// face 5
		m_pVertexPositions[16] = glm::vec4(-1.0f,  1.0f,  1.0f, 1.0f);
		m_pVertexPositions[17] = glm::vec4(-1.0f,  1.0f, -1.0f, 1.0f);
		m_pVertexPositions[18] = glm::vec4( 1.0f,  1.0f, -1.0f, 1.0f);
		m_pVertexPositions[19] = glm::vec4( 1.0f,  1.0f,  1.0f, 1.0f);
		// face 6
		m_pVertexPositions[20] = glm::vec4( 1.0f, -1.0f,  1.0f, 1.0f);
		m_pVertexPositions[21] = glm::vec4( 1.0f, -1.0f, -1.0f, 1.0f);
		m_pVertexPositions[22] = glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f);
		m_pVertexPositions[23] = glm::vec4(-1.0f, -1.0f,  1.0f, 1.0f);

		// normals face 1
		m_pVertexNormals[0] = glm::vec3( 0.0f,	0.0f,	1.0f);
		m_pVertexNormals[1] = glm::vec3( 0.0f,	0.0f,	1.0f);
		m_pVertexNormals[2] = glm::vec3( 0.0f,	0.0f,	1.0f);
		m_pVertexNormals[3] = glm::vec3( 0.0f,	0.0f,	1.0f);
		// normals face 2
		m_pVertexNormals[4] = glm::vec3( 1.0f,	0.0f,	0.0f);
		m_pVertexNormals[5] = glm::vec3( 1.0f,	0.0f,	0.0f);
		m_pVertexNormals[6] = glm::vec3( 1.0f,	0.0f,	0.0f);
		m_pVertexNormals[7] = glm::vec3( 1.0f,	0.0f,	0.0f);
		// normals face 3
		m_pVertexNormals[8]	= glm::vec3( 0.0f,	0.0f, -1.0f);
		m_pVertexNormals[9]	= glm::vec3( 0.0f,	0.0f, -1.0f);
		m_pVertexNormals[10] = glm::vec3( 0.0f,	0.0f, -1.0f);
		m_pVertexNormals[11] = glm::vec3( 0.0f,	0.0f, -1.0f);
		// normals face 4
		m_pVertexNormals[12] = glm::vec3(-1.0f,	0.0f,	0.0f);
		m_pVertexNormals[13] = glm::vec3(-1.0f,	0.0f,	0.0f);
		m_pVertexNormals[14] = glm::vec3(-1.0f,	0.0f,	0.0f);
		m_pVertexNormals[15] = glm::vec3(-1.0f,	0.0f,	0.0f);
		// normals face 5
		m_pVertexNormals[16] = glm::vec3( 0.0f,	1.0f,	0.0f);
		m_pVertexNormals[17] = glm::vec3( 0.0f,	1.0f,	0.0f);
		m_pVertexNormals[18] = glm::vec3( 0.0f,	1.0f,	0.0f);
		m_pVertexNormals[19] = glm::vec3( 0.0f,	1.0f,	0.0f);
		// normals face 6
		m_pVertexNormals[20] = glm::vec3( 0.0f, -1.0f,	0.0f);
		m_pVertexNormals[21] = glm::vec3( 0.0f, -1.0f,	0.0f);
		m_pVertexNormals[22] = glm::vec3( 0.0f, -1.0f,	0.0f);
		m_pVertexNormals[23] = glm::vec3( 0.0f, -1.0f,	0.0f);

		numberOfTriangles = 12;
		m_pIndexData = new ushort[3 * numberOfTriangles];
		// triangles face 1
		m_pIndexData[0] = 0; m_pIndexData[1] = 1; m_pIndexData[2] = 2;
		m_pIndexData[3] = 0; m_pIndexData[4] = 2; m_pIndexData[5] = 3;
		// triangles face 2
		m_pIndexData[6] = 4; m_pIndexData[7]	= 5; m_pIndexData[8]	= 6;
		m_pIndexData[9] = 4; m_pIndexData[10] = 6; m_pIndexData[11] = 7;
		// triangles face 3
		m_pIndexData[12] = 8; m_pIndexData[13] = 9; m_pIndexData[14] = 10;
		m_pIndexData[15] = 8; m_pIndexData[16] = 10; m_pIndexData[17] = 11;
		// triangles face 4
		m_pIndexData[18] = 12; m_pIndexData[19] = 13; m_pIndexData[20] = 14;
		m_pIndexData[21] = 12; m_pIndexData[22] = 14; m_pIndexData[23] = 15;
		// triangles face 5
		m_pIndexData[24] = 16; m_pIndexData[25] = 17; m_pIndexData[26] = 18;
		m_pIndexData[27] = 16; m_pIndexData[28] = 18; m_pIndexData[29] = 19;
		// triangles face 6
		m_pIndexData[30] = 20; m_pIndexData[31] = 21; m_pIndexData[32] = 22;
		m_pIndexData[33] = 20; m_pIndexData[34] = 22; m_pIndexData[35] = 23;
	}
	~CCubeMesh() 
	{
		SAFE_DELETE_ARRAY(m_pVertexPositions);
		SAFE_DELETE_ARRAY(m_pVertexNormals);
		SAFE_DELETE_ARRAY(m_pIndexData);

		std::vector<Triangle*>::iterator it;
		for(it=triangles.begin(); it < triangles.end(); ++it)
		{
			SAFE_DELETE(*it);
		}
		triangles.clear();
	}
};

#endif // _C_MESH_H_