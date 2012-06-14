
#ifndef _OCTAHEDRON_UTIL_H_
#define _OCTAHEDRON_UTIL_H_

#include <glm/glm.hpp>

float sign(const float& f)
{
	float res;
	res = f > 0 ? 1.f : -1.f;
	return res;
}

glm::vec2 sign(const glm::vec2& v)
{
	glm::vec2 res;
	res.x = v.x > 0 ? 1.f : -1.f;
	res.y = v.y > 0 ? 1.f : -1.f;
	return res;
}

glm::vec3 sign(const glm::vec3& v)
{
	glm::vec3 res;
	res.x = v.x > 0 ? 1.f : -1.f;
	res.y = v.y > 0 ? 1.f : -1.f;
	res.z = v.z > 0 ? 1.f : -1.f;
	return res;
}

glm::vec2 abs(const glm::vec2& v)
{
	glm::vec2 res;
	res.x = v.x > 0 ? v.x : -v.x;
	res.y = v.y > 0 ? v.y : -v.y;
	return res;
}

glm::vec3 abs(const glm::vec3& v)
{
	glm::vec3 res;
	res.x = v.x > 0 ? v.x : -v.x;
	res.y = v.y > 0 ? v.y : -v.y;
	res.z = v.z > 0 ? v.z : -v.z;
	return res;
}

glm::vec2 GetTexCoordForDirection(glm::vec3 d) 
{
	d /= glm::dot( glm::vec3(1.f), glm::abs(d) );
	
	if ( d.y < 0.0f )
	{
		float x = (1.f - abs(d.z)) * sign(d.x);
		float z = (1.f - abs(d.x)) * sign(d.z);
		d.x = x;
		d.z = z;
	}
	
	d.x = d.x * 0.5f + 0.5f;
	d.z = d.z * 0.5f + 0.5f;
	
	return glm::vec2(d.x, d.z);
}

glm::vec3 GetDirectionForTexCoord(glm::vec2 tex)
{
	tex = 2.f * (tex - glm::vec2(0.5f));

	float x = tex.x;
	float y = tex.y;
	
	glm::vec3 dir = glm::vec3(0.f);	

	if(	x < 0 && y < 0 && x + y < -1 ||
		x > 0 && y > 0 && x + y >  1 ||
		x > 0 && y < 0 && x - y >  1 ||
		x < 0 && y > 0 && x - y < -1 ) 
	{
		tex.y = - sign(x) * sign(y) * (x - sign(x));
		tex.x = - sign(x) * sign(y) * (y - sign(y));
		dir.y = - (1 - abs(tex.x) - abs(tex.y));
	}
	else
	{
		dir.y = 1 - abs(tex.x) - abs(tex.y);
	}
	
	dir.x = tex.x;
	dir.z = tex.y;
			
	return dir;
}

#endif _OCTAHEDRON_UTIL_H_