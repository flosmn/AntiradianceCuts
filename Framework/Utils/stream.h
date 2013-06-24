#ifndef STREAM_H_
#define STREAM_H_

#include <iostream>
#include <glm/glm.hpp>

inline std::ostream& operator<<(std::ostream& stream, glm::ivec2 const& v)
{
    return stream << "(" << v.x << ", " << v.y << ")";
}

inline std::ostream& operator<<(std::ostream& stream, glm::ivec3 const& v)
{
    return stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

inline std::ostream& operator<<(std::ostream& stream, glm::ivec4 const& v)
{
    return stream << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
}

inline std::ostream& operator<<(std::ostream& stream, glm::uvec2 const& v)
{
    return stream << "(" << v.x << ", " << v.y << ")";
}

inline std::ostream& operator<<(std::ostream& stream, glm::uvec3 const& v)
{
    return stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

inline std::ostream& operator<<(std::ostream& stream, glm::uvec4 const& v)
{
    return stream << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
}

inline std::ostream& operator<<(std::ostream& stream, glm::vec2 const& v)
{
    return stream << "(" << v.x << ", " << v.y << ")";
}

inline std::ostream& operator<<(std::ostream& stream, glm::vec3 const& v)
{
    return stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

inline std::ostream& operator<<(std::ostream& stream, glm::vec4 const& v)
{
    return stream << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
}

inline std::ostream& operator<<(std::ostream& stream, glm::mat4 const& v)
{
    return stream 
		<< "(" 
		<< v[0][0] << ", " << v[1][0] << ", " << v[2][0] << ", " << v[3][0] << " | "
		<< v[0][1] << ", " << v[1][1] << ", " << v[2][1] << ", " << v[3][1] << " | "
		<< v[0][2] << ", " << v[1][2] << ", " << v[2][2] << ", " << v[3][2] << " | "
		<< v[0][3] << ", " << v[1][3] << ", " << v[2][3] << ", " << v[3][3] << ")";
}

#endif
