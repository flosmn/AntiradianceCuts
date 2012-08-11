#include "Intersection.h"

#include "CPrimitive.h"

glm::vec3 Intersection::GetNormal() const {
	return m_pPrimitive->GetNormal(); 
}

MATERIAL Intersection::GetMaterial() const { 
	return m_pPrimitive->GetMaterial(); 
}