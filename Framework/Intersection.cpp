#include "Intersection.h"

#include "CPrimitive.h"

glm::vec3 Intersection::GetNormal() const {
	return m_pPrimitive->GetNormal(); 
}

uint Intersection::GetMaterialIndex() const { 
	return m_pPrimitive->GetMaterialIndex(); 
}