
#ifndef _OCTAHEDRON_UTIL_CL_
#define _OCTAHEDRON_UTIL_CL_

float2 GetTexCoordForDir(float3 dir) {
	// Project from sphere onto octahedron
	dir /= dot(1.0f, fabs(dir));
	
	// If on lower hemisphere...
	if (dir.y < 0.0f) {
		// ...unfold
		float x = (1.0f - fabs(dir.z)) * sign(dir.x);
		float z = (1.0f - fabs(dir.x)) * sign(dir.z);
		dir.x = x;
		dir.z = z;
	}
		
	// [-1,1]^2 to [0,1]^2
	dir.xz = dir.xz * 0.5f + 0.5f;
	
	return dir.xz;
}

float3 GetDirForTexCoord(float2 texCoord) {
	float3 dir;

	dir.xz = texCoord;
	
	// [0,1]^2 to [-1,1]^2 
	dir.xz *= 2.0f;
	dir.xz -= 1.0f;
		
	float3 vAbs = fabs(dir);
	// If on lower hemisphere...
	if (vAbs.x + vAbs.z > 1.0f) {
		// ...fold
		float x = sign(dir.x) * (1.0f - vAbs.z);
		float z = sign(dir.z) * (1.0f - vAbs.x);
		dir.x = x;
		dir.z = z;
	}
	
	// Elevate height
	dir.y = 1.0f - vAbs.x - vAbs.z;

	// Project onto sphere
	dir = normalize(dir);

	return dir;
}

#endif
