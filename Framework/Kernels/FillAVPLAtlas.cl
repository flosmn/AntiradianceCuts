#define PI 3.14159265f

struct AVPL_BUFFER
{
	float4 I;		// Intensity;
	float4 A;		// Antiintensity;
	float4 pos;		// Position
	float4 norm;	// Orientation;
	float4 w_A;		// AntiintensityDirection;
};

float3 GetDirForTexCoord(float2 texCoord);
float2 GetTexCoordForDir(float3 dir);

float4 SampleTexel(uint x, uint y, int tile_dim, const int sqrt_num_ss_samples, const float N, struct AVPL_BUFFER avpl, bool border);
float3 GetIntensity(float3 direction, struct AVPL_BUFFER avpl);
float3 GetAntiintensity(float3 direction, struct AVPL_BUFFER avpl, float N);

__kernel void Fill(
	write_only image2d_t image,
	int tile_dim,
	int atlas_dim,
	int num_avpls,
	int sqrt_num_ss_samples,
	int N,
	int border,
	__global struct AVPL_BUFFER* pAvplBuffer)
{
	const int numColumns = atlas_dim / tile_dim;
	
	int2 group_id;
	group_id.x = get_group_id(0);
	group_id.y = get_group_id(1);
	
	int2 local_id;
	local_id.x = get_local_id(0);
	local_id.y = get_local_id(1);

	int2 local_size;
	local_size.x = get_local_size(0);
	local_size.y = get_local_size(1);
	
	int avplIndex = group_id.y * numColumns + group_id.x;
	if(group_id.y * numColumns + group_id.x > num_avpls - 1)
		return;
	
	int iter_x = tile_dim / local_size.x;
	int iter_y = tile_dim / local_size.y;
	
	int b = border > 0 ? 1 : 0;
	for(int x = 0; x < iter_x; ++x)
	{
		for(int y = 0; y < iter_y; ++y)
		{	
			int2 localCoords = (int2)(x * local_size.x + local_id.x, y * local_size.y + local_id.y);
			float4 color;
			
			// skip border
			if(localCoords.x < b || localCoords.x >= tile_dim - b || 
				localCoords.y < b || localCoords.y >= tile_dim - b)
			{
				continue;
			}
			else
			{
				color = SampleTexel(localCoords.x-b, localCoords.y-b, tile_dim, sqrt_num_ss_samples, N, pAvplBuffer[avplIndex], border);
			}
			
			int2 globalCoords = localCoords + (int2)(group_id.x * tile_dim, group_id.y * tile_dim);
			write_imagef(image, globalCoords, color);

			if(border > 0)
			{
				if(localCoords.y == 1)
				{
					int2 go = (int2)(group_id.x * tile_dim, group_id.y * tile_dim);
					int2 gc2 = (int2)((tile_dim-1)-localCoords.x, 0) + go;
					write_imagef(image, gc2, color);

					if (localCoords.x == 1) {
						gc2 = (int2)(tile_dim-1, tile_dim-1) + go;
						write_imagef(image, gc2, color);
					} else if (localCoords.x == tile_dim-2) {
						gc2 = (int2)(0, tile_dim-1) + go;
						write_imagef(image, gc2, color);
					}
				}
				if(localCoords.y == tile_dim-2)
				{
					int2 go = (int2)(group_id.x * tile_dim, group_id.y * tile_dim);
					int2 gc2 = (int2)((tile_dim-1)-localCoords.x, tile_dim-1) + go;
					write_imagef(image, gc2, color);

					if (localCoords.x == 1) {
						gc2 = (int2)(tile_dim-1, 0) + go;
						write_imagef(image, gc2, color);
					} else if (localCoords.x == tile_dim-2) {
						gc2 = (int2)(0, 0) + go;
						write_imagef(image, gc2, color);
					}
				}
				if(localCoords.x == 1)
				{
					int2 go = (int2)(group_id.x * tile_dim, group_id.y * tile_dim);
					int2 gc2 = (int2)(0, (tile_dim-1)-localCoords.y) + go;
					write_imagef(image, gc2, color);

					if (localCoords.y == 1) {
						gc2 = (int2)(tile_dim-1, tile_dim-1) + go;
						write_imagef(image, gc2, color);
					} else if (localCoords.y == tile_dim-2) {
						gc2 = (int2)(tile_dim-1, 0) + go;
						write_imagef(image, gc2, color);
					}
				}
				if(localCoords.x == tile_dim-2)
				{
					int2 go = (int2)(group_id.x * tile_dim, group_id.y * tile_dim);
					int2 gc2 = (int2)(tile_dim-1, (tile_dim-1)-localCoords.y) + go;
					write_imagef(image, gc2, color);

					if (localCoords.y == 1) {
						gc2 = (int2)(0, tile_dim-1) + go;
						write_imagef(image, gc2, color);
					} else if (localCoords.y == tile_dim-2) {
						gc2 = (int2)(0, 0) + go;
						write_imagef(image, gc2, color);
					}
				}
			}
		}
	}
}

float4 SampleTexel(uint x, uint y, int tile_dim, const int sqrt_num_ss_samples, const float N, struct AVPL_BUFFER avpl, bool border)
{
	uint b = border ? 1 : 0;
	const int num_ss_samples = sqrt_num_ss_samples * sqrt_num_ss_samples;

	const float texel_size = 1.f / (float)(tile_dim - 2 * b);
	const float delta = 1.f / (float)(sqrt_num_ss_samples + 1);

	float3 A = (float3)(0.f);
	float3 I = (float3)(0.f);

	for(int i = 0; i < sqrt_num_ss_samples; ++i)
	{
		for(int j = 0; j < sqrt_num_ss_samples; ++j)
		{
			float2 texCoord = texel_size * (float2)((float)(x + (i+1) * delta), (float)(y + (j+1) * delta));
			float3 direction = normalize(GetDirForTexCoord(texCoord));
			
			A += GetAntiintensity(direction, avpl, N);
			I += GetIntensity(direction, avpl);
		}
	}		
		
	A *= 1.f/(float)(num_ss_samples);
	I *= 1.f/(float)(num_ss_samples);
			
	return (float4)(I - A, 1.f);
}

__kernel void Clear(
	write_only image2d_t image)
{
	int2 global_id;
	global_id.x = get_global_id(0);
	global_id.y = get_global_id(1);
		
	float4 color = (float4)(0.f, 0.f, 1.f, 0.f);
	
	write_imagef(image, global_id, color);
}

float3 GetIntensity(float3 w, struct AVPL_BUFFER avpl)
{
	return clamp(dot(w, avpl.norm.xyz), 0.f, 1.f) * avpl.I.xyz;
}

float3 GetAntiintensity(float3 w, struct AVPL_BUFFER avpl, float N)
{
	float3 res = (float3)(0.f, 0.f, 0.f);
	
	float cos_theta = dot(w, avpl.w_A.xyz);
	if(cos_theta < 0.01f)
	{
		return res;
	}

	const float theta = acos(clamp(cos_theta, 0.f, 1.f));
	
	if(theta < PI/N)
	{
		res = avpl.A.xyz;
	}
	
	return res;
}

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