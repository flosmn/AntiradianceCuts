__kernel void VecAdd(
	__global const int* a,
	__global const int* b,
	__global int* c,
	int numElements
	)
{
	int GID = get_global_id(0);

	if(GID < numElements)
	{
		c[GID] = a[GID] + b[GID];
	}
}

__kernel void TexWriteTest(
	write_only image2d_t image,
	int width,
	int height)
{
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	float4 color;
	if((GID.x + GID.y) % 2 == 0)
		color = (float4)(1.f, 0.f, 0.f, 1.f);
	else
		color = (float4)(0.f, 1.f, 0.f, 1.f);

	write_imagef(image, GID, color);
}