#version 420

layout(std140) uniform;

layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;

smooth out float face;

void main()
{
	int i, layer;
	for (layer = 0; layer < 6; layer++) {
		gl_Layer = layer;
		for (i = 0; i < 3; i++) {
			gl_Position = gl_in[i].gl_Position;
			face = float(layer);
			EmitVertex();
		}
		EndPrimitive();
	}
 }