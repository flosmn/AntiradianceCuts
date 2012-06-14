#version 420

out vec4 outputColor;

smooth in vec3 positionWS;

layout(binding=0) uniform sampler2D samplerOEM;

vec2 GetTexCoordForDirection(vec3 d) 
{
	// projection onto octahedron
	d /= dot( vec3(1.f), abs(d) );
	
	// out-folding of the downward faces
	if ( d.y < 0.0f )
	{
		float x = (1-abs(d.z)) * sign(d.x);
		float z = (1-abs(d.x)) * sign(d.z);
		d.x = x;
		d.z = z;
	}
	// mapping to [0;1]ˆ2 texture space
	d.xz = d.xz * 0.5 + 0.5;
	
	return d.xz;
}

vec3 GetDirectionForTexCoord(vec2 tex)
{
	tex = 2.f * (tex - 0.5f);

	float x = tex.x;
	float y = tex.y;
	
	vec3 dir = vec3(0.f);	

	if(	x < 0 && y < 0 && x + y < -1 ||
		x > 0 && y > 0 && x + y >  1 ||
		x > 0 && y < 0 && x - y >  1 ||
		x < 0 && y > 0 && x - y < -1 ) 
	{
		tex.y = - sign(x) * sign(y) *(x - sign(x));
		tex.x = - sign(x) * sign(y) *(y - sign(y));
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

uniform model
{
	vec3 positionWS;
} uModel;

void main()
{
	vec3 direction = normalize(positionWS - uModel.positionWS);
	vec2 texCoord = GetTexCoordForDirection(direction);
	vec3 dir = GetDirectionForTexCoord(texCoord);
	outputColor = vec4(0.5 * dir + 0.5, 1.f);
	
	//vec4 texel = texture2D(samplerOEM, GetTexCoordForDirection(direction));
	//outputColor = texel;
}