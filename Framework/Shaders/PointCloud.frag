#version 420

in vec3 instancePosition;
in vec3 instanceColor;

out vec4 outputColor;

void main() {
	outputColor = vec4(instanceColor, 1.f);
}

