#version 330 core

in vec3 color_frag;

out vec4 color;

void main()
{
    color = vec4(color_frag,1.0);
}