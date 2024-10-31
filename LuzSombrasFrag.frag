#version 330 core

in vec3 randColor;
out vec4 color;

void main()
{
    color = vec4(randColor, 1.0);
    
}