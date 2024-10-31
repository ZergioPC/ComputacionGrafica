#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color_vertice;

out vec3 randColor;

uniform mat4 vista;
uniform mat4 proyeccion;
uniform mat4 transformacion;

void main()
{
    gl_Position = vec4(position, 1.0)* vista* proyeccion * transformacion;
    randColor = color_vertice;
}