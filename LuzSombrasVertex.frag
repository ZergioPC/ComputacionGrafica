#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 posicion_frag;
out vec3 normal_frag;

uniform mat4 proyeccion;
uniform mat4 vista;
uniform mat4 transformacion;

void main()
{
    gl_Position =  proyeccion * vista * transformacion * vec4(position, 1.0);
    posicion_frag = vec3(transformacion * vec4(position,1.0));
    normal_frag = mat3(transpose(inverse(transformacion))) * normal;
}