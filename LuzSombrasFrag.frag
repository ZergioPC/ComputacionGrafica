#version 330 core

in vec3 posicion_frag;
in vec3 normal_frag;

out vec4 color;

uniform vec3 luz_position;
uniform vec3 luz_ambient;
uniform vec3 luz_difuse;
uniform vec3 luz_specular;

uniform vec3 mat_ambient;
uniform vec3 mat_difuse;
uniform vec3 mat_specular;
uniform float mat_brillo;

void main()
{
    vec3 normal = normalize(normal_frag);
    vec3 direccion_luz = normalize(luz_position - posicion_frag);

    // Componente ambiental
    vec3 componente_ambiental = luz_ambient * mat_ambient;

    // Componente difusa
    float intensidad_difusa = max(dot(normal, direccion_luz), 0.0);
    vec3 componente_difusa = luz_difuse * (intensidad_difusa * mat_difuse);

    // Componente especular
    vec3 direccion_vista = normalize(-posicion_frag);
    vec3 direccion_reflejo = reflect(-direccion_luz, normal);
    float intensidad_especular = pow(max(dot(direccion_vista, direccion_reflejo), 0.0), mat_brillo);
    vec3 componente_especular = luz_specular * (intensidad_especular * mat_specular);

    // Color final
    vec3 color_final = componente_ambiental + componente_difusa + componente_especular;
    color = vec4(color_final, 1.0);
}