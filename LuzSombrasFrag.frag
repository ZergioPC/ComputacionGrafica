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
    vec3 luz_dir = normalize(luz_position - posicion_frag);

    vec3 comp_ambient = luz_ambient * mat_ambient;

    float difuse_intencidad = max(dot(normal,luz_dir),0.0);
    vec3 comp_difuse = luz_difuse * (difuse_intencidad * mat_difuse);

    vec3 dir_vista = normalize(-posicion_frag);
    vec3 dir_reflex = reflect(-luz_dir,normal);
    float specular_intencidad = pow(max(dot(dir_vista,dir_reflex),0.0),mat_brillo);
    vec3 comp_specular = luz_specular * (specular_intencidad  + mat_specular)

    vec3 color_final = comp_ambient + comp_difuse + comp_specular;
    color = vec4(color_final, 1.0);
}