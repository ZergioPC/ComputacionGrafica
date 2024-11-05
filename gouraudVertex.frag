#version 330 core
layout(location = 0) in vec3 posicion;
layout(location = 1) in vec3 normal;

out vec3 color_frag;

uniform mat4 proyeccion; 
uniform mat4 vista;
uniform mat4 transformacion;

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
    // Transformación de la posición
    gl_Position = proyeccion * vista * transformacion * vec4(posicion, 1.0);

    // Calcular posición y normal en el espacio del mundo
    vec3 posicion_mundo = vec3(transformacion * vec4(posicion, 1.0));
    vec3 normal_mundo = normalize(mat3(transpose(inverse(transformacion))) * normal);

    // Dirección de la luz y vista
    vec3 direccion_luz = normalize(luz_position - posicion_mundo);
    vec3 direccion_vista = normalize(-posicion_mundo);

    // Componente ambiental
    vec3 componente_ambiental = luz_ambient * mat_ambient;

    // Componente difusa
    float intensidad_difusa = max(dot(normal_mundo, direccion_luz), 0.0);
    vec3 componente_difusa = luz_difuse * (intensidad_difusa * mat_difuse);

    // Componente especular
    vec3 direccion_reflejo = reflect(-direccion_luz, normal_mundo);
    float intensidad_especular = pow(max(dot(direccion_vista, direccion_reflejo), 0.0), mat_brillo);
    vec3 componente_especular = luz_specular * (intensidad_especular * mat_specular);

    // Color final calculado en el vértice
    color_frag = componente_ambiental + componente_difusa + componente_especular;
}
