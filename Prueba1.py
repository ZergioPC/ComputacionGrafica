import glfw
from OpenGL.GL import *
import numpy as np
import glm
from glm import value_ptr

# Shaders
codigo_shader_vertices = """
#version 330 core
layout(location = 0) in vec3 posicion;
layout(location = 1) in vec3 normal;

out vec3 posicion_frag;
out vec3 normal_frag;

uniform mat4 proyeccion;
uniform mat4 vista;
uniform mat4 transformacion;

void main()
{
    gl_Position = proyeccion * vista * transformacion * vec4(posicion, 1.0);
    posicion_frag = vec3(transformacion * vec4(posicion, 1.0));
    normal_frag = mat3(transpose(inverse(transformacion))) * normal;
}
"""
codigo_shader_fragmentos = """
#version 330 core

in vec3 posicion_frag;
in vec3 normal_frag;

out vec4 color_fragmento;

uniform vec3 luz_posicion;
uniform vec3 luz_ambiental;
uniform vec3 luz_difusa;
uniform vec3 luz_especular;

uniform vec3 material_ambiental;
uniform vec3 material_difuso;
uniform vec3 material_especular;
uniform float material_brillo;

void main()
{
    vec3 normal = normalize(normal_frag);
    vec3 direccion_luz = normalize(luz_posicion - posicion_frag);

    // Componente ambiental
    vec3 componente_ambiental = luz_ambiental * material_ambiental;

    // Componente difusa
    float intensidad_difusa = max(dot(normal, direccion_luz), 0.0);
    vec3 componente_difusa = luz_difusa * (intensidad_difusa * material_difuso);

    // Componente especular
    vec3 direccion_vista = normalize(-posicion_frag);
    vec3 direccion_reflejo = reflect(-direccion_luz, normal);
    float intensidad_especular = pow(max(dot(direccion_vista, direccion_reflejo), 0.0), material_brillo);
    vec3 componente_especular = luz_especular * (intensidad_especular * material_especular);

    // Color final
    vec3 color_final = componente_ambiental + componente_difusa + componente_especular;
    color_fragmento = vec4(color_final, 1.0);
}

"""

def compilar_shader(fuente, tipo_shader):
    shader = glCreateShader(tipo_shader)
    glShaderSource(shader, fuente)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def crear_programa_shader(codigo_vertices, codigo_fragmentos):
    shader_vertices = compilar_shader(codigo_vertices, GL_VERTEX_SHADER)
    shader_fragmentos = compilar_shader(codigo_fragmentos, GL_FRAGMENT_SHADER)
    programa = glCreateProgram()
    glAttachShader(programa, shader_vertices)
    glAttachShader(programa, shader_fragmentos)
    glLinkProgram(programa)
    if glGetProgramiv(programa, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(programa))
    glDeleteShader(shader_vertices)
    glDeleteShader(shader_fragmentos)
    return programa

def configurar_buffers(vertices, normales, indices):
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(2)  # Un VBO para posiciones y otro para normales
    EBO = glGenBuffers(1)  # Buffer de elementos para las caras

    glBindVertexArray(VAO)

    # Buffer de posiciones
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    # Buffer de normales
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
    glBufferData(GL_ARRAY_BUFFER, normales.nbytes, normales, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    # Buffer de índices para las caras
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glBindVertexArray(0)

    return VAO, VBO, EBO

def generar_vertices_toroide(R, r, num_filas, num_cols):
    vertices = []
    normales = []
    for i in range(num_filas):
        theta = 2 * np.pi * i / num_filas
        for j in range(num_cols):
            phi = 2 * np.pi * j / num_cols
            x = (R + r * np.cos(theta)) * np.cos(phi)
            y = (R + r * np.cos(theta)) * np.sin(phi)
            z = r * np.sin(theta)
            vertices.append([x, y, z])

            # Normal calculada
            nx = np.cos(theta) * np.cos(phi)
            ny = np.cos(theta) * np.sin(phi)
            nz = np.sin(theta)
            normales.append([nx, ny, nz])

    return np.array(vertices, dtype=np.float32), np.array(normales, dtype=np.float32)

def generar_caras_toroide(num_filas, num_cols):
    caras = []
    for i in range(num_filas):
        for j in range(num_cols):
            siguiente_i = (i + 1) % num_filas
            siguiente_j = (j + 1) % num_cols
            caras.extend([
                i * num_cols + j,
                siguiente_i * num_cols + j,
                siguiente_i * num_cols + siguiente_j,
                i * num_cols + j,
                siguiente_i * num_cols + siguiente_j,
                i * num_cols + siguiente_j
            ])
    return np.array(caras, dtype=np.uint32)

# Iniciar ventana GLFW y OpenGL
def inicializar_glfw():
    if not glfw.init():
        return None
    ventana = glfw.create_window(800, 600, "Toroide con Sombreado e Iluminación", None, None)
    if not ventana:
        glfw.terminate()
        return None
    glfw.make_context_current(ventana)
    glEnable(GL_DEPTH_TEST)
    return ventana

def dibujar(ventana, programa_shader, VAO, proyeccion, vista, num_indices):
    try:
        while not glfw.window_should_close(ventana):
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(programa_shader)

            # Parámetros de iluminación y material
            glUniform3f(glGetUniformLocation(programa_shader, "luz_posicion"), 5.0, 5.0, 5.0)
            glUniform3f(glGetUniformLocation(programa_shader, "luz_ambiente"), 0.6, 0.2, 0.6)
            glUniform3f(glGetUniformLocation(programa_shader, "luz_difusa"), 0.8, 0.8, 0.8)
            glUniform3f(glGetUniformLocation(programa_shader, "luz_especular"), 1.0, 1.0, 1.0)

            glUniform3f(glGetUniformLocation(programa_shader, "material_ambiente"), 0.5, 0.3, 0.2)
            glUniform3f(glGetUniformLocation(programa_shader, "material_difuso"), 0.5, 0.3, 0.6)
            glUniform3f(glGetUniformLocation(programa_shader, "material_especular"), 1.0, 1.0, 1.0)
            glUniform1f(glGetUniformLocation(programa_shader, "material_brillo"), 32.0)

            # Actualizar la rotación en cada frame
            tiempo = glfw.get_time()
            angulo_rotacion = glm.radians(tiempo * 10)
            transformacion = glm.rotate(glm.mat4(1.0), angulo_rotacion, glm.vec3(0.0, 1.0, 0.0))

            glUniformMatrix4fv(glGetUniformLocation(programa_shader, "transformacion"), 1, GL_FALSE, value_ptr(transformacion))
            glUniformMatrix4fv(glGetUniformLocation(programa_shader, "proyeccion"), 1, GL_FALSE, value_ptr(proyeccion))
            glUniformMatrix4fv(glGetUniformLocation(programa_shader, "vista"), 1, GL_FALSE, value_ptr(vista))

            glBindVertexArray(VAO)
            glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

            glfw.swap_buffers(ventana)
            glfw.poll_events()

    finally:
        glDeleteProgram(programa_shader)
        glDeleteVertexArrays(1, [VAO])
        glfw.terminate()


def proyeccion_perspectiva(fov, relacion_aspecto, cerca, lejos):
    return glm.perspective(glm.radians(fov), relacion_aspecto, cerca, lejos)

def vista_LookAt(ojo, centro, arriba):
    return glm.lookAt(ojo, centro, arriba)

def main():
    ventana = inicializar_glfw()
    if not ventana:
        return

    programa_shader = crear_programa_shader(codigo_shader_vertices, codigo_shader_fragmentos)

    R, r = 1.0, 0.4
    num_filas, num_cols = 24, 24
    vertices, normales = generar_vertices_toroide(R, r, num_filas, num_cols)
    indices = generar_caras_toroide(num_filas, num_cols)

    VAO, VBO, EBO = configurar_buffers(vertices, normales, indices)

    fov = 60
    relacion_aspecto = 800 / 600
    cerca = 0.1
    lejos = 100
    proyeccion = proyeccion_perspectiva(fov, relacion_aspecto, cerca, lejos)

    ojo = glm.vec3(2, 1, 2)
    centro = glm.vec3(0.0, 0.0, 0.0)
    arriba = glm.vec3(0.0, 1.0, 0.0)
    vista = vista_LookAt(ojo, centro, arriba)

    dibujar(ventana, programa_shader, VAO, proyeccion, vista, len(indices))

if __name__ == "__main__":
    main()
