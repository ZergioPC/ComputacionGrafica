"""
Hecho por:
-Sofía Victoria Marín
-Carlos Julian Morales
-Sergio Palacios

"""


import glfw;
from OpenGL.GL import *;
import glm;
from glm import value_ptr;
import numpy as np;
import ctypes;

# Shaders
shaderVertex = """
#version 330 core
layout(location = 0) in vec3 posicion;
uniform mat4 transformacion;
void main()
{
    gl_Position = transformacion * vec4(posicion, 1.0);
}
""";

shaderFragment = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(1.0, 0.5, 0.5, 0.5);
}
"""

def compilarShader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader);
    glShaderSource(shader, codigo);
    glCompileShader(shader);

    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def programa():
    shader_vertices = compilarShader(shaderVertex, GL_VERTEX_SHADER);
    shader_fragmentos = compilarShader(shaderFragment, GL_FRAGMENT_SHADER);

    programa_shader = glCreateProgram();

    glAttachShader(programa_shader, shader_vertices);
    glAttachShader(programa_shader, shader_fragmentos);

    glLinkProgram(programa_shader);

    if glGetProgramiv(programa_shader, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(programa_shader));

    glDeleteShader(shader_vertices);
    glDeleteShader(shader_fragmentos);

    return programa_shader;

def configurar_vao(vertices, indices):
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return VAO, VBO, EBO

def dibujar_objeto(programa_shader, VAO, transformacion, indices_count):
    glUseProgram(programa_shader)
    transform_loc = glGetUniformLocation(programa_shader, "transformacion")
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, value_ptr(transformacion))
    glBindVertexArray(VAO)
    glDrawElements(GL_LINES, indices_count, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

def prisma():
    vertex = [0.5,0.4,0.5,
              0.5,0.4,-0.5,
              -0.5,0.4,-0.5,
              -0.5,0.4,0.5,
              0.5,-1.0,0.5,
              0.5,-1.0,-0.5,
              -0.5,-1.0,-0.5,
              -0.5,-1.0,0.5, 
    ];

    for p in range(len(vertex)):
        vertex[p] = vertex[p]*0.5

    index = [
        0,1,2,2,3,0,
        4,5,6,6,7,4,
        0,1,5,5,4,0,
        3,2,6,6,7,3,
        0,3,7,7,4,0,
        1,2,6,6,5,1
    ]; 

    vertex = np.array(vertex,dtype=np.float32);
    index = np.array(index,dtype=np.uint32).flatten();
    return vertex, index;

def esfera(radio, nstack, nsectors):
    vertices = []
    indices = []
    dfi = np.pi / nstack
    dteta = 2 * np.pi / nsectors
    for i in range(nstack + 1):
        fi = -np.pi / 2 + i * dfi
        temp = radio * np.cos(fi)
        y = radio * np.sin(fi)
        for j in range(nsectors + 1):
            teta = j * dteta
            x = temp * np.sin(teta)
            z = temp * np.cos(teta)
            vertices.append([x])
            vertices.append([y+0.45])
            vertices.append([z])
            if i < nstack and j < nsectors:
                first = i * (nsectors + 1) + j
                second = first + nsectors + 1
                indices.append(first)
                indices.append(second)
                indices.append(first + 1)
                indices.append(second)
                indices.append(second + 1)
                indices.append(first + 1)

    vertices = np.array(vertices, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

def rotArb(punto_rotacion,eje_arbitrario,angulo): 
    eje_arbitrario_normalizado = glm.normalize(eje_arbitrario)
    angulo_y = glm.atan(eje_arbitrario_normalizado.z, eje_arbitrario_normalizado.x)
    rotacion_y = glm.rotate(glm.mat4(1.0), angulo_y, glm.vec3(0.0, 1.0, 0.0))
    vector_transformado = rotacion_y * glm.vec4(eje_arbitrario_normalizado,1.0)
    angulo_x = glm.atan(vector_transformado.y, vector_transformado.z)
    rotacion_x = glm.rotate(glm.mat4(1.0), angulo, glm.vec3(0.0, 0.0, 1.0))
    traslacion_origen = glm.translate(glm.mat4(1.0), -punto_rotacion)
    rotacion_z = glm.rotate(glm.mat4(1.0), angulo, glm.vec3(0.0, 0.0, 1.0))
    desrotacion_x = glm.rotate(glm.mat4(1.0), angulo_x, glm.vec3(1.0, 0.0, 0.0))
    desrotacion_y = glm.rotate(glm.mat4(1.0), angulo_y, glm.vec3(0.0, 1.0, 0.0))
    destraslacion_origen = glm.translate(glm.mat4(1.0), punto_rotacion)
    transformacion_final = destraslacion_origen * desrotacion_y * desrotacion_x * rotacion_z * rotacion_x * rotacion_y * traslacion_origen
    return transformacion_final

def main():
    if not glfw.init():
        raise Exception("No GLFW init");

    screen = glfw.create_window(800,600,"Test",None,None);

    if not screen:
        raise Exception("No vista mano");

    glfw.make_context_current(screen);

    vertices_prisma, indices_prisma = prisma()
    vertices_esfera, indices_esfera = esfera(0.25, 10, 10)

    puntoRot = glm.vec3(0.5, 0.2, -0.5)
    ejeRot = glm.vec3(0.3, 0.8, 0.1)
    angulo = glm.radians(20)

    programa_shader = programa()

    VAO_prisma, VBO_prisma, EBO_prisma = configurar_vao(vertices_prisma, indices_prisma)
    VAO_esfera, VBO_esfera, EBO_esfera = configurar_vao(vertices_esfera, indices_esfera)

    glPointSize(10)

    while not glfw.window_should_close(screen):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        transform_prisma = rotArb(puntoRot, ejeRot, angulo)
        transform_esfera = glm.translate(transform_prisma, glm.vec3(0.0, 0.6, 0.0))  # Mueve la esfera encima del prisma

        #Figura Transformada
        dibujar_objeto(programa_shader, VAO_prisma, transform_prisma, len(indices_prisma))
        dibujar_objeto(programa_shader, VAO_esfera, transform_esfera, len(indices_esfera))

        #Figura en su estado inicial
        #dibujar_objeto(programa_shader, VAO_prisma, glm.mat4(1.0), len(indices_prisma))
        #dibujar_objeto(programa_shader, VAO_esfera, glm.mat4(1.0), len(indices_esfera))

        glfw.swap_buffers(screen);
        glfw.poll_events(); 

    glDeleteVertexArrays(1, [VAO_prisma])
    glDeleteBuffers(1, [VBO_prisma])
    glDeleteBuffers(1, [EBO_prisma])
    glDeleteVertexArrays(1, [VAO_esfera])
    glDeleteBuffers(1, [VBO_esfera])
    glDeleteBuffers(1, [EBO_esfera])
    glDeleteProgram(programa_shader)

    glfw.terminate();

if __name__ == "__main__":
    main();
