import glfw
import glm
import ctypes
import numpy as np
from OpenGL.GL import *



code_shader_vertice = """
#version 330 core
layout(location = 0) in vec3 posicion;

uniform mat4 proyeccion;
uniform mat4 vista;
uniform mat4 modelo;

void main(){
    gl_Position = proyeccion * vista * modelo * vec4(posicion, 1.0);
}
"""

code_shader_frag = """
#version 330 core
out vec4 colorFragmento;

void main(){
    colorFragmento = vec4(0.8,0.3,0.0,1.0);
}
"""

def compilar_shader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader) 
    glShaderSource(shader, codigo) 
    glCompileShader(shader) 
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader 

def crear_programa_shader(frag_Code,vertx_Code):
    shader_vertices = compilar_shader(vertx_Code, GL_VERTEX_SHADER)
    shader_fragmentos = compilar_shader(frag_Code, GL_FRAGMENT_SHADER)

    programa_shader = glCreateProgram()
    glAttachShader(programa_shader, shader_vertices)
    glAttachShader(programa_shader, shader_fragmentos)
    glLinkProgram(programa_shader)
    if glGetProgramiv(programa_shader, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(programa_shader))

    glDeleteShader(shader_vertices)
    glDeleteShader(shader_fragmentos)
    
    return programa_shader

def bufferConfg(vertices,indices):
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

    return VAO

def main():
    if not glfw.init():
        raise Exception("No GLFW")

    window = glfw.create_window(800,600,"Superficies de Recorrido", None, None)

    if not window:
        glfw.terminate()
        raise Exception("No window")
    
    glfw.make_context_current(window)
    glClearColor( 0, 0 , 0, 1)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glfw.swap_buffers(window)
        glfw.poll_events()

    #glDeleteVertexArrays(1, [VAO])
    #glDeleteBuffers(1, [VBO])
    #glDeleteBuffers(1, [EBO])
    #glDeleteProgram(programa_shader)
    
    
    glfw.terminate()

if __name__ == "__main__":
    main()