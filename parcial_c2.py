"""
Sergio Danilo Palacios
"""

import glfw 
from OpenGL.GL import * 
import numpy as np 
import ctypes 

Codigo_shaderVertices = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 vista;
uniform mat4 proyeccion;
uniform mat4 transformacion;

void main()
{
    gl_Position = vec4(position, 1.0)* vista* proyeccion * transformacion;
}
"""

Codigo_shaderFragmentos1 = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(1.0, 0.5, 0.2, 1.0);
    
}
"""

def compilar_shader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader) 
    glShaderSource(shader, codigo) 
    glCompileShader(shader) 
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader 

def crear_programa_shader(frag_Code):
    shader_vertices = compilar_shader(Codigo_shaderVertices, GL_VERTEX_SHADER)
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

def config_Buffers(vertices,indices):
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


def curva(y):
    return y*0.01


def superficie(directriz,r_base,H):
    vertices = []
    indices = []

    u = np.linspace(0,2*np.pi,100)
    v = np.linspace(0,2,50)

    for j in range(len(v)):
        for i in range(len(u)):
            r = r_base + directriz(j)
            x,y,z = r*np.cos(u[i]),v[j],r*np.sin(u[i])
            vertices.append([x,y*H,z])

    num_u = len(u)

    for j in range(len(v)-1):
        for i in range(num_u):
            indices.append(j*num_u+i)
            indices.append((j+1)*num_u+i)
        indices.append(j*num_u)
        indices.append((j+1)*num_u)

    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

def matriz_vista(ojo, centro, arriba):
    f = np.array(centro)-np.array(ojo)
    f = f/np.linalg.norm(f) 
    u = np.array(arriba)
    u = u/np.linalg.norm(u)
    s = np.cross(f,u)
    s = s/np.linalg.norm(s)
    u = np.cross(s,f)
    
    M = np.identity(4)
    M [0,:3]= s
    M [1,:3]= u
    M [2,:3]= -f
    T = np.identity (4)
    T[:3,3]= -np.array(ojo)
    
    return M @ T

def matriz_perspectiva(fovY, aspecto,cerca,lejos):
    f= 1.0 /np.tan(np.radians(fovY)/2)
    q=(lejos+cerca)/(cerca-lejos)
    p=(2*lejos*cerca)/(cerca-lejos)
    
    M= np.array([
        [f/aspecto, 0,0,0],
        [0,f,0,0],
        [0,0,q,p],
        [0,0,-1,0]], dtype=np.float32)
    
    return M

def dibujarFigura(indices,VAO,shader,uvista,modelo_vista,uproyeccion,modelo_proyeccion,umodelo,modelo_figura):
    glUseProgram(shader)
    glUniformMatrix4fv(uvista, 1, GL_FALSE, modelo_vista.flatten())        
    glUniformMatrix4fv(uproyeccion,1,GL_FALSE,modelo_proyeccion.flatten())
    glUniformMatrix4fv(umodelo, 1, GL_FALSE, modelo_figura.flatten())
    glBindVertexArray(VAO)
    glDrawElements(GL_LINE_STRIP, len(indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

def main():
    
    if not glfw.init():
        return

    a,b =800,600
    ventana = glfw.create_window(a, b, "Proyecciones - Sergio Palacios", None, None)
    
    if not ventana: 
        glfw.terminate()
        return

    glfw.make_context_current(ventana)

    vertices_e1, indices_e1 = superficie(curva,0.6,0.2)

    programa_shader_1 = crear_programa_shader(Codigo_shaderFragmentos1)

    VAO_e1, VBO_e1, EBO_e1 = config_Buffers(vertices_e1,indices_e1)
    
    uvista= glGetUniformLocation(programa_shader_1, "vista")
    uproyeccion= glGetUniformLocation(programa_shader_1, "proyeccion")
    umodelo= glGetUniformLocation(programa_shader_1, "transformacion")

    view_base = matriz_vista([-0.15,0.1,5.5],[0,0,0],[0,1,0])
    proyeccion_perspectiva = matriz_perspectiva(45, a/b, 1, 5)
    
    while not glfw.window_should_close(ventana):
        glClearColor(0.3 ,0.2 ,0.3 ,1.0 )
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        dibujarFigura(indices_e1,VAO_e1,programa_shader_1,uvista,view_base,uproyeccion,proyeccion_perspectiva,umodelo,np.identity(4))

        glfw.swap_buffers(ventana)
        glfw.poll_events()

    glDeleteVertexArrays(1, [VAO_e1])
    glDeleteBuffers(1, [VBO_e1])
    glDeleteBuffers(1, [EBO_e1])

    glDeleteProgram(programa_shader_1)
    
glfw.terminate()

#usar las teclas para variar los valores
if __name__ == "__main__":
    main()
