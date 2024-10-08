"""
Hecho Por

Sergio Danilo Palacios
Carlos Julian Morales
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

Codigo_shaderFragmentos2 = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(1.0, 1.0, 1.0, 1.0);
    
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

def curva_potencial(y,a):
    return y*np.exp(a)

def curva_logaritmica(y,a):
    return np.log(y)/np.log(a)

def superficie(directriz,a):
    vertices = []
    indices = []

    p_camino = np.linspace(0,2*np.pi,100)
    p_perfil = np.linspace(0,2,40)

    for th in p_camino[:-1]:
        for y in p_perfil:
            for angulo in [th,th+np.diff(p_camino)[0]]:
                r = directriz(y,a)
                x = r*np.cos(angulo)
                z = r*np.sin(angulo)

                vertices.append([x,y,z])

    # Crear los indices
    for i in range(len(p_camino) - 1):
        for j in range(len(p_perfil) - 1):
            # Índices de los cuatro vértices del cuadrado en la malla
            v0 = i * len(p_perfil) + j
            v1 = (i + 1) * len(p_perfil) + j
            v2 = (i + 1) * len(p_perfil) + (j + 1)
            v3 = i * len(p_perfil) + (j + 1)

            # Crear dos triángulos para cada cuadrado
            indices.append(v0)
            indices.append(v1)
            indices.append(v2)

            indices.append(v0)
            indices.append(v2)
            indices.append(v3)

    vertices = np.array(vertices, dtype=np.float32).flatten()
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

def matriz_vista_uvn(n,v, vrp):
    n=np.array(n)
    v=np.array(v)
    u=np.cross(v,n)
    n=n/np.linalg.norm(n)
    v=v/np.linalg.norm(v)
    u=u/np.linalg.norm(u)
    M=np.identity(4)
    M[0, :3]=u
    M[1, :3]=v
    M[2, :3]=n
    a=np.dot(vrp,u)
    b=np.dot(vrp,v)
    c=np.dot(vrp,n)
    M[0,3], M[1,3], M[2,3]= -a,-b,-c
    return M

def matriz_pr_orto(iuc,dal): #IAC: izq, arriba, cerca
    t=np.identity(4)
    iuc=np.array(iuc)
    dal=np.array(dal)
    i,u,c = iuc
    d,a,l = dal
    t[0,3], t[1,3], t[2,3] =-(d+i)/2, -(u+a)/2,(l+c)/2
    s=np.identity(4)
    s[0,0], s[1,1],s[2,2] = 2/(d-i),2/(u-a),-2/(c-l)
    
    return s@t


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

    #vertices, indices = generar_prismaHex( altura, radio)
    vertices_e1, indices_e1 = superficie(curva_logaritmica,0.2)
    vertices_e2, indices_e2 = superficie(curva_potencial,0.2)

    programa_shader_1 = crear_programa_shader(Codigo_shaderFragmentos1)
    programa_shader_2 = crear_programa_shader(Codigo_shaderFragmentos2)

    VAO_e1, VBO_e1, EBO_e1 = config_Buffers(vertices_e1,indices_e1)
    VAO_e2, VBO_e2, EBO_e2 = config_Buffers(vertices_e2,indices_e2)
    
    uvista= glGetUniformLocation(programa_shader_1, "vista")
    uproyeccion= glGetUniformLocation(programa_shader_1, "proyeccion")
    umodelo= glGetUniformLocation(programa_shader_1, "transformacion")

    view_base = matriz_vista([-0.15,0.1,5.5],[0,0,0],[0,1,0])
    proyeccion_perspectiva = matriz_perspectiva(45, a/b, 1, 5)
    
    while not glfw.window_should_close(ventana):
        glClearColor(0.3 ,0.2 ,0.3 ,1.0 )
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        dibujarFigura(indices_e1,VAO_e1,programa_shader_1,uvista,view_base,uproyeccion,proyeccion_perspectiva,umodelo,np.identity(4))
        dibujarFigura(indices_e2,VAO_e2,programa_shader_2,uvista,view_base,uproyeccion,proyeccion_perspectiva,umodelo,np.identity(4))

        glfw.swap_buffers(ventana)
        glfw.poll_events()

    glDeleteVertexArrays(1, [VAO_e1])
    glDeleteBuffers(1, [VBO_e1])
    glDeleteBuffers(1, [EBO_e1])

    glDeleteVertexArrays(1, [VAO_e2])
    glDeleteBuffers(1, [VBO_e2])
    glDeleteBuffers(1, [EBO_e2])

    glDeleteProgram(programa_shader_1)
    glDeleteProgram(programa_shader_2)
    
glfw.terminate()

#usar las teclas para variar los valores
if __name__ == "__main__":
    main()
