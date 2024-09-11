# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:49:27 2024

@author: Sofyv
"""

import glfw 
from OpenGL.GL import * 
import numpy as np 
import ctypes 

codigo_shader_vertices = """
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

codigo_shader_fragmentos = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(1.5, 0.1, 0.5, 0.5);
    
}
"""

def compilar_shader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader) 
    glShaderSource(shader, codigo) 
    glCompileShader(shader) 
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader 

def crear_programa_shader():
    shader_vertices = compilar_shader(codigo_shader_vertices, GL_VERTEX_SHADER)
    shader_fragmentos = compilar_shader(codigo_shader_fragmentos, GL_FRAGMENT_SHADER)

    programa_shader = glCreateProgram()
    glAttachShader(programa_shader, shader_vertices)
    glAttachShader(programa_shader, shader_fragmentos)
    glLinkProgram(programa_shader)
    if glGetProgramiv(programa_shader, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(programa_shader))

    glDeleteShader(shader_vertices)
    glDeleteShader(shader_fragmentos)
    
    return programa_shader 


def generar_prismaHex(h,r):

    vertices = []
    posz = 0.4
    for  j in [-posz,-(posz+1) ]:
      for i in range(6):
        
        teta =2* np.pi * i/ 6
        x= r*np.cos(teta)
        y= r*np.sin(teta)
        z= j*h/2
        vertices.append([x,y,z])
       
        
    vertices = np.array (vertices, dtype=np.float32)
    
    indices = []
    
    for i in range (6):
        indices.append([i,(i+1)%6])
        
        
    for i in range(6):
        indices.append([i+6, (i+1) % 6 +6])
        
    for i in range (6):
        indices.append([i,i+6])

    indices = np.array (indices, dtype=np.uint32). flatten()
  
    
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

def teclado(ventana,tecla, cescan, accion, modifica):
    global usar_proyeccion
    if tecla == glfw.KEY_P and accion == glfw.PRESS:
        usar_proyeccion = not usar_proyeccion
        print(f"usar_proyeccion: {usar_proyeccion}")

def main():
    
    global usar_proyeccion
    usar_proyeccion=True
    
    if not glfw.init():
        return

    a,b =800,600
    ventana = glfw.create_window(a, b, "SI SE PUEDE", None, None)
    if not ventana: 
        glfw.terminate()
        return

    glfw.make_context_current(ventana)
    #glfw.set_key_callback(ventana, teclado)
  
    altura = 0.4
    radio = 0.3
    
    vertices, indices = generar_prismaHex( altura, radio)
    
   
    
    
    try:
       
        programa_shader = crear_programa_shader()

      
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
        
        ###
        
        uvista= glGetUniformLocation(programa_shader, "vista")
        uproyeccion= glGetUniformLocation(programa_shader, "proyeccion")
        umodelo= glGetUniformLocation(programa_shader, "transformacion")
        

        
        while not glfw.window_should_close(ventana):
            
            glClearColor(0.2 ,0.2 ,0.2 ,0.1 )
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glUseProgram(programa_shader)
            
            
            ###
            
            if usar_proyeccion:
                vista=matriz_vista([-0.15,0.2,2],[0,0,0],[0,1,0])
                proyeccion= matriz_perspectiva(45, a/b, 1, 5)
                glUniformMatrix4fv(uvista,1,GL_FALSE, vista.flatten())
                glUniformMatrix4fv(uproyeccion,1,GL_FALSE,proyeccion.flatten())
                
            else:
                glUniformMatrix4fv(uvista,1,GL_FALSE,np.identity(4).flatten())
                glUniformMatrix4fv(uproyeccion,1,GL_FALSE, np.identity(4).flatten())
            
            
            modelo_prisma = np.identity(4)
            modelo_prisma[0,3] =0.5
            glUniformMatrix4fv(umodelo, 1, GL_FALSE, modelo_prisma.flatten())
            
                
            glBindVertexArray(VAO)
            glDrawElements(GL_LINES, len(indices), GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            
            glfw.swap_buffers(ventana)
            glfw.poll_events()

    except Exception as e:
        print(f"Ocurrió un error: {e}")
    
    finally:
       
        glDeleteVertexArrays(1, [VAO])
        glDeleteBuffers(1, [VBO])
        glDeleteBuffers(1, [EBO])
        glDeleteProgram(programa_shader)
        
        
        glfw.terminate()

#usar las teclas para variar los valores
if __name__ == "__main__":
    main()

