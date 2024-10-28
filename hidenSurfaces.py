"""
Hecho por: Sergio Danilo Palacios
Codigo: 6000806

Utilizar las tecla SPACE para cambiar entre DEPTH TEST y CULL FACE
"""

import glfw 
from OpenGL.GL import * 
import numpy as np 
import ctypes 
import glm
from glm import value_ptr

Codigo_shaderVertices = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color_vertx;

out vec3 randColor;

uniform mat4 vista;
uniform mat4 proyeccion;
uniform mat4 transformacion;

void main()
{
    gl_Position = vec4(position, 1.0)* vista* proyeccion * transformacion;
    randColor = color_vertx;
}
"""

Codigo_shaderFragmentos1 = """
#version 330 core

in vec3 randColor;
out vec4 color;

void main()
{
    color = vec4(randColor, 1.0);
    
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
    VBO = glGenBuffers(2)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0,None)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    
    glBindVertexArray(0)
    
    return VAO, VBO, EBO

# Transformaciones

def aux_multiplyMatrixes(A, B):
    # Verificar si las matrices pueden multiplicarse (el número de columnas de A debe ser igual al número de filas de B)
    if len(A[0]) != len(B):
        raise ValueError("El número de columnas de A debe ser igual al número de filas de B")

    # Inicializar la matriz resultante con ceros, de tamaño m x p
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    # Realizar la multiplicación de matrices
    for i in range(len(A)):  # Iterar sobre las filas de A
        for j in range(len(B[0])):  # Iterar sobre las columnas de B
            for k in range(len(B)):  # Iterar sobre las filas de B
                result[i][j] += A[i][k] * B[k][j]

    return result

def aux_rotacion(x,y,z,vertices):
    matX = [[1, 0, 0],[0, np.cos(x), -np.sin(x)],[0, np.sin(x), np.cos(x)]]
    matY = [[np.cos(y), 0, np.sin(y)],[0, 1, 0],[-np.sin(y), 0, np.cos(y)]]
    matZ = [[np.cos(z), -np.sin(z), 0],[np.sin(z), np.cos(z), 0],[0, 0, 1]]
    
    matRot = aux_multiplyMatrixes(aux_multiplyMatrixes(matZ,matY),matX)

    newVertices = []

    for punto in vertices:
        newPunto = [0,0,0]
        for i in range(3):
            for j in range(3):
                newPunto[i] += matRot[i][j] * punto[j]
        newVertices.append(newPunto)

    return newVertices

def aux_traslacion(vertices,delta):
    new_vertices = []

    for punto in vertices:
        x = punto[0] + delta [0]
        y = punto[1] + delta [1]
        z = punto[2] + delta [2]
        new_vertices.append([x,y,z])
    
    return new_vertices

def aux_escalado(vertices,scale):
    new_vertices = []

    for punto in vertices:
        x = punto[0] * scale
        y = punto[1] * scale
        z = punto[2] * scale
        new_vertices.append([x,y,z])
    
    return new_vertices

#FIGURAS

def elipsoide(radio,nstack,nsectors,pos=[0,0,0],scale=1,rot=[0,0,0]):
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
            vertices.append([x*1.6,y,z])

            if i < nstack and j < nsectors:
                first = i * (nsectors + 1) + j
                second = first + nsectors + 1
                indices.append(first)
                indices.append(second)
                indices.append(first + 1)
                indices.append(second)
                indices.append(second + 1)
                indices.append(first + 1)

    vertices = aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertices,scale)),pos)

    vertices = np.array(vertices, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices


#OPENGL

def dibujarFigura(indices,VAO,shader,uvista,modelo_vista,uproyeccion,modelo_proyeccion,umodelo,modelo_figura):
    glUseProgram(shader)
    glUniformMatrix4fv(uvista, 1, GL_FALSE, modelo_vista.flatten())        
    glUniformMatrix4fv(uproyeccion,1,GL_FALSE,modelo_proyeccion.flatten())
    glUniformMatrix4fv(umodelo, 1, GL_FALSE, modelo_figura)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLE_STRIP, len(indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

def main():    

    depthSurfaces = True

    if not glfw.init():
        return

    a,b =800,600
    ventana = glfw.create_window(a, b, "Proyecciones - Sergio Palacios", None, None)
    
    if not ventana: 
        glfw.terminate()
        return

    glfw.make_context_current(ventana)

    #vertices, indices = generar_prismaHex( altura, radio)
    vertices_e1, indices_e1 = elipsoide(0.4,10,10, pos=[0,0,-1])
    vertices_e2, indices_e2 = elipsoide(0.2,10,10,rot=[45,0,90], pos=[0,0,-0.5])
    vertices_orb, indices_orb = elipsoide(0.05,10,10,rot=[90,0,45])
    
    try:
        programa_shader_1 = crear_programa_shader(Codigo_shaderFragmentos1)

        VAO_e1, VBO_e1, EBO_e1 = config_Buffers(vertices_e1,indices_e1)
        VAO_e2, VBO_e2, EBO_e2 = config_Buffers(vertices_e2,indices_e2)
        VAO_orb, VBO_orb, EBO_orb = config_Buffers(vertices_orb,indices_orb)
        
        uvista= glGetUniformLocation(programa_shader_1, "vista")
        uproyeccion= glGetUniformLocation(programa_shader_1, "proyeccion")
        umodelo= glGetUniformLocation(programa_shader_1, "transformacion")
        
        while not glfw.window_should_close(ventana):
            glClearColor(0.3 ,0.2 ,0.3 ,1.0 )
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            rotacion = glm.rotate(glm.mat4(1.0),glm.radians(glfw.get_time()*30),glm.vec3(0.0,1.0,0.0))

            dibujarFigura(indices_e1,VAO_e1,programa_shader_1,uvista,np.identity(4),uproyeccion,np.identity(4),umodelo,value_ptr(rotacion))
            dibujarFigura(indices_e2,VAO_e2,programa_shader_1,uvista,np.identity(4),uproyeccion,np.identity(4),umodelo,value_ptr(rotacion))
            dibujarFigura(indices_orb,VAO_orb,programa_shader_1,uvista,np.identity(4),uproyeccion,np.identity(4),umodelo,value_ptr(rotacion))

            if depthSurfaces:
                glDisable(GL_CULL_FACE)
                glEnable(GL_DEPTH_TEST)
            else:
                glDisable(GL_DEPTH_TEST)
                glEnable(GL_CULL_FACE)

            if glfw.get_key(ventana,glfw.KEY_A) == glfw.PRESS:
                depthSurfaces = not depthSurfaces
                if depthSurfaces:
                    print("GL DEPTH TEST")
                else:
                    print("GL CULL_FACE")

            glfw.swap_buffers(ventana)
            glfw.poll_events()

    except Exception as e:
        print(f"Ocurrió un error: {e}")
    
    finally:
    
        glDeleteVertexArrays(1, [VAO_e1])
        glDeleteVertexArrays(1, [VAO_e2])
        glDeleteVertexArrays(1, [VAO_orb])
        glDeleteBuffers(2, VBO_e1)
        glDeleteBuffers(2, VBO_e2)
        glDeleteBuffers(2, VBO_orb)
        glDeleteBuffers(1, [EBO_e1])
        glDeleteBuffers(1, [EBO_e2])
        glDeleteBuffers(1, [EBO_orb])
        glDeleteProgram(programa_shader_1)
        
    glfw.terminate()

#usar las teclas para variar los valores
if __name__ == "__main__":
    main()


