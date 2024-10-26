"""
Hecho por: Sergio Danilo Palacios
Codigo: 6000806

Utilizar las teclas correspondientes a los números [1] [2] [3] y [4] para cambiar de visualización teniendo en cuenta:
    1. Sin proyección ni vista.
    2. Con proyección de perspectiva y vista UVN
    3. Con proyección ortogonal y vista Look-At
    4. Con proyección oblicua y vista Look-A

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
    color = vec4(0.0, 0.5, 0.0, 1.0);
    
}
"""

Codigo_shaderFragmentos3 = """
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
    glUniformMatrix4fv(umodelo, 1, GL_FALSE, modelo_figura.flatten())
    glBindVertexArray(VAO)
    glDrawElements(GL_LINES, len(indices), GL_UNSIGNED_INT, None)
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
    vertices_e1, indices_e1 = elipsoide(0.2,10,10)
    vertices_e2, indices_e2 = elipsoide(0.05,10,10,rot=[0,0,90])
    vertices_orb, indices_orb = elipsoide(0.05,10,10,rot=[0,0,45],scale=1.3)
    
    try:
        programa_shader_1 = crear_programa_shader(Codigo_shaderFragmentos1)
        programa_shader_2 = crear_programa_shader(Codigo_shaderFragmentos2)
        programa_shader_3 = crear_programa_shader(Codigo_shaderFragmentos3)

        VAO_e1, VBO_e1, EBO_e1 = config_Buffers(vertices_e1,indices_e1)
        VAO_e2, VBO_e2, EBO_e2 = config_Buffers(vertices_e2,indices_e2)
        VAO_orb, VBO_orb, EBO_orb = config_Buffers(vertices_orb,indices_orb)
        
        uvista= glGetUniformLocation(programa_shader_1, "vista")
        uproyeccion= glGetUniformLocation(programa_shader_1, "proyeccion")
        umodelo= glGetUniformLocation(programa_shader_1, "transformacion")
        
        while not glfw.window_should_close(ventana):
            glClearColor(0.3 ,0.2 ,0.3 ,1.0 )
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            dibujarFigura(indices_e1,VAO_e1,programa_shader_1,uvista,np.identity(4),uproyeccion,np.identity(4),umodelo,np.identity(4))
            dibujarFigura(indices_e2,VAO_e2,programa_shader_2,uvista,np.identity(4),uproyeccion,np.identity(4),umodelo,np.identity(4))
            dibujarFigura(indices_orb,VAO_orb,programa_shader_3,uvista,np.identity(4),uproyeccion,np.identity(4),umodelo,np.identity(4))

            glfw.swap_buffers(ventana)
            glfw.poll_events()

    except Exception as e:
        print(f"Ocurrió un error: {e}")
    
    finally:
    
        glDeleteVertexArrays(1, [VAO_e1])
        glDeleteVertexArrays(1, [VAO_e2])
        glDeleteVertexArrays(1, [VAO_orb])
        glDeleteBuffers(1, [VBO_e1])
        glDeleteBuffers(1, [VBO_e2])
        glDeleteBuffers(1, [VBO_orb])
        glDeleteBuffers(1, [EBO_e1])
        glDeleteBuffers(1, [EBO_e2])
        glDeleteBuffers(1, [EBO_orb])
        glDeleteProgram(programa_shader_1)
        glDeleteProgram(programa_shader_2)
        glDeleteProgram(programa_shader_3)
        
    glfw.terminate()

#usar las teclas para variar los valores
if __name__ == "__main__":
    main()


