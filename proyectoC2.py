""" 
Hecho por:
    - Sofía Victoria Marín
    - Sergio Danilo Palacios
    - Carlos Julián Morales

Resumen de las functiones:
    1.flor: modelado por curvas
    2.piso:
    3.cactus: superficies de recorrido
    4.Pecera:
    5.plato de abajo: superficies curvas

Adicional: Distintos colores con shaders

 """

import glfw
from OpenGL.GL import *
import glm
from glm import value_ptr
import numpy as np
import ctypes
from scipy.interpolate import splprep, splev
from geomdl import NURBS


codigo_shader_vertices = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 proyeccion;
uniform mat4 vista;
uniform mat4 modelo;
 
void main()
{
    gl_Position = proyeccion *  vista *  modelo * vec4(position, 1.0);
}
"""

col_verde_oscuro = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(0.23, 0.72, 0.26, 1);
}
"""

col_verde_musgo = """ 
#version 330 core
out vec4 color;
void main()
{
    color = vec4(0.08, 0.18, 0.18, 1);
}
"""

col_verde = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(0.0, 1.0, 0.0, 1.0);
}
"""

col_plato = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(0.69, 0.50, 0.27 , 1);
}
"""

col_blanco = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(1.0, 1.0, 1.0, 1.0);
}
"""

col_vidrio = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(0.45, 0.54, 0.70, 1);
}
"""

col_flor = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(0.82, 0.36, 0.61 , 1);
}
"""

""" SHADERS """
def compilarShader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader);
    glShaderSource(shader, codigo);
    glCompileShader(shader);

    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def crear_programa_shader(codigo_vertices, codigo_fragmentos):
    shader_vertices = compilarShader(codigo_vertices, GL_VERTEX_SHADER);
    shader_fragmentos = compilarShader(codigo_fragmentos, GL_FRAGMENT_SHADER);

    programa_shader = glCreateProgram();

    glAttachShader(programa_shader, shader_vertices);
    glAttachShader(programa_shader, shader_fragmentos);

    glLinkProgram(programa_shader);

    if glGetProgramiv(programa_shader, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(programa_shader));

    glDeleteShader(shader_vertices);
    glDeleteShader(shader_fragmentos);

    return programa_shader;

def configurar_buffers(vertices, indices, indexBool):
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    if indexBool:
        EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    if indexBool:
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    if indexBool:
        return VAO, VBO, EBO
    else:
        return VAO, VBO, []

""" MATEMÄTICA """

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

#Cactus

def generar_curva_spline(puntos_control, num_puntos=50):
    puntos_control= np.array(puntos_control)
    v,u = splprep([puntos_control[:,0], puntos_control[:,1]],k=4, s=0)
    u_fino = np.linspace(0,1, num_puntos)
    x,y = splev( u_fino, v)
    return np.vstack([x,y]).T

def generar_superficie_spline(curva, resolucion_angular=30,pos=[0,0,0],scale=1,rot=[0,0,0]):

    angulos = np.linspace(0, 2 * np.pi, resolucion_angular, endpoint=False)
    vertices = []
    indices = []
    num_puntos_curva = len(curva)
    
    for i, angulo in enumerate(angulos):
        cos_a = np.cos(angulo)
        sin_a = np.sin(angulo)

        for j, punto in enumerate(curva):
            x, y = punto
            z = x * sin_a
            x_rotado = x * cos_a
            vertices.append([x_rotado, y, z])

    vertices = aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertices,scale)),pos)

    for i in range(len(angulos)):
        next_i = (i + 1) % len(angulos)
        for j in range(num_puntos_curva - 1):
            indices.append(i * num_puntos_curva + j)
            indices.append(i * num_puntos_curva + j + 1)
            indices.append(next_i * num_puntos_curva + j)

            indices.append(i * num_puntos_curva + j + 1)
            indices.append(next_i * num_puntos_curva + j)
            indices.append(next_i * num_puntos_curva + j + 1)

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

def cactus(posicion,rotate,size):
    puntos_control = [
        [0.6, 0.1],   
        [1.635, 0.425],
        [0.025, 2.53],
        [-1.21, 0.355],
        [-0.315, -0.19]
            
    ]

    curva_spline = generar_curva_spline(puntos_control)
    vertices_superficie, indices_superficie = generar_superficie_spline(curva_spline,pos=posicion,rot=rotate,scale=size)
    
    return vertices_superficie, indices_superficie

#Pecera

def pecera(altura,pos=[0,0,0],scale=1,rot=[0,0,0]):
    vertices = []
    indices = []
    
    for i in range(10):
        y = i * altura / 9
        r = 1 + np.sin(np.deg2rad(i*20))*0.4

        for j in range(21):
            theta = j * 2.0 * np.pi / 20
            x = r * np.cos(theta)
            z = r * np.sin(theta)

            vertices.append([x,y,z])

    vertices = aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertices,scale)),pos)

    # Crear los indices
    for i in range(9):
        for j in range(20):
            v0 = i * (20 + 1) + j
            v1 = v0 + 1
            v2 = v0 + (20 + 1)
            v3 = v2 + 1

            # Triángulo 1
            indices.extend([v0, v1, v2])

            # Triángulo 2
            indices.extend([v1, v3, v2])

    vertices = np.array(vertices, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)
    return vertices, indices

#Plato
def generar_curva_nurbs(puntos_control, grado=3, num_puntos=100):
    
    curva = NURBS.Curve() 
    curva.degree = grado  
    curva.ctrlpts = puntos_control 
    curva.knotvector = np.linspace(0, 1, len(puntos_control) + grado + 1)
    puntos_curva = curva.evalpts

    return puntos_curva

def puntos(escala,M):
    puntosc = [
        [-8.0, 4.0, 0.0],    
        [1.7, -2.9, 0.39],
        [2.0, 4.0, 0.0],
        [4.6, -7.6, 0.5]
    ]
    puntos_escala = [(x * escala+M, y * escala, z * escala) for x, y, z in puntosc]
    return puntos_escala
    
def generar_superficie_revolucion_nurbs(curva, resolucion_angular=150):
    
    angulos = np.linspace(0, 2 * np.pi, resolucion_angular, endpoint=False)
    vertices = []

    for angulo in angulos:
        cos_a = np.cos(angulo)  
        sin_a = np.sin(angulo)  

        for punto in curva:
            x, y, z = punto  
            x_rotado = x * cos_a 
            z_rotado = x * sin_a

            vertices.append([x_rotado, y, z_rotado])

    return vertices

def plato(a,b,c,pos=[0,0,0],scale=1,rot=[0,0,0]):
    vertices=[]
    
    puntos_control=puntos(a,1)
    curva_nurbs = generar_curva_nurbs(puntos_control)
    vertices.append(generar_superficie_revolucion_nurbs(curva_nurbs))
   
    puntos_control=puntos(b, 0.6)
    curva_nurbs = generar_curva_nurbs(puntos_control)
    vertices.append(generar_superficie_revolucion_nurbs(curva_nurbs))
    
    puntos_control=puntos(c, 1)
    curva_nurbs = generar_curva_nurbs(puntos_control)
    vertices.append(generar_superficie_revolucion_nurbs(curva_nurbs))
    
    newVertices = []

    for vertexList in vertices:
        newVertices.append(aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertexList,scale)),pos))

    newVertices=np.array(newVertices, dtype=np.float32).reshape(-1, 3)
    
    return newVertices


#Flor

def generar_superficie_revolucion_nurbs2(curva, resolucion_angular=300):
    m=0
    angulos = np.linspace(0, 2 * np.pi, resolucion_angular, endpoint=False)

    # Lista para almacenar los vértices de la superficie generada
    vertices = []
        
    # Para cada ángulo, rotar los puntos de la curva alrededor del eje Y
    for angulo in angulos:
                     
     if m <  4:
        cos_a = np.cos(angulo)  # Rotar la coordenada x
        sin_a = np.sin(angulo)  # Rotar la coordenada z
        m+=1
        for punto in curva:
            x, y, z = punto  # Descomponer las coordenadas del punto actual
            x_rotado = x * cos_a 
            z_rotado = x * sin_a

            # Añadir el nuevo punto rotado a la lista de vértices
            vertices.append([x_rotado, y, z_rotado])
     elif m <  25:
         m+=1
         pass
     else:
         m=0
     
    # Convertir la lista de vértices a un array de tipo float32 y devolverlo
    return vertices

def superficie_nurbs(a,b,c,pos=[0,0,0],scale=1,rot=[0,0,0]):
    vertices=[]
    
    puntos_control=puntos(a,1)
    curva_nurbs = generar_curva_nurbs(puntos_control)
    vertices.append(generar_superficie_revolucion_nurbs(curva_nurbs))
   
    puntos_control=puntos(b, 0.6)
    curva_nurbs = generar_curva_nurbs(puntos_control)
    vertices.append(generar_superficie_revolucion_nurbs(curva_nurbs))
    
    puntos_control=puntos(c, 1)
    curva_nurbs = generar_curva_nurbs(puntos_control)
    vertices.append(generar_superficie_revolucion_nurbs(curva_nurbs))
    
    newVertices = []

    for vertexList in vertices:
        newVertices.append(aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertexList,scale)),pos))

    newVertices=np.array(newVertices, dtype=np.float32).reshape(-1, 3)
    
    return newVertices

# Suelo

def suelo(radio,pos=[0,0,0],scale=1,rot=[0,0,0]):
    # Generar el centro del disco
    vertices = [[0.0, 0.0, 0.0]]  # Vértice central

    # Generar los vértices en la circunferencia
    for i in range(15):
        theta = i * 2.0 * np.pi / 15  # Ángulo
        x = radio * np.cos(theta)
        y = radio * np.sin(theta)
        vertices.append([x, y, 0.0])  # Añadir el vértice al disco

    vertices = aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertices,scale)),pos)
    
    # Generar los índices
    indices = []
    for i in range(1, 15 + 1):
        indices.append(0)          # Índice del centro
        indices.append(i)          # Índice del vértice actual
        indices.append(i % 15 + 1)  # Índice del siguiente vértice

    vertices = np.array(vertices, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)

    return vertices, indices

""" Dibujar Elementos """

def dibujar(programa, VAO, vertices, indices, proyeccion,modelo,vista,elementsBool,style):
    glUseProgram(programa)
    
    glUniformMatrix4fv(glGetUniformLocation(programa, "proyeccion"), 1, GL_FALSE, value_ptr(proyeccion))
    glUniformMatrix4fv(glGetUniformLocation(programa, "vista"), 1, GL_FALSE, value_ptr(vista))
    glUniformMatrix4fv(glGetUniformLocation(programa, "modelo"), 1, GL_FALSE, value_ptr(modelo))

    glBindVertexArray(VAO)

    if elementsBool:
        glDrawElements(style, len(indices), GL_UNSIGNED_INT, None)
    else:
        glDrawArrays(style, 0, len(vertices))
    glBindVertexArray(0)

def main():
    #vertices_superficie, indices_superficie = pecera(1.7,pos=[1,0,1],scale=0.3,rot=[45,45,10])
    #vertices_superficie, indices_superficie = cactus([-0.4,0.3,0],[90,0,45],0.5)
    #vertices_superficie, indices_superficie = plato(1,0.6,-0.2,[1,0,1],0.3,[45,45,10]) , []
    #vertices_superficie, indices_superficie = superficie_nurbs(1,0.6,-0.2,[1,0,1],0.3,[45,45,10]) , []
    
    # VERTICES DE FIGURAS
    vertices_agua1, indices_agua1 = pecera(1.7,scale=1.2)
    vertices_agua2, indices_agua2 = pecera(1.7,rot=[0,15,0])

    vertices_plato, indices_plato = plato(1,0.6,-0.2,[0,-0.5,0],0.4,[0,0,0]) , []
    vertices_pecera, indices_pecera = pecera(1.7,scale=1.3)
    vertices_suelo, indices_suelo = suelo(1.4,rot=[90,0,0])
    
    vertices_cactus1, indices_cactus1 = cactus([0,0,0.6],[0,0,0],0.2)
    vertices_cactus2, indices_cactus2 = cactus([-0.4,0,0.3],[0,0,0],0.3)
    vertices_cactus3, indices_cactus3 = cactus([0.2,0,-0.6],[0,0,0],0.5)

    vertices_flor1, indices_flor1 = superficie_nurbs(1,0.6,-0.2,[0.5,0.2,0.5],0.1,[45,45,10]) , []
    vertices_flor2, indices_flor2 = superficie_nurbs(1,0.6,-0.2,[-0.2,1,0.2],0.07,[45,-55,10]) , []


    # GLFW
    if not glfw.init():
        print("Ayuda")
        return None

    ventana = glfw.create_window(800, 600, "Proyecto - Corte 2", None, None)

    if not ventana: 
        glfw.terminate()
        print (":(")
        return

    glfw.make_context_current(ventana)
    glEnable(GL_DEPTH_TEST)
    
    proyeccion = glm.perspective(glm.radians(45), 800 / 600, 1, 50)
    vista = glm.translate(glm.mat4(1.0), glm.vec3(0.0, -1.0, -5.0))
    modelo = glm.rotate(glm.mat4(1.0), 2*np.pi, glm.vec3(0.0, 1.0, 0.0))

    # SHADERS
    programa_blanco = crear_programa_shader(codigo_shader_vertices, col_blanco)
    programa_plato = crear_programa_shader(codigo_shader_vertices, col_plato)
    programa_verde = crear_programa_shader(codigo_shader_vertices, col_verde)
    programa_verde_oscuro = crear_programa_shader(codigo_shader_vertices, col_verde_oscuro)
    programa_verde_musgo = crear_programa_shader(codigo_shader_vertices, col_verde_musgo)
    programa_vidrio = crear_programa_shader(codigo_shader_vertices, col_vidrio)
    programa_flor = crear_programa_shader(codigo_shader_vertices, col_flor)


    # BUFFERS
    #VAO, VBO, EBO = configurar_buffers(vertices_superficie, indices_superficie,True)
    #VAO, VBO, EBO = configurar_buffers(vertices_superficie, indices_superficie,False)

    VAO_agua1, VBO_agua1, EBO_agua1 = configurar_buffers(vertices_agua1, indices_agua1,True)
    VAO_agua2, VBO_agua2, EBO_agua2 = configurar_buffers(vertices_agua2, indices_agua2,True)

    VAO_plato, VBO_plato, EBO_plato = configurar_buffers(vertices_plato, indices_plato,False)
    VAO_pecera, VBO_pecera, EBO_pecera = configurar_buffers(vertices_pecera, indices_pecera,True)
    VAO_suelo, VBO_suelo, EBO_suelo = configurar_buffers(vertices_suelo,indices_suelo,True)

    VAO_cactus1, VBO_cactus1, EBO_cactus1 = configurar_buffers(vertices_cactus1, indices_cactus1,True)
    VAO_cactus2, VBO_cactus2, EBO_cactus2 = configurar_buffers(vertices_cactus2, indices_cactus2,True)
    VAO_cactus3, VBO_cactus3, EBO_cactus3 = configurar_buffers(vertices_cactus3, indices_cactus3,True)

    VAO_flor1, VBO_flor1, EBO_flor1 = configurar_buffers(vertices_flor1, indices_flor1,False)
    VAO_flor2, VBO_flor2, EBO_flor2 = configurar_buffers(vertices_flor2, indices_flor2,False)

    while not glfw.window_should_close(ventana):
        glClearColor(0.3, 0.3, 0.5, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #dibujar(programa_shader, VAO, vertices_superficie, indices_superficie,proyeccion, modelo, vista, True, GL_LINES)
        #dibujar(programa_shader, VAO, vertices_superficie, indices_superficie,proyeccion, modelo, vista, False, GL_LINES)

        dibujar(programa_blanco, VAO_agua1, vertices_agua1, indices_agua1,proyeccion, modelo, vista, True, GL_POINTS)
        dibujar(programa_blanco, VAO_agua2, vertices_agua2, indices_agua2,proyeccion, modelo, vista, True, GL_POINTS)

        dibujar(programa_plato, VAO_plato, vertices_plato, indices_plato,proyeccion, modelo, vista, False, GL_LINE_LOOP)
        dibujar(programa_vidrio, VAO_pecera, vertices_pecera, indices_pecera,proyeccion, modelo, vista, True, GL_LINES)
        dibujar(programa_verde_musgo, VAO_suelo, vertices_suelo, indices_suelo,proyeccion, modelo, vista, True, GL_TRIANGLE_STRIP)
        
        dibujar(programa_verde, VAO_cactus1, vertices_cactus1, indices_cactus1,proyeccion, modelo, vista, True, GL_LINES)
        dibujar(programa_verde_oscuro, VAO_cactus2, vertices_cactus2, indices_cactus2,proyeccion, modelo, vista, True, GL_LINES)
        dibujar(programa_verde, VAO_cactus3, vertices_cactus3, indices_cactus3,proyeccion, modelo, vista, True, GL_LINES)

        dibujar(programa_flor, VAO_flor1, vertices_flor1, indices_flor1,proyeccion, modelo, vista, False, GL_LINE_LOOP)
        dibujar(programa_flor, VAO_flor2, vertices_flor2, indices_flor2,proyeccion, modelo, vista, False, GL_LINE_LOOP)

        glfw.swap_buffers(ventana)
        glfw.poll_events()

    glDeleteProgram(programa_blanco)

    # AGUA
    glDeleteVertexArrays(1, [VAO_agua1])
    glDeleteBuffers(1, [VBO_agua1])
    glDeleteBuffers(1, [EBO_agua1])

    glDeleteVertexArrays(1, [VAO_agua2])
    glDeleteBuffers(1, [VBO_agua2])
    glDeleteBuffers(1, [EBO_agua2])

    # MISELANEOS
    glDeleteVertexArrays(1, [VAO_plato])
    glDeleteBuffers(1, [VBO_plato])
    glDeleteBuffers(1, [EBO_plato])

    glDeleteVertexArrays(1, [VAO_pecera])
    glDeleteBuffers(1, [VBO_pecera])
    glDeleteBuffers(1, [EBO_pecera])

    glDeleteVertexArrays(1, [VAO_suelo])
    glDeleteBuffers(1, [VBO_suelo])
    glDeleteBuffers(1, [EBO_suelo])

    # CACTUS
    glDeleteVertexArrays(1, [VAO_cactus1])
    glDeleteBuffers(1, [VBO_cactus1])
    glDeleteBuffers(1, [EBO_cactus1])

    glDeleteVertexArrays(1, [VAO_cactus2])
    glDeleteBuffers(1, [VBO_cactus2])
    glDeleteBuffers(1, [EBO_cactus2])
    
    glDeleteVertexArrays(1, [VAO_cactus3])
    glDeleteBuffers(1, [VBO_cactus3])
    glDeleteBuffers(1, [EBO_cactus3])
    
    # FLORES
    glDeleteVertexArrays(1, [VAO_flor1])
    glDeleteBuffers(1, [VBO_flor1])
    glDeleteBuffers(1, [EBO_flor1])

    glDeleteVertexArrays(1, [VAO_flor2])
    glDeleteBuffers(1, [VBO_flor2])
    glDeleteBuffers(1, [EBO_flor2])

    glfw.terminate()

if __name__ == "__main__":
    main()