import glfw
import traceback
import glm
from OpenGL.GL import *
import numpy as np
import random as rng

global_animate_cubo = False
global_animate_man = False 
global_animate_players = False 
global_animate_sombr1illa = False   
global_animate_time = True

global_camara_look = False
global_camara_ortg = False
global_camara_proy = True

PosicionesA = [
    [ 0.1 , 0.2 , 1.0],
    [-0.01 , 0.2 , 0.2],
    [-0.4 , 0.2 , 0.4],
    [ 0.4 , 0.2 , 0.3],
]

PosicionesB = [
    [-0.02 , 0.2 , 0.3],
    [ 0.2 , 0.2 , 1.1],
    [ 0.3 , 0.2 , 0.3],
    [-0.3 , 0.2 , 0.4],
]

playersPos = PosicionesA

phong_vertexCode = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 posicion_frag;
out vec3 normal_frag;

uniform mat4 proyeccion;
uniform mat4 vista;
uniform mat4 transformacion; 

void main()
{
    gl_Position =  proyeccion * vista * transformacion * vec4(position, 1.0);
    posicion_frag = vec3(transformacion * vec4(position,1.0));
    normal_frag = mat3(transpose(inverse(transformacion))) * normal;
}
"""
gouraud_vertexCode = """
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

"""

phong_fragmentCode = """
#version 330 core

in vec3 posicion_frag;
in vec3 normal_frag;

out vec4 color;

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
    vec3 normal = normalize(normal_frag);
    vec3 direccion_luz = normalize(luz_position - posicion_frag);

    // Componente ambiental
    vec3 componente_ambiental = luz_ambient * mat_ambient;

    // Componente difusa
    float intensidad_difusa = max(dot(normal, direccion_luz), 0.0);
    vec3 componente_difusa = luz_difuse * (intensidad_difusa * mat_difuse);

    // Componente especular
    vec3 direccion_vista = normalize(-posicion_frag);
    vec3 direccion_reflejo = reflect(-direccion_luz, normal);
    float intensidad_especular = pow(max(dot(direccion_vista, direccion_reflejo), 0.0), mat_brillo);
    vec3 componente_especular = luz_specular * (intensidad_especular * mat_specular);

    // Color final
    vec3 color_final = componente_ambiental + componente_difusa + componente_especular;
    color = vec4(color_final, 1.0);
}
"""
gouaud_fragmentCode = """
#version 330 core

in vec3 color_frag;

out vec4 color;

void main()
{
    color = vec4(color_frag,1.0);
}
"""

class figura:

    def __init__(self,VAO,indices,shaders,luz,material):
        self.VAO = VAO
        self.indices = indices
        self.programa = shaders["program"]
        self.luz = luz
        self.material = material

    def dibujar(self,drawTipe,projeccion,vista,transform=glm.mat4(1.0)):
        glUseProgram(self.programa)

        glUniform3f(glGetUniformLocation(self.programa,"luz_position"),self.luz["pos"][0],self.luz["pos"][1],self.luz["pos"][2]) 
        glUniform3f(glGetUniformLocation(self.programa,"luz_ambient"),self.luz["amb"][0],self.luz["amb"][1],self.luz["amb"][2])  
        glUniform3f(glGetUniformLocation(self.programa,"luz_difuse"),self.luz["dif"][0],self.luz["dif"][1],self.luz["dif"][2])   
        glUniform3f(glGetUniformLocation(self.programa,"luz_specular"),self.luz["spc"][0],self.luz["spc"][1],self.luz["spc"][2]) 

        glUniform3f(glGetUniformLocation(self.programa,"mat_ambient"),self.material["amb"][0],self.material["amb"][0],self.material["amb"][0]) 
        glUniform3f(glGetUniformLocation(self.programa,"mat_difuse"),self.material["dif"][0],self.material["dif"][0],self.material["dif"][0])     
        glUniform3f(glGetUniformLocation(self.programa,"mat_specular"),self.material["spc"][0],self.material["spc"][0],self.material["spc"][0])   
        glUniform1f(glGetUniformLocation(self.programa,"mat_brillo"),self.material["brillo"])           

        glUniformMatrix4fv(glGetUniformLocation(self.programa,"transformacion"),1,GL_FALSE,glm.value_ptr(transform))
        glUniformMatrix4fv(glGetUniformLocation(self.programa,"proyeccion"),1,GL_FALSE,glm.value_ptr(projeccion))
        glUniformMatrix4fv(glGetUniformLocation(self.programa,"vista"),1,GL_FALSE,glm.value_ptr(vista))

        glBindVertexArray(self.VAO)
        glDrawElements(drawTipe, self.indices, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

""" COMPILAR SHADER """

def compilar_shader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader) 
    glShaderSource(shader, codigo) 
    glCompileShader(shader) 
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader 

def crear_programa_shader(frag_Code,vertx_code):
    shader_vertices = compilar_shader(vertx_code, GL_VERTEX_SHADER)
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

def config_buffers(vertices,normales,indices):
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(2)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER,VBO[0])
    glBufferData(GL_ARRAY_BUFFER,vertices.nbytes,vertices,GL_STATIC_DRAW)
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER,VBO[1])
    glBufferData(GL_ARRAY_BUFFER,normales.nbytes,normales,GL_STATIC_DRAW)
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,0,None)
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,GL_STATIC_DRAW)  
    glBindBuffer(GL_ARRAY_BUFFER,0)

    glBindVertexArray(0)

    return VAO,VBO,EBO

def teclado(ventana,tecla, cescan, accion, modifica):
    global global_animate_time
    global global_animate_sombr1illa
    global global_animate_cubo
    global global_animate_man
    global global_animate_players

    global global_camara_look
    global global_camara_ortg
    global global_camara_proy

    if tecla == glfw.KEY_A and accion == glfw.PRESS:
        global_animate_cubo = True
        print("Cubo Pasamdo")
    
    if tecla == glfw.KEY_S and accion == glfw.PRESS:
        global_animate_man = not global_animate_man
        print("Man Pasamdo")
    
    if tecla == glfw.KEY_D and accion == glfw.PRESS:
        global_animate_players = not global_animate_players
        print("Cambio de Formacion")
    
    if tecla == glfw.KEY_F and accion == glfw.PRESS:
        global_animate_sombr1illa = not global_animate_sombr1illa
        print("Paraguas on")
    
    if tecla == glfw.KEY_G and accion == glfw.PRESS:
        global_animate_time = not global_animate_time
        print("ZA WARDO")

    if tecla == glfw.KEY_1 and accion == glfw.PRESS:
        global_camara_look = True
        global_camara_ortg = False
        global_camara_proy = False
        print("Vista Look at")
    
    if tecla == glfw.KEY_2 and accion == glfw.PRESS:
        global_camara_look = False
        global_camara_ortg = True
        global_camara_proy = False
        print("Vista Ortogonal")
    
    if tecla == glfw.KEY_3 and accion == glfw.PRESS:
        global_camara_look = False
        global_camara_ortg = False
        global_camara_proy = True
        print("Vista Proyeccion")

""" Auxiliares """

def curva_pot(y,a):
    return y*a

def trayectoria(inicio,fin,delta):
    position = [inicio[0]*(1-delta)+fin[0]*delta,inicio[1]*(1-delta)+fin[1]*delta,inicio[2]*(1-delta)+fin[2]*delta]
    return position

def convert2vec3(array):
    return glm.vec3(array[0],array[1],array[2])

def randomNumber(last,maximo):
    aux = rng.randint(0,maximo)
    while aux==last:
        aux = rng.randint(0,maximo)
    return aux 
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

def transformacion(vertices,delta,scale):
    vertices_1 = []
    vertices_2 = []

    for punto in vertices:
        x = punto[0] * scale
        y = punto[1] * scale
        z = punto[2] * scale
        vertices_1.append([x,y,z])

    for punto in vertices_1:
        x = punto[0] + delta [0]
        y = punto[1] + delta [1]
        z = punto[2] + delta [2]
        vertices_2.append([x,y,z])

    return vertices_2

""" FIGURAS Y TRANSFORMACIONES """

def generate_cube(position=[0.0,0.0,0.0],scale=1.0):
    # Definir vértices para un cubo de tamaño 1x1x1 centrado en el origen
    vertices = [
        # Cara frontal
        [-0.5, -0.5,  0.5],  # Vértice 0
         [0.5, -0.5,  0.5],  # Vértice 1
         [0.5,  0.5,  0.5],  # Vértice 2
        [-0.5,  0.5,  0.5],  # Vértice 3
        # Cara trasera
        [-0.5, -0.5, -0.5],  # Vértice 4
         [0.5, -0.5, -0.5],  # Vértice 5
         [0.5,  0.5, -0.5],  # Vértice 6
        [-0.5,  0.5, -0.5]   # Vértice 7
    ]

    vertices = transformacion(vertices,position,scale)
    
    # Definir normales para cada cara del cubo
    normals = [
        # Cara frontal (0, 0, 1)
        0.0, 0.0, 1.0,  # Normal para el vértice 0
        0.0, 0.0, 1.0,  # Normal para el vértice 1
        0.0, 0.0, 1.0,  # Normal para el vértice 2
        0.0, 0.0, 1.0,  # Normal para el vértice 3
        # Cara trasera (0, 0, -1)
        0.0, 0.0, -1.0, # Normal para el vértice 4
        0.0, 0.0, -1.0, # Normal para el vértice 5
        0.0, 0.0, -1.0, # Normal para el vértice 6
        0.0, 0.0, -1.0  # Normal para el vértice 7
    ]
    
    # Definir índices para los triángulos de cada cara del cubo
    indices = [
        # Cara frontal
        0, 1, 2,  2, 3, 0,
        # Cara trasera
        4, 5, 6,  6, 7, 4,
        # Cara izquierda
        4, 7, 3,  3, 0, 4,
        # Cara derecha
        1, 5, 6,  6, 2, 1,
        # Cara superior
        3, 2, 6,  6, 7, 3,
        # Cara inferior
        0, 1, 5,  5, 4, 0
    ]



    vertices = np.array(vertices, dtype=np.float32).flatten()
    normals = np.array(normals, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)

    return vertices,normals,indices

def generate_piramide(position=[0.0,0.0,0.0],scale=1.0,rot=[0,0,0]):

    vertices = [
        [ 0.5, 0.0, 0.5],
        [ 0.5, 0.0,-0.5],
        [-0.5, 0.0,-0.5],
        [-0.5, 0.0, 0.5],
        [ 0.0, 0.5, 0.0]
    ]

    vertices = aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertices,scale)),position)
    
    indices = [
        0,1,2, 2,3,0,    #Base
        0,1,4,
        1,2,4,
        2,3,4,
        3,0,4
    ]

    normals =[
         1.0, 0.0, 0.0,
         1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0,
        -1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
    ]

    vertices = np.array(vertices, dtype=np.float32).flatten()
    normals = np.array(normals, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)

    return vertices,normals,indices

def generate_plano(position=[0.0,0.0,0.0],scale=1.0):
    vertices = [
        [ 0.5, 0.0, 0.5],
        [ 0.5, 0.0,-0.5],
        [-0.5, 0.0,-0.5],
        [-0.5, 0.0, 0.5]
    ]
    
    vertices = transformacion(vertices,position,scale)

    indices = [
        0,1,2, 2,3,0
    ]

    normals = [
        0.0,1.0,0.0,
        0.0,1.0,0.0,
        0.0,1.0,0.0,
        0.0,1.0,0.0
    ]

    vertices = np.array(vertices, dtype=np.float32).flatten()
    normals = np.array(normals, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)

    return vertices,normals,indices

def generate_icosaedro(position=[0.0,0.0,0.0],scale=1.0):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    prop = scale / np.sqrt(t**2 + 1)
    t *= prop

    vertices = [
        [-prop,  t,  0], [ prop,  t,  0], [-prop, -t,  0], [ prop, -t,  0],
        [ 0, -prop,  t], [ 0,  prop,  t], [ 0, -prop, -t], [ 0,  prop, -t],
        [ t,  0, -prop], [ t,  0,  prop], [-t,  0, -prop], [-t,  0,  prop]
    ]

    vertices = transformacion(vertices,position,scale)

    indices = [
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
        1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
        3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
        4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
    ]

    vertices = np.array(vertices, dtype=np.float32).flatten()
    normals = np.array(vertices, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)

    return vertices,normals,indices

def generate_superE(r, position, scale, rot):
    vertices =[]
    indices = []
    u = np.linspace(0,2*np.pi,100, endpoint=False) 
    v = np.linspace (0,3,40) 

    
    for j in range (len(v)):
        for  i in range (len(u)):
            
            x = r * np.cos(u[i])
            z = r*np.sin(u[i])
            y = v[j] 
            
            vertices.append([x,y,z])

    vertices = aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertices,scale)),position)

    numu =len (u)
    for j in range (len(v)-1):
        for i in range (numu):
            indices.append(j*numu+i)
            indices.append ((j+1)* numu +i)
        indices.append(j * numu)
        indices.append((j+1)*numu)

    vertices = np.array(transformacion(vertices, position, scale), dtype = np.float32)
    indices =np.array(indices, dtype=np.uint32)

    return vertices, np.array(vertices, dtype=np.float32), indices

def generate_rectangular_mesh(corner1, corner2, corner3, corner4, subdivisions_x, subdivisions_y):
    vertices = []
    
    for i in range(subdivisions_y + 1):
        t_y = i / subdivisions_y
        row = []
        for j in range(subdivisions_x + 1):
            t_x = j / subdivisions_x
            # Interpolación bilineal para obtener la posición del vértice
            point = (
                (1 - t_x) * (1 - t_y) * np.array(corner1) +
                t_x * (1 - t_y) * np.array(corner2) +
                t_x * t_y * np.array(corner3) +
                (1 - t_x) * t_y * np.array(corner4)
            )
            row.append(point)
        vertices.extend(row)

    normal = np.cross(
        np.array(corner2) - np.array(corner1),
        np.array(corner4) - np.array(corner1)
    )
    normal = normal / np.linalg.norm(normal)  # Normalizar
    normals = np.tile(normal, (len(vertices), 1))

    # Generar los índices de los triángulos
    indices = []
    for i in range(subdivisions_y):
        for j in range(subdivisions_x):
            # Índices de los 4 vértices del rectángulo actual
            top_left = i * (subdivisions_x + 1) + j
            top_right = top_left + 1
            bottom_left = top_left + (subdivisions_x + 1)
            bottom_right = bottom_left + 1

            # Triángulos del rectángulo actual
            indices.append([top_left, bottom_left, top_right])  # Primer triángulo
            indices.append([top_right, bottom_left, bottom_right])  # Segundo triángulo
    
    vertices = np.array(vertices, dtype=np.float32).flatten()
    normals = np.array(vertices, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)

    return vertices, normals, indices

def generate_persona(radio=0.45, nstack=10, nsectors=10,position=[0.0,0.0,0.0],scale=1.0):
    vertices = [
        [0.5,0.4,0.5],
        [0.5,0.4,-0.5],
        [-0.5,0.4,-0.5],
        [-0.5,0.4,0.5],
        [0.8,-2.0,0.5],
        [0.8,-2.0,-0.5],
        [-0.8,-2.0,-0.5],
        [-0.8,-2.0,0.5]
    ]

    for p in range(len(vertices)):
        vertices[p][0] = vertices[p][0]*0.5
        vertices[p][1] = vertices[p][1]*0.5
        vertices[p][2] = vertices[p][2]*0.5

    indices = [
        0,1,2,2,3,0,
        4,5,6,6,7,4,
        0,1,5,5,4,0,
        3,2,6,6,7,3,
        0,3,7,7,4,0,
        1,2,6,6,5,1
    ]

    dfi = np.pi / nstack
    dteta = 2*np.pi / nsectors
    for i in range(nstack + 1):
        fi = -np.pi / 2 + i * dfi
        temp = radio * np.cos(fi)
        y = radio * np.sin(fi)
        for j in range(nsectors + 1):
            teta = j * dteta
            x = temp * np.sin(teta)
            z = temp * np.cos(teta)
            vertices.append([x,y+0.45,z])

            if i < nstack and j < nsectors:
                first = i * (nsectors + 1) + j
                second = first + nsectors + 1
                indices.append(first)
                indices.append(second)
                indices.append(first + 1)
                indices.append(second)
                indices.append(second + 1)
                indices.append(first + 1)

    vertices = transformacion(vertices,position,scale)

    vertices = np.array(vertices, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)
    normales = np.array(vertices)
    
    return vertices,normales, indices

def generate_short_supere(a, position, scale,rot):
    vertices =[]
    indices = []
    u = np.linspace(0,2*np.pi,7, endpoint=False) 
    v = np.linspace (0,0.4,40) 
    for j in range (len(v)):
        for  i in range (len(u)):
            r = curva_pot(v[j], a)
            x = r * np.cos(u[i])
            z = r * np.sin(u[i])
            y = v[j] 

            vertices.append([x,-y,z])

    numu =len (u)
    for j in range (len(v)-1):
        for i in range (numu):
            indices.append(j*numu+i)
            indices.append ((j+1)*numu +i)
        indices.append(j * numu)
        indices.append((j+1)*numu)

    vertices = aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertices,scale)),position)

    vertices = np.array(vertices, dtype = np.float32)
    indices =np.array(indices, dtype=np.uint32)

    return vertices,vertices,indices

#MAIN

def main():
    global global_animate_time
    global global_animate_sombr1illa
    global global_animate_cubo
    global global_animate_man
    global global_animate_players
    global playersPos

    ancho,alto = 800,600
    
    if not glfw.init():
        return
    
    ventana = glfw.create_window(ancho,alto,"Proyecto Final - Corte 3", None, None)

    if not ventana:
        glfw.terminate()
        raise Exception("Ventana Fail")
    
    glfw.make_context_current(ventana)
    glEnable(GL_DEPTH_TEST)

    glfw.set_key_callback(ventana, teclado)

    #FIGURAS
    playerReceptor = []

    vertices_Deus, normales_Deus, indices_Deus = generate_cube(scale=0.4)

    vertices_Pelota, normales_Pelota, indices_Pelota = generate_icosaedro(scale=0.4)

    vertices_Monte1, normales_Monte1, indices_Monte1 = generate_piramide(scale=2,position=[1.0,-0.4,-3.0],rot=[0,32,0])
    vertices_Monte2, normales_Monte2, indices_Monte2 = generate_piramide(scale=3,position=[-0.6,0.0,-3.0])
    
    vertices_Playa, normales_Playa, indices_Playa = generate_plano(scale=4.0,position=[1.0,0.0,0.0])
    vertices_Cancha, normales_Cancha, indices_Cancha = generate_plano(scale=1.2,position=[0.0,0.03,0.6])

    vertices_PR1, normales_PR1, indices_PR1 = generate_persona(scale=0.2)
    vertices_PR2, normales_PR2, indices_PR2 = generate_persona(scale=0.2)
    vertices_PR3, normales_PR3, indices_PR3 = generate_persona(scale=0.2)

    vertices_PA1, normales_PA1, indices_PA1 = generate_persona(scale=0.2)
    vertices_PA2, normales_PA2, indices_PA2 = generate_persona(scale=0.2)
    vertices_PA3, normales_PA3, indices_PA3 = generate_persona(scale=0.2)

    vertices_Chamo, normales_Chamo, indices_Chamo = generate_persona(scale=0.2)
    
    vertices_Mar, normales_Mar, indices_Mar = generate_plano(scale=5.3,position=[-2.3,-0.5,-1.5])
    
    vertices_Poste1, normales_Poste1, indices_Poste1 = generate_superE(0.1, np.array([-0.35,0,0.83]), 0.4,[0,0,0])
    vertices_Poste2, normales_Poste2, indices_Poste2 = generate_superE(0.1, np.array([0.38,0,0.1]), 0.4,[0,0,0])
    vertices_malla, normales_malla, indices_malla = generate_rectangular_mesh([-0.5,0.2,1.16],[0.55,0.2,0],[0.55,1,0],[-0.5,1,1.16],10,10)

    vertices_sombr1, normales_sombr1, indices_sombr1 = generate_short_supere(0.1, [1.3, 0.5, -0.5], 1.0,[-40,-30,0])
    vertices_sombr2, normales_sombr2, indices_sombr2 = generate_short_supere(0.9, [1.3, 0.5, -0.5], 1.0,[-40,-30,0])
    vertices_PosteS, normales_PosteS, indices_PosteS = generate_superE(0.1, [0.83,0,-0.2], 0.4,[-40,-30,0])

    try:
        phong_programa = crear_programa_shader(phong_fragmentCode,phong_vertexCode)
        gouraund_programa = crear_programa_shader(gouaud_fragmentCode,gouraud_vertexCode)

        VAO_Deus,VBO_Deus,EBO_Deus = config_buffers(vertices_Deus,normales_Deus,indices_Deus)

        VAO_Pelota,VBO_Pelota,EBO_Pelota = config_buffers(vertices_Pelota,normales_Pelota,indices_Pelota)

        VAO_Monte1,VBO_Monte1,EBO_Monte1 = config_buffers(vertices_Monte1,normales_Monte1,indices_Monte1)
        VAO_Monte2,VBO_Monte2,EBO_Monte2 = config_buffers(vertices_Monte2,normales_Monte2,indices_Monte2)

        VAO_Playa,VBO_Playa,EBO_Playa = config_buffers(vertices_Playa,normales_Playa,indices_Playa)

        VAO_Cancha,VBO_Cancha,EBO_Cancha = config_buffers(vertices_Cancha,normales_Cancha,indices_Cancha)

        VAO_PR1,VBO_PR1,EBO_PR1 = config_buffers(vertices_PR1,normales_PR1,indices_PR1)
        VAO_PR2,VBO_PR2,EBO_PR2 = config_buffers(vertices_PR2,normales_PR2,indices_PR2)
        VAO_PR3,VBO_PR3,EBO_PR3 = config_buffers(vertices_PR3,normales_PR3,indices_PR3)

        VAO_PA1,VBO_PA1,EBO_PA1 = config_buffers(vertices_PA1,normales_PA1,indices_PA1)
        VAO_PA2,VBO_PA2,EBO_PA2 = config_buffers(vertices_PA2,normales_PA2,indices_PA2)
        VAO_PA3,VBO_PA3,EBO_PA3 = config_buffers(vertices_PA3,normales_PA3,indices_PA3)

        VAO_Chamo,VBO_Chamo,EBO_Chamo = config_buffers(vertices_Chamo,normales_Chamo,indices_Chamo)
        
        VAO_Mar,VBO_Mar,EBO_Mar = config_buffers(vertices_Mar,normales_Mar,indices_Mar)

        VAO_sombr1, VBO_sombr1, EBO_sombr1 = config_buffers(vertices_sombr1, normales_sombr1, indices_sombr1)
        VAO_sombr2, VBO_sombr2, EBO_sombr2 = config_buffers(vertices_sombr2, normales_sombr2, indices_sombr2)
        VAO_PosteS,VBO_PosteS,EBO_PosteS = config_buffers(vertices_PosteS,normales_PosteS,indices_PosteS)

        VAO_Poste1,VBO_Poste1,EBO_Poste1 = config_buffers(vertices_Poste1,normales_Poste1,indices_Poste1)
        VAO_Poste2,VBO_Poste2,EBO_Poste2 = config_buffers(vertices_Poste2,normales_Poste2,indices_Poste2)
        VAO_Malla,VBO_Malla,EBO_Malla = config_buffers(vertices_malla,normales_malla,indices_malla)

        cuboShaders = {
            "program":gouraund_programa,
        }
        cubo2Shaders = {
            "program":phong_programa,
        }
        BLANCO_luz = {
            "pos":[-1.0 , 1.0 , 2.6],
            "amb":[1.0 , 1.0 , 1.0],
            "dif":[0.8 , 0.8 , 0.8],
            "spc":[1.0 , 1.0 , 1.0]
        }
        BLANCO_material = {
            "amb":[1.0 , 1.0 , 1.0],
            "dif":[0.5 , 0.3 , 0.2], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":8.0
        }
        NEGRO_luz = {
            "pos":[-0.5 , 1.0 , 0.6],
            "amb":[0.0 , 0.0 , 0.0],
            "dif":[0.8 , 0.8 , 0.8],
            "spc":[1.0 , 1.0 , 1.0]
        }
        NEGRO_material = {
            "amb":[0.0 , 0.0 , 0.0],
            "dif":[0.5 , 0.3 , 0.2], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":8.0
        }
        cuboLuz = {
            "pos":[0.0 , 1.0 , 0.5],
            "amb":[0.5 , 0.1 , 0.2],
            "dif":[0.1 , 0.1 , 0.1],
            "spc":[1.0 , 1.0 , 1.0]
        }
        cuboMaterial = {
            "amb":[0.6 , 0.1 , 0.1],
            "dif":[0.3 , 0.3 , 0.3], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":8.0
        }
        arenaLuz = {
            "pos":[0.0 , 0.1 , 0.0],
            "amb":[0.9 , 0.8 , 0.4],
            "dif":[0.1 , 0.1 , 0.1],
            "spc":[1.0 , 1.0 , 1.0]
        }
        arenaMaterial = {
            "amb":[0.9 , 0.8 , 0.3],
            "dif":[0.5 , 0.3 , 0.2], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":12.0
        }
        marLuz = {
            "pos":[-2.0 , 0.1 , -1.0],
            "amb":[0.3 , 0.2 , 0.9],
            "dif":[0.8 , 0.8 , 0.8],
            "spc":[1.0 , 1.0 , 1.0]
        }
        marMaterial = {
            "amb":[0.2 , 0.2 , 0.8],
            "dif":[0.5 , 0.3 , 0.2], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":30.0
        }
        monte1Luz = {
            "pos":[-1.0 , 1.0 , 2.6],
            "amb":[0.9 , 0.8 , 0.3],
            "dif":[0.8 , 0.8 , 0.8],
            "spc":[1.0 , 1.0 , 1.0]
        }
        monte1Material = {
            "amb":[0.9 , 0.8 , 0.3],
            "dif":[0.5 , 0.3 , 0.2], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":30.0
        }
        monte2Luz = {
            "pos":[-1.0 , 1.0 , 2.6],
            "amb":[0.9 , 0.8 , 0.3],
            "dif":[0.8 , 0.8 , 0.8],
            "spc":[1.0 , 1.0 , 1.0]
        }
        monte2Material = {
            "amb":[0.9 , 0.8 , 0.3],
            "dif":[0.5 , 0.3 , 0.2], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":30.0
        }
        E1Luz = {
            "pos":[0.0 , 0.1 , -0.5],
            "amb":[0.58, 0.07, 0.59],
            "dif":[0.1 , 0.1 , 0.1],
            "spc":[1.0 , 1.0 , 1.0]
        }
        E1Material = {
            "amb":[0.58, 0.07, 0.59],
            "dif":[0.3 , 0.3 , 0.3], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":30.0
        }
        E2Luz = {
            "pos":[-0.2 , 0.1 , 0.2],
            "amb":[0.91, 0.56, 0.24],
            "dif":[0.1 , 0.1 , 0.1],
            "spc":[1.0 , 1.0 , 1.0]
        }
        E2Material = {
            "amb":[0.91, 0.56, 0.24],
            "dif":[0.3 , 0.3 , 0.3], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":30.0
        }
        ChamoLuz = {
            "pos":[0.0 , 0.1 , 0.0],
            "amb":[0.2 , 0.5 , 0.5],
            "dif":[0.1 , 0.1 , 0.1],
            "spc":[1.0 , 1.0 , 1.0]
        }
        ChamoMaterial = {
            "amb":[0.2 , 0.5 , 0.5],
            "dif":[0.3 , 0.3 , 0.3], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":30.0
        }

        numPlayer = 1

        #Objetos del Entorno

        deus = figura(VAO_Deus,len(indices_Deus),cuboShaders,BLANCO_luz,BLANCO_material)

        pelota = figura(VAO_Pelota,len(indices_Pelota),cuboShaders,cuboLuz,cuboMaterial)

        monte1 = figura(VAO_Monte1,len(indices_Monte1),cuboShaders,monte1Luz,monte1Material)
        monte2 = figura(VAO_Monte2,len(indices_Monte2),cuboShaders,monte2Luz,monte2Material)

        playa = figura(VAO_Playa,len(indices_Playa),cubo2Shaders,arenaLuz,arenaMaterial)
        mar = figura(VAO_Mar,len(indices_Mar),cubo2Shaders,marLuz,marMaterial)
        cancha = figura(VAO_Cancha,len(indices_Cancha),cuboShaders,BLANCO_luz,BLANCO_material)
        sombr1 = figura(VAO_sombr1,len(indices_sombr1),cuboShaders,cuboLuz,cuboMaterial)
        sombr2 = figura(VAO_sombr2,len(indices_sombr2),cuboShaders,cuboLuz,cuboMaterial)
        posteS = figura(VAO_PosteS,len(indices_PosteS),cuboShaders,NEGRO_luz,NEGRO_material)

        poste1 = figura(VAO_Poste1,len(indices_Poste1),cuboShaders,NEGRO_luz,NEGRO_material)
        poste2 = figura(VAO_Poste2,len(indices_Poste2),cuboShaders,NEGRO_luz,NEGRO_material)
        malla = figura(VAO_Malla,len(indices_malla),cuboShaders,BLANCO_luz,BLANCO_material)

        pR1 = figura(VAO_PR1,len(indices_PR1),cubo2Shaders,E1Luz,E1Material)
        pR2 = figura(VAO_PR2,len(indices_PR2),cubo2Shaders,E1Luz,E1Material)
        pR3 = figura(VAO_PR3,len(indices_PR3),cuboShaders,E1Luz,E1Material)

        pA1 = figura(VAO_PA1,len(indices_PA1),cuboShaders,E2Luz,E2Material)
        pA2 = figura(VAO_PA2,len(indices_PA2),cuboShaders,E2Luz,E2Material)
        pA3 = figura(VAO_PA3,len(indices_PA3),cuboShaders,E2Luz,E2Material)

        chamo = figura(VAO_Chamo,len(indices_Chamo),cuboShaders,ChamoLuz,ChamoMaterial)

        pelotaPos = playersPos[0]
        playerReceptor = playersPos[numPlayer]
        
        timeLocal_1 = 0.0
        deltaTime_1 = 0.003

        timeLocal_2 = 0.0
        deltaTime_2 = 0.001

        timeLocal_3 = 0.0
        deltaTime_3 = 0.001

        fov = 60
        aspect_ratio = ancho/alto
        cerca = 0.1
        lejos = 100

        ojo = glm.vec3(0,1,2.4)
        centro = glm.vec3(0.0,0.0,0.0)
        arriba = glm.vec3(0.0,1.0,0.0)

        projection = glm.perspective(glm.radians(fov),aspect_ratio,cerca,lejos)
        vista = glm.lookAt(ojo,centro,arriba)
        
        while not glfw.window_should_close(ventana):
            glClearColor(0.7 ,0.8 ,1.0 ,1.0 )
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)               
            
            #Proyecciones y Vistas
            if global_camara_look:
                ojo = glm.vec3(0.0, 0.1, 2.0)
                centro = glm.rotate( glm.vec3(0.8, 0.2, 0.0),glfw.get_time()*4,glm.vec3(0.0,1.0,0.0))
                arriba = glm.vec3(0.0, 1.0, 0.0)

                vista = glm.lookAt(ojo,centro,arriba)
                projection = glm.perspective(glm.radians(fov),aspect_ratio,cerca,lejos)
            if global_camara_ortg:
                ojo = glm.vec3(0,1,2.4)
                centro = glm.vec3(0.0,0.0,0.0)
                arriba = glm.vec3(0.0,1.0,0.0)

                vista = glm.lookAt(ojo,centro,arriba)
                projection = glm.ortho(-0.6, 1.6, -0.5, 1, 0.1, 100)
            if global_camara_proy:
                ojo = glm.vec3(0,1,2.4)
                centro = glm.vec3(0.0,0.0,0.0)
                arriba = glm.vec3(0.0,1.0,0.0)

                vista = glm.lookAt(ojo,centro,arriba)
                projection = glm.perspective(glm.radians(fov),aspect_ratio,cerca,lejos)
                
            #Animacion de la Pelota
            direccion = trayectoria(pelotaPos,playerReceptor,timeLocal_1)
            timeVertical = -(2*timeLocal_1-1)**4+1 

            pelota_t_rotar = glm.rotate(glm.mat4(1.0), glfw.get_time()*3 ,glm.vec3(0.4 , 1.0 , 0.0))
            pelota_t_posV = glm.translate(glm.mat4(1.0),glm.vec3(0.0,timeVertical*1.0,0.0))
            pelota_t_posX = glm.translate(glm.mat4(1.0),convert2vec3(direccion))

            pelota_transform = pelota_t_posX*pelota_t_posV*pelota_t_rotar

            #Traslacion Jugadores
            T_pR1 = glm.translate(glm.mat4(1.0),convert2vec3(playersPos[2]))
            T_pR2 = glm.translate(glm.mat4(1.0),convert2vec3(playersPos[1]))

            T_pA1 = glm.translate(glm.mat4(1.0),convert2vec3(playersPos[0]))
            T_pA2 = glm.translate(glm.mat4(1.0),convert2vec3(playersPos[3]))

            #DEUS EX MACHINE
            T_deus_rotate = glm.rotate(glm.mat4(1.0),glfw.get_time()*80,glm.vec3(0.0,1.0,1.0))
            T_deus_x = glm.translate(glm.mat4(1.0),glm.vec3(-2+(timeLocal_2*4),1,1))
            T_Deus = T_deus_x * T_deus_rotate

            #OE CHAMO VACILA
            T_Chamo_origen = glm.translate(glm.mat4(1.0),glm.vec3(1.0,0.2,1.0))
            T_chamo_rot = glm.rotate(glm.mat4(1.0),glfw.get_time()*timeLocal_3,glm.vec3(0.3,0.5,1.0))
            T_chamo_Y = glm.translate(glm.mat4(1.0),glm.vec3(0.0,timeLocal_3*2,0.0))
            T_Chamo = T_chamo_Y*T_chamo_rot*T_Chamo_origen

            #Dibujado de Figuras
            deus.dibujar(GL_TRIANGLES,projection,vista,transform=T_Deus)
            chamo.dibujar(GL_TRIANGLES,projection,vista,transform=T_Chamo)

            pelota.dibujar(GL_TRIANGLES,projection,vista,transform=pelota_transform)
            
            posteS.dibujar(GL_TRIANGLE_STRIP,projection,vista)

            if global_animate_cubo:
                timeLocal_2 += deltaTime_2
                if timeLocal_2 >= 1.0:
                    global_animate_cubo = False
                    timeLocal_2 = 0.0

            if global_animate_man:
                timeLocal_3 += deltaTime_3
                if timeLocal_3 >= 1.0:
                    timeLocal_3 = 0.0
            else:
                timeLocal_3 = 0.0

            if global_animate_sombr1illa:
                sombr1.dibujar(GL_TRIANGLE_STRIP,projection,vista)
            else:
                sombr2.dibujar(GL_TRIANGLE_STRIP,projection,vista)

            if global_animate_players:
                playersPos = PosicionesA
            else:
                playersPos = PosicionesB
            
            monte1.dibujar(GL_TRIANGLES,projection,vista)
            monte2.dibujar(GL_TRIANGLES,projection,vista)

            playa.dibujar(GL_TRIANGLES,projection,vista)
            mar.dibujar(GL_TRIANGLES,projection,vista)
            cancha.dibujar(GL_LINE_STRIP,projection,vista)
            
            poste1.dibujar(GL_TRIANGLE_STRIP,projection,vista)
            poste2.dibujar(GL_TRIANGLE_STRIP,projection,vista)
            malla.dibujar(GL_LINE_STRIP,projection,vista)

            pR1.dibujar(GL_TRIANGLES,projection,vista,transform=T_pR1)
            pR2.dibujar(GL_TRIANGLES,projection,vista,transform=T_pR2)
            pA1.dibujar(GL_TRIANGLES,projection,vista,transform=T_pA1)
            pA2.dibujar(GL_TRIANGLES,projection,vista,transform=T_pA2)

            if global_animate_time:
                timeLocal_1 += deltaTime_1
                    
            if timeLocal_1 >= 1.0:
                timeLocal_1 = 0.0
                pelotaPos = playerReceptor
                numPlayer = randomNumber(numPlayer,len(playersPos)-1)
                playerReceptor = playersPos[numPlayer]

            glfw.swap_buffers(ventana)
            glfw.poll_events()

    except Exception as e:
        print(f"Error: \n {e}")
        traceback.print_exc()

    finally:
        glDeleteVertexArrays(1,[VAO_Pelota])
        glDeleteVertexArrays(1,[VAO_Monte1])
        glDeleteVertexArrays(1,[VAO_Monte2])
        glDeleteVertexArrays(1,[VAO_Playa])
        glDeleteVertexArrays(1,[VAO_Poste1])
        glDeleteVertexArrays(1,[VAO_Poste2])
        glDeleteVertexArrays(1,[VAO_PosteS])
        glDeleteVertexArrays(1,[VAO_Mar])
        glDeleteVertexArrays(1,[VAO_Cancha])
        glDeleteVertexArrays(1,[VAO_Malla])
        glDeleteVertexArrays(1,[VAO_sombr1])
        glDeleteVertexArrays(1,[VAO_sombr2])
        glDeleteVertexArrays(1,[VAO_PR1])
        glDeleteVertexArrays(1,[VAO_PR2])
        glDeleteVertexArrays(1,[VAO_PR3])
        glDeleteVertexArrays(1,[VAO_PA1])
        glDeleteVertexArrays(1,[VAO_PA2])
        glDeleteVertexArrays(1,[VAO_PA3])
        glDeleteVertexArrays(1,[VAO_Deus])
        glDeleteVertexArrays(1,[VAO_Chamo])

        glDeleteBuffers(2,VBO_Pelota)
        glDeleteBuffers(2,VBO_Monte1)
        glDeleteBuffers(2,VBO_Monte2)
        glDeleteBuffers(2,VBO_Playa)
        glDeleteBuffers(2,VBO_Poste1)
        glDeleteBuffers(2,VBO_Poste2)
        glDeleteBuffers(2,VBO_PosteS)
        glDeleteBuffers(2,VBO_Mar)
        glDeleteBuffers(2,VBO_Cancha)
        glDeleteBuffers(2,VBO_Malla)
        glDeleteBuffers(2,VBO_sombr1)
        glDeleteBuffers(2,VBO_sombr2)
        glDeleteBuffers(2,VBO_PR1)
        glDeleteBuffers(2,VBO_PR2)
        glDeleteBuffers(2,VBO_PR3)
        glDeleteBuffers(2,VBO_PA1)
        glDeleteBuffers(2,VBO_PA2)
        glDeleteBuffers(2,VBO_PA3)
        glDeleteBuffers(2,VBO_Deus)
        glDeleteBuffers(2,VBO_Chamo)

        glDeleteBuffers(1,[EBO_Pelota])
        glDeleteBuffers(1,[EBO_Monte1])
        glDeleteBuffers(1,[EBO_Monte2])
        glDeleteBuffers(1,[EBO_Playa])
        glDeleteBuffers(1,[EBO_Poste1])
        glDeleteBuffers(1,[EBO_Poste2])
        glDeleteBuffers(1,[EBO_PosteS])
        glDeleteBuffers(1,[EBO_Mar])
        glDeleteBuffers(1,[EBO_Cancha])
        glDeleteBuffers(1,[EBO_Malla])
        glDeleteBuffers(1,[EBO_sombr1])
        glDeleteBuffers(1,[EBO_sombr2])
        glDeleteBuffers(1,[EBO_PR1])
        glDeleteBuffers(1,[EBO_PR2])
        glDeleteBuffers(1,[EBO_PR3])
        glDeleteBuffers(1,[EBO_PA1])
        glDeleteBuffers(1,[EBO_PA2])
        glDeleteBuffers(1,[EBO_PA3])
        glDeleteBuffers(1,[EBO_Deus])
        glDeleteBuffers(1,[EBO_Chamo])

        glDeleteProgram(gouraund_programa)
        glDeleteProgram(phong_programa)

    glfw.terminate()

if __name__ == "__main__":
    main()