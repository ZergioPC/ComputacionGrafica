import glfw
import traceback
import glm
from OpenGL.GL import *
import numpy as np
import random as rng

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
        self.proyeccion = shaders["projection"]
        self.vista = shaders["vista"]
        self.luz = luz
        self.material = material

    def dibujar(self,transform):
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
        glUniformMatrix4fv(glGetUniformLocation(self.programa,"proyeccion"),1,GL_FALSE,glm.value_ptr(self.proyeccion))
        glUniformMatrix4fv(glGetUniformLocation(self.programa,"vista"),1,GL_FALSE,glm.value_ptr(self.vista))

        glBindVertexArray(self.VAO)
        glDrawElements(GL_TRIANGLES, self.indices, GL_UNSIGNED_INT, None)
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

""" FIGURAS Y TRANSFORMACIONES """

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

def generate_piramide(position=[0.0,0.0,0.0],scale=1.0):

    vertices = [
        [ 0.5, 0.0, 0.5],
        [ 0.5, 0.0,-0.5],
        [-0.5, 0.0,-0.5],
        [-0.5, 0.0, 0.5],
        [ 0.0, 0.5, 0.0]
    ]

    vertices = transformacion(vertices,position,scale)
    
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
# Auxiliares
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

#MAIN

def main():
    ancho,alto = 800,600
    
    if not glfw.init():
        return
    
    ventana = glfw.create_window(ancho,alto,"Proyecto Final - Corte 3", None, None)

    if not ventana:
        glfw.terminate()
        raise Exception("Ventana Fail")
    
    glfw.make_context_current(ventana)
    glEnable(GL_DEPTH_TEST)

    fov = 60
    aspect_ratio = ancho/alto
    cerca = 0.1
    lejos = 100

    ojo = glm.vec3(0,1,2)
    centro = glm.vec3(0.0,0.0,0.0)
    arriba = glm.vec3(0.0,1.0,0.0)

    vertices_Pelota, normales_Pelota, indices_Pelota = generate_icosaedro(scale=0.4)
    vertices_Monte1, normales_Monte1, indices_Monte1 = generate_piramide(scale=0.4)
    vertices_Playa, normales_Playa, indices_Playa = generate_plano(scale=0.6)

    playersPos = [
        [ 0.3 , 0.0 , 0.3],
        [-0.3 , 0.0 ,-0.3],
        [-0.3 , 0.0 , 0.3],
        [ 0.3 , 0.0 ,-0.3],
    ]

    playerReceptor = []

    try:
        phong_programa = crear_programa_shader(phong_fragmentCode,phong_vertexCode)
        gouraund_programa = crear_programa_shader(gouaud_fragmentCode,gouraud_vertexCode)

        VAO_Pelota,VBO_Pelota,EBO_Pelota = config_buffers(vertices_Pelota,normales_Pelota,indices_Pelota)
        VAO_Monte1,VBO_Monte1,EBO_Monte1 = config_buffers(vertices_Monte1,normales_Monte1,indices_Monte1)
        VAO_Playa,VBO_Playa,EBO_Playa = config_buffers(vertices_Playa,normales_Playa,indices_Playa)

        projection = glm.perspective(glm.radians(fov),aspect_ratio,cerca,lejos)
        vista = glm.lookAt(ojo,centro,arriba)

        cuboShaders = {
            "program":phong_programa,
            "projection":projection,
            "vista":vista
        }
        cuboLuz = {
            "pos":[0.0 , 0.5 , 0.0],
            "amb":[0.5 , 0.1 , 0.2],
            "dif":[0.1 , 0.1 , 0.1],
            "spc":[1.0 , 1.0 , 1.0]
        }
        cuboMaterial = {
            "amb":[0.6 , 0.1 , 0.1],
            "dif":[1.0 , 1.0 , 1.0], 
            "spc":[1.0 , 1.0 , 1.0],
            "brillo":30.0
        }

        numPlayer = 1

        pelota = figura(VAO_Pelota,len(indices_Pelota),cuboShaders,cuboLuz,cuboMaterial)
        monte1 = figura(VAO_Monte1,len(indices_Monte1),cuboShaders,cuboLuz,cuboMaterial)
        playa = figura(VAO_Playa,len(indices_Playa),cuboShaders,cuboLuz,cuboMaterial)

        pelotaPos = playersPos[0]
        playerReceptor = playersPos[numPlayer]
        
        timeLocal_1 = 0.0
        deltaTime_1 = 0.0005
        
        while not glfw.window_should_close(ventana):
            glClearColor(0.7 ,0.7 ,0.7 ,1.0 )
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)               
            
            """ 
            #Animacion de la Pelota
            direccion = trayectoria(pelotaPos,playerReceptor,timeLocal_1)
            timeVertical = -(2*timeLocal_1-1)**4+1 

            pelota_t_Origen = glm.translate(glm.mat4(1.0),-convert2vec3(direccion))
            pelota_t_rotar = glm.rotate(glm.mat4(1.0), glfw.get_time()*3 ,glm.vec3(0.4 , 1.0 , 0.0))
            pelota_t_Delta = glm.translate(glm.mat4(1.0),convert2vec3(direccion))
            pelota_t_posV = glm.translate(glm.mat4(1.0),glm.vec3(0.0,timeVertical*1.0,0.0))
            pelota_t_posX = glm.translate(glm.mat4(1.0),convert2vec3(direccion))

            pelota_transform = pelota_t_posX*pelota_t_posV*pelota_t_Delta*pelota_t_rotar*pelota_t_Origen
            """

            #Dibujado de Figuras
            """ 
            pelota.dibujar(pelota_transform)
            monte1.dibujar(glm.mat4(1.0))
            """
            playa.dibujar(glm.mat4(1.0))


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
        glDeleteVertexArrays(1,[VAO_Playa])
        glDeleteBuffers(2,VBO_Pelota)
        glDeleteBuffers(2,VBO_Monte1)
        glDeleteBuffers(2,VBO_Playa)
        glDeleteBuffers(1,[EBO_Pelota])
        glDeleteBuffers(1,[EBO_Monte1])
        glDeleteBuffers(1,[EBO_Playa])

        glDeleteProgram(gouraund_programa)
        glDeleteProgram(phong_programa)

    glfw.terminate()

if __name__ == "__main__":
    main()