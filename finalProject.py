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
    def __init__(self,VAO,vertices,indices,shaders,luz,material,origen):
        #OpenGL
        self.VAO = VAO
        self.vertices = vertices
        self.indices = indices
        self.programa = shaders["program"]
        self.proyeccion = shaders["projection"]
        self.vista = shaders["vista"]
        self.luz = luz
        self.material = material
        #Posicion inicial
        self.startPos = origen[0]
        self.startScale = origen[2]
        self.startRot = origen[1]

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

def dibujarFigura(programa,VAO,indices,projection,transform,vista,luz,difuse,shine):
    glUseProgram(programa)

    glUniform3f(glGetUniformLocation(programa,"luz_position"),luz[0],luz[1],luz[2]) 
    glUniform3f(glGetUniformLocation(programa,"luz_ambient"),0.5 , 0.1 , 0.2)  
    glUniform3f(glGetUniformLocation(programa,"luz_difuse"),0.1 , 0.1 , 0.1)   
    glUniform3f(glGetUniformLocation(programa,"luz_specular"),1.0 , 1.0 , 1.0) 

    glUniform3f(glGetUniformLocation(programa,"mat_ambient"),0.03 , 0.03 , 0.03) 
    glUniform3f(glGetUniformLocation(programa,"mat_difuse"),difuse[0],difuse[1],difuse[2])     
    glUniform3f(glGetUniformLocation(programa,"mat_specular"),1.0 , 1.0 , 1.0)   
    glUniform1f(glGetUniformLocation(programa,"mat_brillo"),shine)                                               

    glUniformMatrix4fv(glGetUniformLocation(programa,"transformacion"),1,GL_FALSE,glm.value_ptr(transform))
    glUniformMatrix4fv(glGetUniformLocation(programa,"proyeccion"),1,GL_FALSE,glm.value_ptr(projection))
    glUniformMatrix4fv(glGetUniformLocation(programa,"vista"),1,GL_FALSE,glm.value_ptr(vista))

    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, indices, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

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

def generate_cube(position,scale):
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

    vertices_1, normales_1, indices_1 = generate_cube([0.0, 0.0, 0.0], 1.0)
    vertices_2, normales_2, indices_2 = generate_cube([-0.4, 0.0, 0.0], 0.6)
    vertices_3, normales_3, indices_3 = generate_cube([0.0, 0.0, 0.5], 0.5)

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

        VAO_1,VBO_1,EBO_1 = config_buffers(vertices_1,normales_1,indices_1)
        VAO_2,VBO_2,EBO_2 = config_buffers(vertices_2,normales_2,indices_2)
        VAO_3,VBO_3,EBO_3 = config_buffers(vertices_3,normales_3,indices_3)

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
        cuboOrigen = [
            [ 0.0 , 0.0 , 0.0],
            [ 0.0 , 0.0 , 0.0],
            1.0
        ]

        numPlayer = 1

        cubo = figura(VAO_1,vertices_1,len(indices_1),cuboShaders,cuboLuz,cuboMaterial,cuboOrigen)

        pelotaPos = playersPos[0]
        playerReceptor = playersPos[numPlayer]
        
        timeLocal = 0.0
        deltaTime = 0.05
        
        while not glfw.window_should_close(ventana):
            glClearColor(0.1 ,0.1 ,0.1 ,1.0 )
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            #time = np.abs(np.sin((glfw.get_time()*6)))                
                        
            #Animacion de la Pelota
            direccion = trayectoria(pelotaPos,playerReceptor,timeLocal)


            #t_Origen = glm.translate(glm.mat4(1.0),-pelotaPos)
            #t_rotar = glm.rotate(glm.mat4(1.0), glfw.get_time()*4 ,glm.vec3(0.0 , 1.0 , 1.0))
            #t_Delta = glm.translate(glm.mat4(1.0),pelotaPos)
            #t_posV = glm.translate(glm.mat4(1.0),glm.vec3(0.0,time*0.0,0.0))

            t_posX = glm.translate(glm.mat4(1.0),convert2vec3(direccion))

            #transform = t_posX*t_posV*t_Delta*t_rotar*t_Origen
            transform = t_posX

            #dibujarFigura(phong_programa,VAO_1,len(indices_1),projection,transform,vista,luz,[0.2,0.5,0.2],30.0)
            #dibujarFigura(phong_programa,VAO_2,len(indices_2),projection,transform,vista,luz,[1.0,1.0,1.0],60.0)
            #dibujarFigura(gouraund_programa,VAO_3,len(indices_3),projection,transform,vista,luz,[0.5,0.2,0.6],20.0)
            cubo.dibujar(transform)

            timeLocal += deltaTime
                    
            if timeLocal >= 1.0:
                timeLocal = 0.0
                pelotaPos = playerReceptor
                numPlayer = randomNumber(numPlayer,len(playersPos)-1)
                playerReceptor = playersPos[numPlayer]

            glfw.swap_buffers(ventana)
            glfw.poll_events()

    except Exception as e:
        print(f"Error: \n {e}")
        traceback.print_exc()

    finally:
        glDeleteVertexArrays(1,[VAO_1])
        glDeleteVertexArrays(1,[VAO_2])
        glDeleteVertexArrays(1,[VAO_3])
        glDeleteBuffers(2,VBO_1)
        glDeleteBuffers(2,VBO_2)
        glDeleteBuffers(2,VBO_3)
        glDeleteBuffers(1,[EBO_1])
        glDeleteBuffers(1,[EBO_2])
        glDeleteBuffers(1,[EBO_3])

        glDeleteProgram(gouraund_programa)
        glDeleteProgram(phong_programa)

    glfw.terminate()

if __name__ == "__main__":
    main()