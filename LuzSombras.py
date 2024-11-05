""" 
Hecho por: Sergio Danilo Palacios
Codigo: 6000806
"""


import glfw
import glm
import OpenGL.GL as op
import numpy as np

#SHADERS y OPENGL CONFIG

def getShaderCode(path):
    shader = open(path,"r")
    return shader.read()

def compilar_shader(codigo, tipo_shader):
    shader = op.glCreateShader(tipo_shader) 
    op.glShaderSource(shader, codigo) 
    op.glCompileShader(shader) 
    if op.glGetShaderiv(shader, op.GL_COMPILE_STATUS) != op.GL_TRUE:
        raise RuntimeError(op.glGetShaderInfoLog(shader))
    return shader 

def crear_programa_shader(frag_Code,vertx_code):
    shader_vertices = compilar_shader(vertx_code, op.GL_VERTEX_SHADER)
    shader_fragmentos = compilar_shader(frag_Code, op.GL_FRAGMENT_SHADER)

    programa_shader = op.glCreateProgram()
    op.glAttachShader(programa_shader, shader_vertices)
    op.glAttachShader(programa_shader, shader_fragmentos)
    op.glLinkProgram(programa_shader)
    if op.glGetProgramiv(programa_shader, op.GL_LINK_STATUS) != op.GL_TRUE:
        raise RuntimeError(op.glGetProgramInfoLog(programa_shader))

    op.glDeleteShader(shader_vertices)
    op.glDeleteShader(shader_fragmentos)
    
    return programa_shader 

def config_buffers(vertices,normales,indices):
    VAO = op.glGenVertexArrays(1)
    VBO = op.glGenBuffers(2)
    EBO = op.glGenBuffers(1)

    op.glBindVertexArray(VAO)

    op.glBindBuffer(op.GL_ARRAY_BUFFER,VBO[0])
    op.glBufferData(op.GL_ARRAY_BUFFER,vertices.nbytes,vertices,op.GL_STATIC_DRAW)
    op.glVertexAttribPointer(0,3,op.GL_FLOAT,op.GL_FALSE,0,None)
    op.glEnableVertexAttribArray(0)

    op.glBindBuffer(op.GL_ARRAY_BUFFER,VBO[1])
    op.glBufferData(op.GL_ARRAY_BUFFER,normales.nbytes,normales,op.GL_STATIC_DRAW)
    op.glVertexAttribPointer(1,3,op.GL_FLOAT,op.GL_FALSE,0,None)
    op.glEnableVertexAttribArray(1)

    op.glBindBuffer(op.GL_ELEMENT_ARRAY_BUFFER,EBO)
    op.glBufferData(op.GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,op.GL_STATIC_DRAW)  
    op.glBindBuffer(op.GL_ARRAY_BUFFER,0)

    op.glBindVertexArray(0)

    return VAO,VBO,EBO

def dibujarFigura(programa,VAO,indices,projection,rotacion,vista,luz,material,vel):
    op.glUseProgram(programa)
    
    transform = glm.rotate(glm.mat4(1.0),glm.radians(glfw.get_time()*vel),rotacion)

    op.glUniform3f(op.glGetUniformLocation(programa,"luz_position"),luz["pos"][0],luz["pos"][1],luz["pos"][2])   #Luz Posicion
    op.glUniform3f(op.glGetUniformLocation(programa,"luz_ambient"),luz["amb"][0],luz["amb"][1],luz["amb"][2])    #Luz Ambiente
    op.glUniform3f(op.glGetUniformLocation(programa,"luz_difuse"),luz["dif"][0],luz["dif"][1],luz["dif"][2])     #Luz Difusion
    op.glUniform3f(op.glGetUniformLocation(programa,"luz_specular"),luz["spc"][0],luz["spc"][1],luz["spc"][2])   #Luz Specular

    op.glUniform3f(op.glGetUniformLocation(programa,"mat_ambient"),material["ambnt"][0],material["ambnt"][1],material["ambnt"][2])   #Material Ambiente
    op.glUniform3f(op.glGetUniformLocation(programa,"mat_difuse"),material["diff"][0],material["diff"][1],material["diff"][2])       #Material Difusion
    op.glUniform3f(op.glGetUniformLocation(programa,"mat_specular"),material["spec"][0],material["spec"][1],material["spec"][2])     #Material Specular
    op.glUniform1f(op.glGetUniformLocation(programa,"mat_brillo"),material["shine"])                                                 #Material Brillo

    op.glUniformMatrix4fv(op.glGetUniformLocation(programa,"transformacion"),1,op.GL_FALSE,glm.value_ptr(transform))
    op.glUniformMatrix4fv(op.glGetUniformLocation(programa,"proyeccion"),1,op.GL_FALSE,glm.value_ptr(projection))
    op.glUniformMatrix4fv(op.glGetUniformLocation(programa,"vista"),1,op.GL_FALSE,glm.value_ptr(vista))

    op.glBindVertexArray(VAO)
    op.glDrawElements(op.GL_TRIANGLES, indices, op.GL_UNSIGNED_INT, None)
    op.glBindVertexArray(0)

#FIGURAS

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

def esfera(radius,stacks,sectors,pos=[0,0,0],rot=[0,0,0],scale=1.0):
    vertices = []
    normals = []
    indices = []

    for i in range(stacks + 1):
        stack_angle = np.pi / 2 - i * np.pi / stacks  # De +pi/2 a -pi/2
        xy = radius * np.cos(stack_angle)  # Radio en el plano x-y
        z = radius * np.sin(stack_angle)   # Coordenada z

        for j in range(sectors + 1):
            sector_angle = j * 2 * np.pi / sectors  # De 0 a 2pi

            # Coordenadas de los vértices (x, y, z)
            x = xy * np.cos(sector_angle)
            y = xy * np.sin(sector_angle)
            vertices.append([x, y, z])

            # Normales (normalizadas)
            nx = x / radius
            ny = y / radius
            nz = z / radius
            normals.extend([nx, ny, nz])

    # Crear índices de los triángulos
    for i in range(stacks):
        for j in range(sectors):
            first = i * (sectors + 1) + j
            second = first + sectors + 1

            # Triángulo 1
            indices.extend([first, second, first + 1])

            # Triángulo 2
            indices.extend([second, second + 1, first + 1])

    vertices = aux_traslacion(aux_rotacion(np.deg2rad(rot[0]),np.deg2rad(rot[1]),np.deg2rad(rot[2]),aux_escalado(vertices,scale)),pos)

    vertices = np.array(vertices, dtype=np.float32).flatten()
    normales = np.array(vertices, dtype=np.uint32)
    indices = np.array(indices, dtype=np.uint32)
    return vertices, normales, indices

#MAIN

def main():
    vertex_shader_phong = getShaderCode("phongVertex.frag")
    vertex_shader_gouraud = getShaderCode("gouraudVertex.frag")

    fragment_shader_phong = getShaderCode("phongFragment.frag")
    fragment_shader_gouraud = getShaderCode("gouraudFragment.frag")

    ancho,alto = 800,600
    
    if not glfw.init():
        return
    
    ventana = glfw.create_window(ancho,alto,"Iluminacion - Sergio Palacios", None, None)

    if not ventana:
        glfw.terminate()
        raise Exception("Ventana Fail")
    
    glfw.make_context_current(ventana)
    op.glEnable(op.GL_DEPTH_TEST)

    fov = 60
    aspect_ratio = ancho/alto
    cerca = 0.1
    lejos = 100

    ojo = glm.vec3(2,1,2)
    centro = glm.vec3(0.0,0.0,0.0)
    arriba = glm.vec3(0.0,1.0,0.0)

    vertices_1,normales_1,indices_1 = esfera(0.5,10,10,pos=[0.5,0.5,0.5])
    vertices_2,normales_2,indices_2 = esfera(0.5,10,10,pos=[-0.3,-0.5,0.1],scale=0.5)
    vertices_3,normales_3,indices_3 = esfera(0.5,10,10,pos=[-0.5,0.5,0.5])

    luz_config_1 = {
        "pos":[9.0 , 9.0 , 9.0],
        "amb":[0.5 , 0.1 , 0.2],
        "dif":[0.1 , 0.1 , 0.1],
        "spc":[1.0 , 1.0 , 1.0]
    }

    luz_config_2 = {
        "pos":[1.0 , 0.0 , -1.0],
        "amb":[0.5 , 0.1 , 0.2],
        "dif":[0.1 , 0.1 , 0.1],
        "spc":[1.0 , 1.0 , 1.0]
    }

    luz_config_3 = {
        "pos":[9.0 , 9.0 , 9.0],
        "amb":[1.0 , 0.1 , 0.2],
        "dif":[1.0 , 1.0 , 0.1],
        "spc":[1.0 , 1.0 , 1.0]
    }

    mat_config_1 = {
        "shine":12.0,
        "ambnt":[1.0 , 0.3 , 0.2],  #Color
        "diff":[0.5 , 0.3 , 0.2],
        "spec":[1.0 , 1.0 , 1.0]
    }

    mat_config_2 = {
        "shine":8.0,
        "ambnt":[0.1 , 1.0 , 0.1],  #Color
        "diff":[0.5 , 0.3 , 0.2],
        "spec":[1.0 , 1.0 , 1.0]
    }

    mat_config_3 = {
        "shine":82.0,
        "ambnt":[0.1 , 0.3 , 1.0],  #Color
        "diff":[0.5 , 0.3 , 0.2],
        "spec":[1.0 , 1.0 , 1.0]
    }

    try:
        programa_shader_phong = crear_programa_shader(fragment_shader_phong,vertex_shader_phong)
        programa_shader_gouraud = crear_programa_shader(fragment_shader_gouraud,vertex_shader_gouraud)

        VAO_1,VBO_1,EBO_1 = config_buffers(vertices_1,normales_1,indices_1)
        VAO_2,VBO_2,EBO_2 = config_buffers(vertices_2,normales_2,indices_2)
        VAO_3,VBO_3,EBO_3 = config_buffers(vertices_3,normales_3,indices_3)

        projection = glm.perspective(glm.radians(fov),aspect_ratio,cerca,lejos)
        vista = glm.lookAt(ojo,centro,arriba)

        direction_1 = glm.vec3(1.0,1.0,0.0)
        direction_2 = glm.vec3(0.0,1.0,0.4)
        direction_3 = glm.vec3(0.0,-1.0,0.0)

        while not glfw.window_should_close(ventana):
            op.glClearColor(0.8 ,0.9 ,0.7 ,1.0 )
            op.glClear(op.GL_COLOR_BUFFER_BIT | op.GL_DEPTH_BUFFER_BIT)

            dibujarFigura(programa_shader_phong,VAO_1,len(indices_1),projection,direction_1,vista,luz_config_1,mat_config_1,20)
            dibujarFigura(programa_shader_gouraud,VAO_2,len(indices_2),projection,direction_2,vista,luz_config_2,mat_config_2,150)
            dibujarFigura(programa_shader_gouraud,VAO_3,len(indices_3),projection,direction_3,vista,luz_config_3,mat_config_3,100)
            
            glfw.swap_buffers(ventana)
            glfw.poll_events()

    except Exception as e:
        print(f"Error: \n {e}")

    finally:
        op.glDeleteVertexArrays(1,[VAO_1])
        op.glDeleteVertexArrays(1,[VAO_2])
        op.glDeleteVertexArrays(1,[VAO_3])
        op.glDeleteBuffers(2,VBO_1)
        op.glDeleteBuffers(2,VBO_2)
        op.glDeleteBuffers(2,VBO_3)
        op.glDeleteBuffers(1,[EBO_1])
        op.glDeleteBuffers(1,[EBO_2])
        op.glDeleteBuffers(1,[EBO_3])

        op.glDeleteProgram(programa_shader_gouraud)
        op.glDeleteProgram(programa_shader_phong)

    glfw.terminate()

if __name__ == "__main__":
    main()