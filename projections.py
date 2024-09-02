import glfw  # Importar la biblioteca GLFW para manejo de ventanas y contexto OpenGL
from OpenGL.GL import *  # Importar las funciones de OpenGL
import numpy as np  # Importar numpy para manejo de arrays
import ctypes  # Importar ctypes para conversiones de tipos necesarios por OpenGL

# Código fuente del shader de vértices, en GLSL, que define cómo se transforman las posiciones de los vértices
codigo_shader_vertices = """
#version 330 core
layout(location = 0) in vec3 position; // Atributo de entrada para la posición del vértice
uniform mat4 vista; // Matriz de vista para la transformación de cámara
uniform mat4 proyeccion; // Matriz de proyección para la transformación de perspectiva
uniform mat4 transformacion;
void main()
{
    gl_Position = proyeccion * vista * transformacion * vec4(position, 1.0);
}
"""

# Código fuente del shader de fragmentos, en GLSL, que define el color de los fragmentos (píxeles) en pantalla
codigo_shader_fragmentos = """
#version 330 core
out vec4 color; // Color de salida del fragmento
void main()
{
    color = vec4(1.0, 0.5, 0.5, 0.5); // Asignar color RGBA al fragmento
}
"""

# Función para compilar un shader dado su código fuente y tipo (vértice o fragmento)
def compilar_shader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader)  # Crear un objeto shader vacío
    glShaderSource(shader, codigo)  # Asignar el código fuente al shader
    glCompileShader(shader)  # Compilar el shader
    # Verificar si la compilación fue exitosa, si no, lanzar un error con el log de errores
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader  # Devolver el shader compilado

# Función para crear un programa de shaders, uniendo el shader de vértices y el de fragmentos
def crear_programa_shader():
    # Compilar los shaders de vértices y fragmentos
    shader_vertices = compilar_shader(codigo_shader_vertices, GL_VERTEX_SHADER)
    shader_fragmentos = compilar_shader(codigo_shader_fragmentos, GL_FRAGMENT_SHADER)

    # Crear un programa de shader vacío
    programa_shader = glCreateProgram()
    # Adjuntar los shaders compilados al programa
    glAttachShader(programa_shader, shader_vertices)
    glAttachShader(programa_shader, shader_fragmentos)

    # Linkear (unir) los shaders en el programa
    glLinkProgram(programa_shader)
    # Verificar si el linkeo fue exitoso, si no, lanzar un error con el log de errores
    if glGetProgramiv(programa_shader, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(programa_shader))

    # Eliminar los shaders ya que no son necesarios después de ser linkeados en el programa
    glDeleteShader(shader_vertices)
    glDeleteShader(shader_fragmentos)
    return programa_shader  # Devolver el programa de shader creado

def generar_prisma_hexagonal(altura,radio):
    vertices = []
    posZ = 0.4

    for j in [posZ,-(posZ + 1)]:
        for i in range(6):
            a = 2*np.pi*i/6
            x = radio*np.cos(a)
            y = radio*np.sin(a)
            z = j*altura/2
            vertices.append([x,y,z])
    
    indices =[]

    for i in range(6):
        indices.append([i,(i+1)%6])
    
    for i in range(6):
        indices.append([i+6,(i+1)%6+6])

    for i in range(6):
        indices.append([i,i+6])

    vertices = np.array(vertices,dtype=np.float32)
    indices = np.array(indices,dtype=np.uint32)

    return vertices,indices

# Función para crear una matriz de vista (cámara)
def matriz_vista(ojo, centro, arriba):
    f = np.array(centro) - np.array(ojo)  # Calcular el vector hacia adelante
    f = f / np.linalg.norm(f)  # Normalizar el vector
    u = np.array(arriba)
    u = u / np.linalg.norm(u)  # Normalizar el vector arriba
    s = np.cross(f, u)  # Calcular el vector derecha
    s = s / np.linalg.norm(s)  # Normalizar el vector derecha
    u = np.cross(s, f)  # Recalcular el vector arriba para asegurar ortogonalidad

    M = np.identity(4)  # Crear matriz de vista
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.identity(4)
    T[:3, 3] = -np.array(ojo)  # Matriz de traslación para mover la cámara

    return M @ T  # Devolver la matriz de vista completa

# Función para crear una matriz de proyección perspectiva
def matriz_perspectiva(fov_y, aspecto, cercano, lejano):
    f = 1.0 / np.tan(np.radians(fov_y) / 2)  # Calcular el factor de escala basado en el FOV
    M = np.array([
        [f / aspecto, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (lejano + cercano) / (cercano - lejano), (2 * lejano * cercano) / (cercano - lejano)],
        [0, 0, -1, 0]
    ], dtype=np.float32)  # Crear matriz de proyección perspectiva

    return M  # Devolver la matriz de proyección

# Función de callback para manejar eventos del teclado
def teclado(ventana, tecla, codigo_escaneo, accion, modificadores):
    global usar_proyeccion
    # Verificar si la tecla 'p' fue presionada
    if tecla == glfw.KEY_P and accion == glfw.PRESS:
        # Alternar el estado de 'usar_proyeccion' entre True y False
        usar_proyeccion = not usar_proyeccion

# Función principal del programa
def main():
    global usar_proyeccion
    usar_proyeccion = False  # Inicialmente sin proyección

    # Inicializar GLFW, si falla, salir de la función
    if not glfw.init():
        return

    # Crear una ventana con tamaño y título
    ventana = glfw.create_window(800, 800, "Objeto 3D", None, None)
    if not ventana:  # Si la ventana no se pudo crear, terminar GLFW y salir de la función
        glfw.terminate()
        return

    # Hacer que el contexto OpenGL de la ventana recién creada sea el actual
    glfw.make_context_current(ventana)
    glfw.set_key_callback(ventana, teclado)  # Configurar la función de callback para teclado

    # Definir los vértices y los índices del prisma
    altura = 1
    radio = 0.6
    vertices, indices = generar_prisma_hexagonal(altura, radio)  # Generar vértices e índices del prisma

    # Inicializar las variables VAO, VBO, y EBO fuera del bloque try
    VAO = VBO = EBO = None

    try:
        # Crear y compilar el programa de shaders
        programa_shader = crear_programa_shader()

        # Generar un array de vértices (VAO), un buffer de vértices (VBO) y un buffer de elementos (EBO)
        VAO = glGenVertexArrays(1)  # Crear un VAO
        VBO = glGenBuffers(1)  # Crear un VBO
        EBO = glGenBuffers(1)  # Crear un EBO

        # Vincular (bind) el VAO para configurarlo
        glBindVertexArray(VAO)

        # Vincular el VBO y copiarle los datos de los vértices
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Vincular el EBO y copiarle los datos de los índices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Especificar cómo OpenGL debe interpretar los datos del VBO
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Desvincular el VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        # Desvincular el VAO
        glBindVertexArray(0)

        u_vista = glGetUniformLocation(programa_shader,"vista")
        u_proyeccion = glGetUniformLocation(programa_shader,"proyeccion")
        u_modelo = glGetUniformLocation(programa_shader,"transformacion")

        while not glfw.window_should_close(ventana):
            glClearColor(0.2,0.2,0.2,1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(programa_shader)

            if usar_proyeccion:
                vista = matriz_vista([-0.15,0.2,2],[0,0,0],[0,1,0])
                proyeccion = matriz_perspectiva(45,600/600, 10, 60)
                glUniformMatrix4fv(u_vista,1,GL_FALSE,vista.flatten())
                glUniformMatrix4fv(u_proyeccion,1,GL_FALSE,proyeccion.flatten())
            else:
                glUniformMatrix4fv(u_vista,1,GL_FALSE,np.identity(4).flatten())
                glUniformMatrix4fv(u_proyeccion,1,GL_FALSE,np.identity(4).flatten())

            model_prisma = np.identity(4)
            model_prisma[0,3] = 0.5
            glUniformMatrix4fv(u_modelo,1,GL_FALSE,model_prisma.flatten())

            glBindVertexArray(VAO)
            glDrawElements(GL_LINES,len(indices),GL_UNSIGNED_INT,None)
            glBindVertexArray(0)

            glfw.swap_buffers(ventana)
            glfw.poll_events()
    
    except Exception as e:
        print(f"Error {e}")

    finally:
        if VAO:
            glDeleteVertexArrays(1,[VAO])
        if VBO:
            glDeleteBuffers(1,[VBO])
        if EBO:
            glDeleteBuffers(1,[EBO])
        if programa_shader:
            glDeleteProgram(programa_shader)

        glfw.terminate()

if __name__ == '__main__':
    main()
