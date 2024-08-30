import glfw;
from OpenGL.GL import *;
import glm;
from glm import value_ptr;
import numpy as np;
import ctypes;

shaderVertex = """
#version 330 core
layout(location = 0) in vec3 posicion;
void main(){
    gl_Position = vec4(posicion, 1.0);
}
""";

shaderFragment = """
#version 330 core
out vec4 color;
void main(){
    color = vec4(0.6, 0.1, 1.0, 0.5);
}
"""

def compilarShader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader);
    glShaderSource(shader, codigo);
    glCompileShader(shader);

    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def programa():
    shader_vertices = compilarShader(shaderVertex, GL_VERTEX_SHADER);
    shader_fragmentos = compilarShader(shaderFragment, GL_FRAGMENT_SHADER);

    programa_shader = glCreateProgram();

    glAttachShader(programa_shader, shader_vertices);
    glAttachShader(programa_shader, shader_fragmentos);

    glLinkProgram(programa_shader);

    if glGetProgramiv(programa_shader, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(programa_shader));

    glDeleteShader(shader_vertices);
    glDeleteShader(shader_fragmentos);

    return programa_shader;

# Función para configurar el VAO, VBO y EBO utilizando los datos de vértices e índices proporcionados
def configurar_vao(vertices, indices):
    VAO = glGenVertexArrays(1)  # Generar un VAO (Vertex Array Object)
    VBO = glGenBuffers(1)       # Generar un VBO (Vertex Buffer Object)
    EBO = glGenBuffers(1)       # Generar un EBO (Element Buffer Object)

    glBindVertexArray(VAO)  # Vincular el VAO

    # Vincular y establecer los datos del VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Vincular y establecer los datos del EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Especificar cómo se deben interpretar los datos de los vértices
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)  # Habilitar el atributo de vértices en la posición 0

    glBindBuffer(GL_ARRAY_BUFFER, 0)  # Desvincular el VBO
    glBindVertexArray(0)  # Desvincular el VAO

    return VAO, VBO, EBO  # Devolver los IDs del VAO, VBO y EBO

# Función para dibujar un cubo utilizando un programa de shader y un VAO específicos
def dibujar_cubo(programa_shader, VAO, transformacion):
    glUseProgram(programa_shader)  # Usar el programa de shader proporcionado
    transform_loc = glGetUniformLocation(programa_shader, "transformacion")  # Obtener la ubicación del uniform "transform"
    glUniformMatrix4fv(transform_loc, 1, GL_FALSE, value_ptr(transformacion))  # Establecer la matriz de transformación
    glBindVertexArray(VAO)  # Vincular el VAO que contiene los datos del cubo
    glDrawElements(GL_LINES, 36, GL_UNSIGNED_INT, None)  # Dibujar el cubo usando los índices en el EBO
    glBindVertexArray(0)  # Desvincular el VAO


def prisma():
    vertex = [0.5,0.4,0.5,
              0.5,0.4,-0.5,
              -0.5,0.4,-0.5,
              -0.5,0.4,0.5,
              0.5,-1.0,0.5,
              0.5,-1.0,-0.5,
              -0.5,-1.0,-0.5,
              -0.5,-1.0,0.5, 
    ];

    for p in range(len(vertex)):
        vertex[p] = vertex[p]*0.5
        #vertex[p] = vertex[p]+0.3

    index = [
        0,1,2,2,3,0,
        4,5,6,6,7,4,
        0,1,5,5,4,0,
        3,2,6,6,7,3,
        0,3,7,7,4,0,
        1,2,6,6,5,1
    ]; 

    vertex = np.array(vertex,dtype=np.float32);
    index = np.array(index,dtype=np.uint32).flatten();
    return vertex, index;

def efera(radio, nstack, nsectors):
    vertices = []
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
            vertices.append([x, z, y])
    vertices = np.array(vertices, dtype=np.float32)
    return vertices

def rotArb(punto,eje,angulo):
    normalEje = glm.normalize(eje)
    
    anguloY = glm.atan(normalEje.z,normalEje.x)
    rotY = glm.rotate(glm.mat4(1.0),-anguloY,glm.vec3(0.0,1.0,0.0))
    vectorTransformado = rotY*glm.vec4(normalEje,1.0)

    anguloX = glm.atan(vectorTransformado.y,vectorTransformado.z)
    rotX = glm.rotate(glm.mat4(1.0),-anguloX,glm.vec3(1.0,0.0,0.0))

    traslaOrigen = glm.translate(glm.mat4(1.0),-punto)
    rotZ = glm.rotate(glm.mat4(1.0), angulo, glm.vec3(0.0, 0.0, 1.0))   # Rotación alrededor del eje z (alineado)
    desrotacion_x = glm.rotate(glm.mat4(1.0), anguloX, glm.vec3(1.0, 0.0, 0.0))  # Inversa de la rotación X
    desrotacion_y = glm.rotate(glm.mat4(1.0), anguloY, glm.vec3(0.0, 1.0, 0.0))  # Inversa de la rotación Y
    destraslacion_origen = glm.translate(glm.mat4(1.0), punto)

    # Multiplicar las matrices en el orden correcto para obtener la transformación final
    transformacion_final = destraslacion_origen * desrotacion_y * desrotacion_x * rotZ * rotX * rotY * traslaOrigen

    return transformacion_final

                                                                                     

def main():
    if not glfw.init():
        raise Exception("No GLFW init");

    screen = glfw.create_window(800,600,"Test",None,None);

    if not screen:
        raise Exception("No vista mano");

    glfw.make_context_current(screen);

    vertices,indices = prisma();
    puntoRot = glm.vec3(3.0,-2.0,5.0)
    ejeRot = glm.vec3(1.5,0.8,1.1)
    angulo = glm.radians(50)

    # Crear y compilar el programa de shaders
    programa_shader = programa()


    VAO,VBO,EBO = configurar_vao(vertices,indices)

    glPointSize(10)
    glUseProgram(programa_shader);

    #print(rotArb(puntoRot,ejeRot,angulo))

    while not glfw.window_should_close(screen):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        transform = rotArb(puntoRot,ejeRot,angulo)
        dibujar_cubo(programa_shader,VAO,transform)

        glfw.swap_buffers(screen);
        glfw.poll_events(); 

    glDeleteVertexArrays(1,[VAO]);
    glDeleteBuffers(1,[VBO]);
    glDeleteBuffers(1,[EBO]);
    glDeleteProgram(programa_shader)

    glfw.terminate();

if __name__ == "__main__":
    main();