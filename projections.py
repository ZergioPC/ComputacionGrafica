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
    gl_Position = proyeccion * vista * transformacion * vec4(position, 1.0);
}
"""

codigo_shader_fragmentos_1 = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(1.0, 0.0, 0.0, 1.0);  // Fragment shader 1
}
"""

codigo_shader_fragmentos_2 = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(0.0, 1.0, 0.0, 1.0);  // Fragment shader 2
}
"""

def compilar_shader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader)
    glShaderSource(shader, codigo)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        error = glGetShaderInfoLog(shader)
        print(f"Error compilando shader: {error}")
        raise RuntimeError(error)
    return shader


def crear_programa_shader_2():
    shader_vertices = compilar_shader(codigo_shader_vertices, GL_VERTEX_SHADER)
    shader_fragmentos = compilar_shader(codigo_shader_fragmentos_2, GL_FRAGMENT_SHADER)

    programa_shader = glCreateProgram()
    glAttachShader(programa_shader, shader_vertices)
    glAttachShader(programa_shader, shader_fragmentos)
    glLinkProgram(programa_shader)
    if glGetProgramiv(programa_shader, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(programa_shader))

    glDeleteShader(shader_vertices)
    glDeleteShader(shader_fragmentos)
    
    return programa_shader 

def crear_programa_shader_1():
    shader_vertices = compilar_shader(codigo_shader_vertices, GL_VERTEX_SHADER)
    shader_fragmentos = compilar_shader(codigo_shader_fragmentos_1, GL_FRAGMENT_SHADER)

    programa_shader = glCreateProgram()
    glAttachShader(programa_shader, shader_vertices)
    glAttachShader(programa_shader, shader_fragmentos)
    glLinkProgram(programa_shader)
    if glGetProgramiv(programa_shader, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(programa_shader))

    glDeleteShader(shader_vertices)
    glDeleteShader(shader_fragmentos)
    
    return programa_shader 

def configurar_vao(vertices, indices):
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

def dibujar_objeto(programa_shader, VAO, transform, indices_count):
    glUseProgram(programa_shader)
    uvista= glGetUniformLocation(programa_shader, "vista")
    uproyeccion= glGetUniformLocation(programa_shader, "proyeccion")
    umodelo= glGetUniformLocation(programa_shader, "transformacion")
    glUniformMatrix4fv(uvista,1,GL_FALSE, transform.flatten())
    glUniformMatrix4fv(uproyeccion,1,GL_FALSE,transform.flatten())
    glUniformMatrix4fv(umodelo, 1, GL_FALSE, transform.flatten())
    glBindVertexArray(VAO)
    glDrawElements(GL_LINES, indices_count, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

def elipsoide(radio,nstack,nsectors):
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
            vertices.append([x])
            vertices.append([y])
            vertices.append([z])
            if i < nstack and j < nsectors:
                first = i * (nsectors + 1) + j
                second = first + nsectors + 1
                indices.append(first)
                indices.append(second)
                indices.append(first + 1)
                indices.append(second)
                indices.append(second + 1)
                indices.append(first + 1)

    vertices = np.array(vertices, dtype=np.float32).flatten()
    indices = np.array(indices, dtype=np.uint32)
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

def closeWindows(w1,w2,w3,w4):
    if glfw.window_should_close(w1) or glfw.window_should_close(w2) or glfw.window_should_close(w3) or glfw.window_should_close(w4):
        return True
    else:
        return False

def main():

    programa_shader1 = crear_programa_shader_1()
    programa_shader2 = crear_programa_shader_2()
    
    if not glfw.init():
        return

    a,b =800,600

    v1 = glfw.create_window(a, b, "No Vista - No Proyeccion", None, None)
    v2 = glfw.create_window(a, b, "Vista UVN - Proyeccion Ortogonal", None, None)
    v3 = glfw.create_window(a, b, "Vista LookAt - Proyeccion ortogonal", None, None)
    v4 = glfw.create_window(a, b, "Vista LookAt - Proyeccion oblicua", None, None)

    if not v1 or not v2 or not v3 or not v4: 
        glfw.terminate()
        return

    e1_vertices, e1_indices = elipsoide(0.25,10,10)
    VAO_e1,VBO_e1,EBO_e1 = configurar_vao(e1_vertices, e1_indices)
    
    glfw.make_context_current(v1)

    while not glfw.window_should_close(v1):
        glClearColor(0.2 ,0.2 ,0.2 ,0.1 )
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        vista=matriz_vista([0.0,0.0,2],[0,0,0],[0,1,0])
        proyeccion= matriz_perspectiva(45, a/b, 1, 5)

        dibujar_objeto(programa_shader1,VAO_e1,np.identity(4),len(e1_indices))

        """
        if usar_proyeccion:
            
            glUniformMatrix4fv(uvista,1,GL_FALSE, vista.flatten())
            glUniformMatrix4fv(uproyeccion,1,GL_FALSE,proyeccion.flatten())
            
        else:
            glUniformMatrix4fv(uvista,1,GL_FALSE,np.identity(4).flatten())
            glUniformMatrix4fv(uproyeccion,1,GL_FALSE, np.identity(4).flatten())
        
        modelo_prisma = np.identity(4)
        modelo_prisma[0,3] =0.5
        glUniformMatrix4fv(umodelo, 1, GL_FALSE, modelo_prisma.flatten())
        """
        
        glfw.swap_buffers(v1)
        glfw.poll_events()

    glDeleteVertexArrays(1, [VAO_e1])
    glDeleteBuffers(1, [VBO_e1])
    glDeleteBuffers(1, [EBO_e1])
    glDeleteProgram(programa_shader1)
    
    
    glfw.terminate()

#usar las teclas para variar los valores
if __name__ == "__main__":
    main()

