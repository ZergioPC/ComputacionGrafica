# Online Python - IDE, Editor, Compiler, Interpreter
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
    color = vec4(1.5, 0.1, 0.5, 1.0);
    
}
"""

Codigo_shaderFragmentos2 = """
#version 330 core
out vec4 color;
void main()
{
    color = vec4(0.0, 1.0, 0.0, 1.0);
    
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

def elipsoide(radio,nstack,nsectors,delta):
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
            vertices.append([(x*1.6)+delta])
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

def matriz_uvn(n,v, vrp):
    n=np.array(n)
    v=np.array(v)
    u=np.dot(v,n)
    n=n/np.linalg.norm(n)
    v=v/np.linalg.norm(v)
    u=u/np.linalg.norm(u)
    M=np.identity(4)
    M[0, :3]=u
    M[1, :3]=v
    M[2, :3]=n
    a=np.dot(vrp,u)
    b=np.dot(vrp,v)
    c=np.dot(vrp,n)
    M[0,3], M[1,3], M[2,3]= -a,-b,-c
    return M

def matriz_orto(iac,dl): #IAC: izq, arriba, cerca
    t=np.identity(4)
    i,u,c = iac
    d,a,l = dl
    t[0,3], t[1,3], t[2,3] =-(d+i)/2, -(u+a)/2,(l+c)/2
    s=np.identity(4)
    s[0,0], s[1,1],s[2,2] = 2/(d-i),2/(u,a),-2/(c-l)
    return s@t

def teclado(ventana,tecla, cescan, accion, modifica):
    global vista_1, vista_2, vista_3, vista_4
    
    if tecla == glfw.KEY_2 and accion == glfw.PRESS:
        vista_2 = True
        vista_3 = False
        vista_4 = False
        print(f"vista_2: {vista_2}")
    elif tecla == glfw.KEY_3 and accion == glfw.PRESS:
        vista_3 = True
        vista_2 = False
        vista_4 = False
        print(f"vista_3: {vista_3}")
    elif tecla == glfw.KEY_4 and accion == glfw.PRESS:
        vista_4 = True
        vista_2 = False
        vista_3 = False
        print(f"vista_4: {vista_4}")
    elif tecla == glfw.KEY_1 and accion == glfw.PRESS:
        vista_1 = True
        vista_2 = False
        vista_3 = False
        vista_4 = False
        print(f"vista_1: {vista_1}")

def dibujarFigura(indices,VAO,shader,uvista,modelo_vista,uproyeccion,modelo_proyeccion,umodelo,modelo_figura):
    glUseProgram(shader)
    glUniformMatrix4fv(uvista, 1, GL_FALSE, modelo_vista.flatten())        
    glUniformMatrix4fv(uproyeccion,1,GL_FALSE,modelo_proyeccion.flatten())
    glUniformMatrix4fv(umodelo, 1, GL_FALSE, modelo_figura.flatten())
    glBindVertexArray(VAO)
    glDrawElements(GL_LINES, len(indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

def main():
    global vista_2
    vista_2=False
    
    if not glfw.init():
        return

    a,b =800,600
    ventana = glfw.create_window(a, b, "Proyecciones - Actividad", None, None)
    
    if not ventana: 
        glfw.terminate()
        return

    glfw.make_context_current(ventana)
    glfw.set_key_callback(ventana, teclado)

    #vertices, indices = generar_prismaHex( altura, radio)
    vertices_e1, indices_e1 = elipsoide(0.14,10,10,0.3)
    vertices_e2, indices_e2 = elipsoide(0.2,10,10,-0.3)
    
    try:
        programa_shader_1 = crear_programa_shader(Codigo_shaderFragmentos1)
        programa_shader_2 = crear_programa_shader(Codigo_shaderFragmentos2)

        VAO_e1, VBO_e1, EBO_e1 = config_Buffers(vertices_e1,indices_e1)
        VAO_e2, VBO_e2, EBO_e2 = config_Buffers(vertices_e2,indices_e2)
        
        uvista= glGetUniformLocation(programa_shader_1, "vista")
        uproyeccion= glGetUniformLocation(programa_shader_1, "proyeccion")
        umodelo= glGetUniformLocation(programa_shader_1, "transformacion")

        vista=matriz_vista([-0.15,0.2,2],[0,0,0],[0,1,0])
        proyeccion= matriz_perspectiva(45, a/b, 1, 5)
        
        while not glfw.window_should_close(ventana):
            glClearColor(0.2 ,0.2 ,0.2 ,0.1 )
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            if vista_2:
                dibujarFigura(indices_e1,VAO_e1,programa_shader_1,uvista,vista,uproyeccion,proyeccion,umodelo,np.identity(4))
                dibujarFigura(indices_e2,VAO_e2,programa_shader_2,uvista,vista,uproyeccion,proyeccion,umodelo,np.identity(4))
            else:
                dibujarFigura(indices_e1,VAO_e1,programa_shader_1,uvista,np.identity(4),uproyeccion,np.identity(4),umodelo,np.identity(4))
                dibujarFigura(indices_e2,VAO_e2,programa_shader_2,uvista,np.identity(4),uproyeccion,np.identity(4),umodelo,np.identity(4))
            
            glfw.swap_buffers(ventana)
            glfw.poll_events()

    except Exception as e:
        print(f"Ocurrió un error: {e}")
    
    finally:
    
        glDeleteVertexArrays(1, [VAO_e1])
        glDeleteVertexArrays(1, [VAO_e2])
        glDeleteBuffers(1, [VBO_e1])
        glDeleteBuffers(1, [VBO_e2])
        glDeleteBuffers(1, [EBO_e1])
        glDeleteBuffers(1, [EBO_e2])
        glDeleteProgram(programa_shader_1)
        glDeleteProgram(programa_shader_2)
        
    glfw.terminate()

#usar las teclas para variar los valores
if __name__ == "__main__":
    main()
