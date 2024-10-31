import glfw
import glm
import ctypes
import OpenGL.GL as op
import numpy as np

def getShaderCode(path):
    shader = open(path,"r")
    return shader.read

def compilar_shader(codigo, tipo_shader):
    shader = op.glCreateShader(tipo_shader) 
    op.glShaderSource(shader, codigo) 
    op.glCompileShader(shader) 
    if op.glGetShaderiv(shader, op.GL_COMPILE_STATUS) != op.GL_TRUE:
        raise RuntimeError(op.glGetShaderInfoLog(shader))
    return shader 

def crear_programa_shader(frag_Code):
    shader_vertices = compilar_shader(op.Codigo_shaderVertices, op.GL_VERTEX_SHADER)
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

def config_buffers(vertices,indices):
    VAO = op.glGenVertexArrays(1)
    VBO = op.glGenBuffers(1)
    EBO = op.glGenBuffers(1)

    op.glBindVertexArray(VAO)

    op.glBindBuffer(op.GL_ARRAY_BUFFER,VBO)
    op.glBufferData(op.GL_ARRAY_BUFFER,vertices.nbytes,vertices,op.GL_STATIC_DRAW)
    op.glVertexAttribPointer(0,3,op.GL_FLOAT,op.GL_FALSE,3*vertices.itemsize,ctypes.c_void_p(0))

    op.glBindBuffer(op.GL_ELEMENT_ARRAY_BUFFER,EBO)
    op.glBufferData(op.GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,op.GL_STATIC_DRAW)
    op.glVertexAttribPointer(0,3,op.GL_FLOAT,op.GL_FALSE,3*vertices.itemsize,ctypes.c_void_p(0))
    
    op.glEnableVertexAttribArray(0)
    op.glBindBuffer(op.GL_ARRAY_BUFFER,0)
    op.glBindVertexArray(0)

    return VAO,VBO,EBO

def main():
    ancho,alto = 800,600
    
    if not glfw.init():
        return
    
    ventana = glfw.create_window(ancho,alto,"Iluminacion - Sergio Palacios", None, None)

    if not ventana:
        glfw.terminate()
        raise Exception("Ventana Fail")
    
    glfw.make_context_current(ventana)

    vertices,indices = [],[]

    try:
        programa_shader = crear_programa_shader('shader')

        VAO,VBO,EBO = config_buffers(vertices,indices)

        while not glfw.window_should_close(ventana):
            op.glClearColor(0.3 ,0.2 ,0.3 ,1.0 )
            op.glClear(op.GL_COLOR_BUFFER_BIT | op.GL_DEPTH_BUFFER_BIT)
            
            glfw.swap_buffers(ventana)
            glfw.poll_events()

    except Exception as e:
        print(f"Error: \n {e}")

    finally:
        op.glDeleteVertexArrays(1,[VAO])
        op.glDeleteBuffers(1,[VBO])
        op.glDeleteBuffers(1,[EBO])
        op.glDeleteProgram(programa_shader)

    glfw.terminate()

if __name__ == "__main__":
    main()