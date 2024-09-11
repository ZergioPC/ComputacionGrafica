import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
import numpy as np
import ctypes
import glm
from glm import value_ptr

shaderVertice = """
#version 330
layout(location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 fragColor;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    fragColor = position;  // Pass vertex position to fragment shader
}
"""

shaderFragmento = """
#version 330
in vec3 fragColor;
out vec4 color;
void main() {
    color = vec4(fragColor, 1.0);  // Simple color output
}
"""

class Icosaedro:
    t = (1.0 + np.sqrt(5.0))/2.0

    def __init__(self,radio=1):
        self.radio = radio
        self.prop = radio/np.sqrt(self.t**2 + 1)
        self.t *= self.prop

        self.vertices = np.array([
            [-self.prop, self.t, 0], [self.prop, self.t, 0], [-self.prop, -self.t, 0], [self.prop, -self.t, 0],
            [0, -self.prop, self.t], [0, self.prop, self.t], [0, -self.prop, -self.t], [0, self.prop, -self.t],
            [self.t, 0, -self.prop], [self.t, 0, self.prop], [-self.t, 0, -self.prop], [-self.t, 0, self.prop]
            ],dtype=np.float32)
        self.caras =[
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
            ]
        
        self.colores = np.array([
            (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
            (1, 0, 1), (0, 1, 1), (0.5, 0.5, 0.5), (0.5, 0, 0),
            (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5),
            (0, 0.5, 0.5), (0.8, 0.8, 0.8), (0.8, 0, 0),
            (0, 0.8, 0), (0, 0, 0.8), (0.8, 0.8, 0), (0.8, 0, 0.8), (0.8, 0.8, 0.8)
            ])

def compilarShader(codigo, tipo_shader):
    shader = glCreateShader(tipo_shader)
    glShaderSource(shader, codigo)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def programa():
    vertex_shader = shaders.compileShader(shaderVertice, GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(shaderFragmento, GL_FRAGMENT_SHADER)
    shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
    return shader_program

def createBuffers(vertices,indices):
    VAO = glGenVertexArrays(1)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER,VBO)
    glBufferData(GL_ARRAY_BUFFER,vertices.nbytes,vertices,GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,indices.nbytes,indices,GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return VAO

def main():
    if not glfw.init():
        return None
    
    ventana = glfw.create_window(800,600,"Icosaedro Programable-Pipeline",None,None)

    if not ventana:
        glfw.terminate()
        return None
    
    glfw.make_context_current(ventana)
    glEnable(GL_DEPTH_TEST)
    icosaedro = Icosaedro()
    programaShader = programa()

    projection = glm.perspective(glm.radians(60.0), 800 / 600, 0.001, 10)
    view = glm.lookAt(glm.vec3(0, 0, 2), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    
    indices = np.array([index for face in icosaedro.caras for index in face], dtype=np.uint32)
    VAO = createBuffers(icosaedro.vertices,indices)

    while not glfw.window_should_close(ventana):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(programaShader)

        model = glm.rotate(glm.mat4(1.0), glfw.get_time(), glm.vec3(0.0, 1.0, 0.0))

        model_loc = glGetUniformLocation(programaShader, "model")
        view_loc = glGetUniformLocation(programaShader, "view")
        projection_loc = glGetUniformLocation(programaShader, "projection")

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(ventana)
        glfw.poll_events()
    
    glDeleteVertexArrays(1,[VAO])
    glfw.terminate()

if __name__ == "__main__":
    main()