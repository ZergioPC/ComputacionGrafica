import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import glm

# Vertex and Fragment Shader code
VERTEX_SHADER_CODE = """
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

FRAGMENT_SHADER_CODE = """
#version 330
in vec3 fragColor;
out vec4 color;
void main() {
    color = vec4(fragColor, 1.0);  // Simple color output
}
"""

def generar_vertices(radio=1):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    prop = radio / np.sqrt(t**2 + 1)
    t *= prop

    return np.array([
        [-prop,  t,  0], [ prop,  t,  0], [-prop, -t,  0], [ prop, -t,  0],
        [ 0, -prop,  t], [ 0,  prop,  t], [ 0, -prop, -t], [ 0,  prop, -t],
        [ t,  0, -prop], [ t,  0,  prop], [-t,  0, -prop], [-t,  0,  prop]
    ], dtype=np.float32)

def generar_caras():
    return [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

def setup_shader_program():
    vertex_shader = shaders.compileShader(VERTEX_SHADER_CODE, GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(FRAGMENT_SHADER_CODE, GL_FRAGMENT_SHADER)
    shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
    return shader_program

def setup_buffers(vertices, indices):
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(0)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    return VAO

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Icosaedro", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)

    vertices = generar_vertices()
    faces = generar_caras()
    indices = np.array([index for face in faces for index in face], dtype=np.uint32)

    shader_program = setup_shader_program()
    VAO = setup_buffers(vertices, indices)

    projection = glm.perspective(glm.radians(60.0), 800 / 600, 0.001, 10)
    view = glm.lookAt(glm.vec3(0, 0, 2), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)

        model = glm.rotate(glm.mat4(1.0), glfw.get_time(), glm.vec3(0.0, 1.0, 0.0))

        model_loc = glGetUniformLocation(shader_program, "model")
        view_loc = glGetUniformLocation(shader_program, "view")
        projection_loc = glGetUniformLocation(shader_program, "projection")

        glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()