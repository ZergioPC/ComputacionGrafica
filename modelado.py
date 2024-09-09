import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

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
            ])
        self.caras =[
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
            ]
        
        self.colores = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
            (1, 0, 1), (0, 1, 1), (0.5, 0.5, 0.5), (0.5, 0, 0),
            (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5),
            (0, 0.5, 0.5), (0.8, 0.8, 0.8), (0.8, 0, 0),
            (0, 0.8, 0), (0, 0, 0.8), (0.8, 0.8, 0), (0.8, 0, 0.8), (0.8, 0.8, 0.8)
            ]

def camara():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60,800/600,0.001,10)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0,0,2,0,0,0,0,1,0)
        
def dibujar(vertices,caras,colores):
    for indice, cara in enumerate(caras):
        glColor3fv(colores[indice % len(colores)])  
        glBegin(GL_TRIANGLES)  
        for vertice in cara:
            glVertex3fv(vertices[vertice])  
        glEnd()

def main():
    
    icosaedro = Icosaedro()
    
    if not glfw.init():
        return None
    
    ventana = glfw.create_window(800,600,"Icosaedro Programable-Pipeline",None,None)

    if not ventana:
        glfw.terminate()
        return None
    
    glfw.make_context_current(ventana)

    camara()
    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(ventana):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glPushMatrix()
        glRotate(glfw.get_time()*50,0,1,0)
        dibujar(icosaedro.vertices, icosaedro.caras, icosaedro.colores)
        glPopMatrix()

        glfw.swap_buffers(ventana)
        glfw.poll_events()
    
    glfw.terminate()

if __name__ == "__main__":
    main()