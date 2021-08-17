import numpy as np
from glumpy import app, gl, glm, gloo, data
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from glumpy.geometry import colorcube
from glumpy.ext import png
from PIL import Image
from PIL import ImageOps

# Source: https://github.com/rougier/python-opengl

# To make UV-textured cubes, substitute line 38 for line 37

vertex = """
uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec3 a_position;      // Vertex position
attribute vec2 a_texcoord;      // Vertex texture coordinates
varying vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)
void main()
{
    // Assign varying variables
    v_texcoord  = a_texcoord;
    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""

fragment = """
uniform sampler2D u_texture;  // Texture 
varying vec2      v_texcoord; // Interpolated fragment texture coordinates (in)
void main()
{
    // Get texture color - swap out below two lines for UV cubes instead of textured cubes
    vec4 t_color = texture2D(u_texture, v_texcoord);
    //vec4 t_color = vec4(v_texcoord, 0.0, 1.0);
    // Final color
    gl_FragColor = t_color;
}
"""

window = app.Window(width=256, height=256,
                    color=(0.00, 0.00, 0.00, 1.00))

framebuffer = np.zeros((256, 256*3), dtype=np.uint8)

@window.event
def on_draw(dt):
    global phi, theta, duration, count

    window.clear()
    
    gl.glDisable(gl.GL_BLEND)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glViewport(0, 0, 256, 256)
    cube.draw(gl.GL_TRIANGLES, indices)

    # Rotate cube
    theta += 0.5 #degrees
    phi += 0.5 #degrees
    count += 1
    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, theta, 0, 0, 1)
    glm.rotate(model, phi, 0, 1, 0)
    cube['u_model'] = model
    data = glReadPixels(0, 0, window.width, window.height, GL_RGBA, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGBA", (window.width, window.height), data)
    image = ImageOps.flip(image)
    image.save('./test_train/%05d.png' % count)


def cube():
    vtype = [('a_position', np.float32, 3),
             ('a_texcoord', np.float32, 2),
             ('a_normal',   np.float32, 3)]
    itype = np.uint32

    # Vertex positions
    p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                  [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=float)
    # Face normals
    n = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0],
                  [-1, 0, 1], [0, -1, 0], [0, 0, -1]])
    # Texture coords
    t = np.array([[ 0.375, 0], [ 0.375, 1], [ 0.625, 0], [ 0.625, 1],
                [ 0.375, 0.75], [ 0.625, 0.75], [ 0.375, 0.5], [ 0.625, 0.5],
                [ 0.375, 0.25], [ 0.625, 0.25],
                [ 0.125, 1], [ 0.125, 0.75],
                [ 0.875, 1], [ 0.875, 0.75]])

    faces_p = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_n = [0, 0, 0, 0,  1, 1, 1, 1,   2, 2, 2, 2,
               3, 3, 3, 3,  4, 4, 4, 4,   5, 5, 5, 5]
    faces_t = [9, 8, 0, 2,  12, 3, 5, 13,   9, 7, 6, 8,
               10, 11, 4, 1,  4, 5, 3, 1,   5, 4, 6, 7]

    vertices = np.zeros(24, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal']   = n[faces_n]
    vertices['a_texcoord'] = t[faces_t]

    filled = np.resize(
       np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
    filled += np.repeat(4 * np.arange(6, dtype=itype), 6)
    vertices = vertices.view(gloo.VertexBuffer)
    filled = filled.view(gloo.IndexBuffer)

    return vertices, filled

@window.event
def on_resize(width, height):
    cube['u_projection'] = glm.perspective(45.0, width / float(height), 2.0, 100.0)

@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)

vertices, indices = cube()
cube = gloo.Program(vertex, fragment)
cube.bind(vertices)
cube['u_texture'] = np.array(Image.open("new_textures/wood1.png")) # Use desired texture image here
cube['u_model'] = np.eye(4, dtype=np.float32)
cube['u_view'] = glm.translation(0, 0, -5)
phi, theta = 40, 30
count = -1

app.run(framecount=1000) #framecount = number of desired images in dataset