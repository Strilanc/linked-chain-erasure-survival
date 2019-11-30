import contextlib
from dataclasses import dataclass, field
from typing import Sequence, Tuple, List, Union, Optional, Callable, Iterator

import numpy as np

from OpenGL.GL import shaders
from OpenGL.GL import *
from OpenGL.GLU import *

with contextlib.redirect_stdout(None):
    import pygame
    import pygame.locals


@dataclass
class Vertex:
    x: float
    y: float
    z: float

    def array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def gl_render(self):
        glVertex3f(self.x, self.y, self.z)


@dataclass
class Edge:
    v1: Vertex
    v2: Vertex
    color: Tuple[float, float, float]

    def transformed(self, func: Callable[[Vertex], Vertex]) -> 'Edge':
        return Edge(func(self.v1), func(self.v2), self.color)

    def gl_render(self):
        glColor3fv(self.color)
        self.v1.gl_render()
        self.v2.gl_render()


def cross(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


@dataclass
class Quad:
    v1: Vertex
    v2: Vertex
    v3: Vertex
    v4: Vertex
    color: Tuple[float, float, float]
    edge_color: Optional[Tuple[float, float, float]] = None

    def edges(self) -> List[Edge]:
        return [
            Edge(self.v1, self.v2, self.edge_color),
            Edge(self.v2, self.v3, self.edge_color),
            Edge(self.v3, self.v4, self.edge_color),
            Edge(self.v4, self.v1, self.edge_color),
        ]

    def normal(self) -> np.ndarray:
        a = self.v1.array()
        b = self.v2.array()
        c = self.v3.array()
        u = b - a
        v = c - a
        w = cross(u, v)
        w /= np.sum(w**2)**0.5
        return w

    def gl_render(self):
        glColor3fv(self.color)
        glNormal3fv(self.normal())
        self.v1.gl_render()
        self.v2.gl_render()
        self.v3.gl_render()
        self.v4.gl_render()

    def transformed(self, func: Callable[[Vertex], Vertex]) -> 'Quad':
        return Quad(func(self.v1), func(self.v2), func(self.v3), func(self.v4), self.color, self.edge_color)


@dataclass
class Triangle:
    v1: Vertex
    v2: Vertex
    v3: Vertex
    color: Tuple[float, float, float]
    edge_color: Optional[Tuple[float, float, float]] = None

    def edges(self) -> List[Edge]:
        return [
            Edge(self.v1, self.v2, self.edge_color),
            Edge(self.v2, self.v3, self.edge_color),
            Edge(self.v3, self.v1, self.edge_color),
        ]

    def normal(self) -> np.ndarray:
        a = self.v1.array()
        b = self.v2.array()
        c = self.v3.array()
        u = b - a
        v = c - a
        w = cross(u, v)
        w /= np.sum(w**2)**0.5
        return w

    def gl_render(self):
        glColor3fv(self.color)
        glNormal3fv(self.normal())
        self.v1.gl_render()
        self.v2.gl_render()
        self.v3.gl_render()

    def transformed(self, func: Callable[[Vertex], Vertex]) -> 'Triangle':
        return Triangle(func(self.v1), func(self.v2), func(self.v3), self.color, self.edge_color)


@dataclass
class Camera:
    ratio: float = 800 / 600
    pitch: float = 0
    yaw: float = 0
    xyz: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=np.float32))
    distance: float = 10

    def apply_transform(self):
        glLoadIdentity()
        gluPerspective(45, self.ratio, 0.1, 500.0)
        glTranslatef(0, 0, -self.distance)
        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.yaw, 0, 1, 0)
        glTranslatef(*self.xyz)


@dataclass
class Model:
    edges: List[Edge] = field(default_factory=list)
    quads: List[Quad] = field(default_factory=list)
    triangles: List[Triangle] = field(default_factory=list)

    def iter_all_vertices(self) -> Iterator[Vertex]:
        for e in self.edges:
            yield e.v1
            yield e.v2
        for e in self.triangles:
            yield e.v1
            yield e.v2
            yield e.v3
        for q in self.quads:
            yield q.v1
            yield q.v2
            yield q.v3
            yield q.v4

    def transformed(self, func: Callable[[Vertex], Vertex]) -> 'Model':
        return Model(
            edges=[e.transformed(func) for e in self.edges],
            quads=[e.transformed(func) for e in self.quads],
            triangles=[e.transformed(func) for e in self.triangles],
        )

    def show(self, camera: Optional[Camera] = None):
        display_model(self, camera)

    def __iadd__(self, other):
        if isinstance(other, Model):
            self.edges += other.edges
            self.quads += other.quads
            self.triangles += other.triangles
            return self
        return NotImplemented

    def __add__(self, other):
        result = Model(edges=list(self.edges), quads=list(self.quads), triangles=list(self.triangles))
        return result.__iadd__(other)

    def add_cube(self, *, center: Sequence[float], diameter: Union[float, Sequence[float]], color: Tuple[float, float, float], edge_color: Tuple[float, float, float] = (0, 0, 0)):
        corner_coors = [
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, -1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, -1, 1),
            (-1, 1, 1)
        ]
        edge_indices = [
            (0, 1),
            (0, 3),
            (0, 4),
            (2, 1),
            (2, 3),
            (2, 7),
            (6, 3),
            (6, 4),
            (6, 7),
            (5, 1),
            (5, 4),
            (5, 7)
        ]
        quad_indices = [
            (0, 1, 2, 3),
            (3, 2, 7, 6),
            (6, 7, 5, 4),
            (4, 5, 1, 0),
            (1, 5, 7, 2),
            (4, 0, 3, 6)
        ]

        if isinstance(diameter, (int, float)):
            diameter = [diameter, diameter, diameter]
        vertices = [Vertex(*tuple(base * diam/2 + offset for base, offset, diam in zip(xyz, center, diameter)))
                    for xyz in corner_coors]
        for a, b in edge_indices:
            self.edges.append(Edge(vertices[a], vertices[b], color=edge_color))
        for a, b, c, d in quad_indices:
            self.quads.append(Quad(vertices[a], vertices[b], vertices[c], vertices[d], color=color))

    def gl_render_inner(self, camera: Camera):
        camera.apply_transform()

        glBegin(GL_TRIANGLES)
        for triangle in self.triangles:
            triangle.gl_render()
        glEnd()

        glBegin(GL_QUADS)
        for quad in self.quads:
            quad.gl_render()
        glEnd()

    def gl_render_outer(self, camera: Camera):
        # HACK: Work around z-fighting between lines and quads.
        camera.distance -= 1/32
        camera.apply_transform()
        camera.distance += 1/32

        glBegin(GL_LINES)
        for edge in self.edges:
            edge.gl_render()
        for quad in self.quads:
            if quad.edge_color is not None:
                for edge in quad.edges():
                    edge.gl_render()
        for triangle in self.triangles:
            if triangle.edge_color is not None:
                for edge in triangle.edges():
                    edge.gl_render()
        glEnd()


def display_model(model: Model, camera: Optional[Camera] = None):
    pygame.init()
    display = (800, 600)
    pygame.display.set_caption('3d viewer')
    pygame.display.set_mode(display, pygame.locals.DOUBLEBUF | pygame.locals.OPENGL)
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_DEPTH_TEST)
    glClearColor(1, 1, 1, 1)
    glLineWidth(2)

    vertex = shaders.compileShader("""
                void main() {
                    gl_Position = ftransform();
                    vec3 light = vec3(-0.36, -0.48, -0.64);
                    gl_FrontColor = gl_Color * (dot(gl_Normal, light) + 2) / 3;
                }""", GL_VERTEX_SHADER)
    fragment = shaders.compileShader("""void main() {
                    gl_FragColor = gl_Color;
                }""", GL_FRAGMENT_SHADER)
    lighting_shader = shaders.compileProgram(vertex, fragment)

    vertex = shaders.compileShader("""
                void main() {
                    gl_Position = ftransform();
                    gl_FrontColor = gl_Color;
                }""", GL_VERTEX_SHADER)
    fragment = shaders.compileShader("""void main() {
                    gl_FragColor = gl_Color;
                }""", GL_FRAGMENT_SHADER)
    do_nothing_shader = shaders.compileProgram(vertex, fragment)

    if camera is None:
        min_x = min(v.x for v in model.iter_all_vertices())
        max_x = max(v.x for v in model.iter_all_vertices())
        min_y = min(v.y for v in model.iter_all_vertices())
        max_y = max(v.y for v in model.iter_all_vertices())
        min_z = min(v.z for v in model.iter_all_vertices())
        max_z = max(v.z for v in model.iter_all_vertices())
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        cz = (min_z + max_z) / 2
        dx = max_x - min_x
        dy = max_y - min_y
        dz = max_z - min_z
        d = (dx**2 + dy**2 + dz**2)**0.5 * 2
        cam = Camera(
            xyz = np.array([-cx, -cy, -cz], dtype=np.float32),
            yaw=20,
            pitch=20,
            distance=d)
    else:
        cam = Camera()
    cam.ratio = display[0] / display[1]

    prev_x = 0
    prev_y = 0
    done = False

    while not done:
        events = pygame.event.get()
        if not events:
            pygame.time.wait(10)
            continue

        for event in events:
            if event.type == pygame.QUIT:
                done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

            if event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                dx, dy = x - prev_x, y - prev_y
                prev_x, prev_y = x, y
                if event.buttons[0]:
                    cam.yaw += dx
                    cam.pitch += dy
                    cam.pitch = max(min(90, cam.pitch), -90)
                if event.buttons[2]:
                    mat = glGetDoublev(GL_MODELVIEW_MATRIX)
                    vx = mat[:3, 0]
                    vy = mat[:3, 1]
                    cam.xyz += vx * dx/100 - vy * dy/200

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Wheel away.
                    cam.distance /= 1.1
                    cam.distance = max(cam.distance, 1)

                if event.button == 5:  # Wheel toward.
                    cam.distance *= 1.1

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(lighting_shader)
        model.gl_render_inner(cam)
        glUseProgram(do_nothing_shader)
        model.gl_render_outer(cam)
        pygame.display.flip()

    pygame.quit()


def main():
    model = Model()
    model.add_cube(center=(0, 0, 0), diameter=(2, 1, 3), color=(1, 0, 0))
    model.add_cube(center=(4, 0, 0), diameter=1, color=(0, 0, 1))
    model.show()


if __name__ == '__main__':
    main()
