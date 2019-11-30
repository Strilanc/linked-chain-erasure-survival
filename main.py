import abc
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Iterator, Optional, Iterable, Mapping, Sequence
import numpy as np
import viewer


from chp_sim import ChpSimulator, MeasureResult

code_distance = 7
sim = ChpSimulator(num_qubits=code_distance**2)
loc_qubit_map = {}
qubit_loc_map = {}
all_qubits = list(range(code_distance**2))
for i in range(code_distance):
    for j in range(code_distance):
        loc_qubit_map[(i, j)] = len(loc_qubit_map)
        qubit_loc_map[len(qubit_loc_map)] = (i, j)


@dataclass
class Stabilizer:
    axis: bool
    qubits: List[int]
    x: float
    y: float

    @property
    def qubit_locs(self) -> List[Tuple[int, int]]:
        return [qubit_loc_map[q] for q in self.qubits]

    def model(self) -> viewer.Model:
        pts = sorted(self.qubit_locs, key=lambda pt: math.atan2(pt[1] - self.y, pt[0] - self.x))
        color = (0.8, 0.8, 0.8) if self.axis else (0.4, 0.4, 0.4)
        z = -0.1
        if len(pts) == 2:
            cx = (pts[0][0] + pts[1][0]) / 2
            cy = (pts[0][1] + pts[1][1]) / 2
            dx = self.x - cx
            dy = self.y - cy
            return wedge(
                center=(cx, cy, z),
                u=(dx, dy, 0),
                v=(-dy, dx, 0),
                angle1=-np.pi/2,
                angle2=+np.pi/2,
                color=color)
        else:
            return viewer.Model(quads=[
                viewer.Quad(
                    *[viewer.Vertex(x, y, z) for x, y in pts],
                    color=color,
                )
            ])



@dataclass(unsafe_hash=True)
class SpaceTimeLocation:
    t: float
    x: float
    y: float


def make_patch_stabilizers():
    for i in range(-1, code_distance):
        for j in range(-1, code_distance):
            axis = bool((i + j) % 2)
            if i == -1 or i == code_distance - 1:
                if axis:
                    continue
            if j == -1 or j == code_distance - 1:
                if not axis:
                    continue
            qubits = [
                (i, j),
                (i, j + 1),
                (i + 1, j),
                (i + 1, j + 1)
            ]
            qubits = [loc_qubit_map[(x, y)] for (x, y) in qubits if 0 <= x < code_distance and 0 <= y < code_distance]
            s = Stabilizer(axis=axis, qubits=qubits, x=i+0.5, y=j+0.5)
            loc_stabilizer_map[(s.x, s.y)] = s


loc_stabilizer_map: Dict[Tuple[float, float], Stabilizer] = {}
make_patch_stabilizers()


for s1 in loc_stabilizer_map.values():
    for s2 in loc_stabilizer_map.values():
        if s1.axis != s2.axis:
            if len(set(s1.qubits) & set(s2.qubits)) % 2 != 0:
                raise ValueError(f'Anti-commuting stabilizers:\n{s1}\n{s2}\n{s1.qubit_locs}\n{s2.qubit_locs}')

stab_results: Dict[SpaceTimeLocation, MeasureResult] = {}


def measure_stabilizer(time: float, stabilizer: Stabilizer) -> MeasureResult:
    if stabilizer.axis:
        for q in stabilizer.qubits:
            sim.hadamard(q)
    for q in stabilizer.qubits[1:]:
        sim.cnot(q, stabilizer.qubits[0])

    result = sim.measure(stabilizer.qubits[0])

    for q in stabilizer.qubits[1:]:
        sim.cnot(q, stabilizer.qubits[0])
    if stabilizer.axis:
        for q in stabilizer.qubits:
            sim.hadamard(q)

    stab_results[SpaceTimeLocation(t=time, x=stabilizer.x, y=stabilizer.y)] = result
    return result


def measure_patch_stabilizers(time: float):
    for stabilizer in loc_stabilizer_map.values():
        measure_stabilizer(time, stabilizer)


def reset(q: int):
    if sim.measure(q):
        sim.hadamard(q)
        sim.phase(q)
        sim.phase(q)
        sim.hadamard(q)
    assert sim.measure(q) == MeasureResult(value=False, determined=True)


def init_logical_zero():
    for q in all_qubits:
        reset(q)


def measure_patch(t: float):
    for q in all_qubits:
        x, y = qubit_loc_map[q]
        stab_results[SpaceTimeLocation(t=t, x=x, y=y)] = sim.measure(q)


def erase_qubit(q: int):
    if random.random() < 0.5:
        sim.phase(q)
        sim.phase(q)
    if random.random() < 0.5:
        sim.hadamard(q)
        sim.phase(q)
        sim.phase(q)
        sim.hadamard(q)


def qubit_loc_square(*,
                     x_min: Optional[float] = None,
                     x_max: Optional[float] = None,
                     y_min: Optional[float] = None,
                     y_max: Optional[float] = None):
    if x_min is None:
        x_min = -0.5
    if y_min is None:
        y_min = -0.5
    if x_max is None:
        x_max = code_distance - 0.5
    if y_max is None:
        y_max = code_distance - 0.5
    assert x_min % 1 == x_max % 1 == y_min % 1 == y_max % 1 == 0.5
    result = []
    for x in range(int(x_min + 0.5), int(x_max + 0.5)):
        for y in range(int(y_min + 0.5), int(y_max + 0.5)):
            result.append((x, y))
    return result


def erase_locs(locs: Iterable[Tuple[int, int]]):
    for x, y in locs:
        q = loc_qubit_map[(x, y)]
        erase_qubit(q)


class CorrelationPiece(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def measured_value(self) -> bool:
        pass

    @abc.abstractmethod
    def loc(self) -> SpaceTimeLocation:
        pass

    @abc.abstractmethod
    def model(self) -> viewer.Model:
        pass


@dataclass
class TimelikeCorrelationSquare(CorrelationPiece):
    t: float
    x: float
    y: float

    def measured_value(self) -> bool:
        return False

    def loc(self):
        return SpaceTimeLocation(t=self.t, x=self.x, y=self.y)

    def model(self) -> viewer.Model:
        loc = self.loc()
        if (loc.x + loc.y) % 2 == 1:
            dx = 0.5
            dy = 0.5
        else:
            dx = 0.5
            dy = -0.5
        return viewer.Model(
            quads=[
                viewer.Quad(
                    viewer.Vertex(loc.x + dx, loc.y + dy, loc.t),
                    viewer.Vertex(loc.x + dx, loc.y + dy, loc.t + 1),
                    viewer.Vertex(loc.x - dx, loc.y - dy, loc.t + 1),
                    viewer.Vertex(loc.x - dx, loc.y - dy, loc.t),
                    color=(0, 0, 1),
                    edge_color=(0, 0, 0),
                )
            ]
        )


@dataclass
class SpacelikeCorrelationSquare(CorrelationPiece):
    t: int
    stabilizer: Stabilizer

    def measured_value(self) -> bool:
        return stab_results[self.loc()].value

    def loc(self):
        return SpaceTimeLocation(t=self.t, x=self.stabilizer.x, y=self.stabilizer.y)

    def model(self) -> viewer.Model:
        loc = self.loc()
        return viewer.Model(
            quads=[
                viewer.Quad(
                    viewer.Vertex(loc.x - 1, loc.y, loc.t),
                    viewer.Vertex(loc.x, loc.y - 1, loc.t),
                    viewer.Vertex(loc.x + 1, loc.y, loc.t),
                    viewer.Vertex(loc.x, loc.y + 1, loc.t),
                    color=(0, 0, 1),
                    edge_color=(0, 0, 0),
                )
            ]
        )


def wedge(*, center: Sequence[float], u: Sequence[float], v: Sequence[float], angle1: float, angle2: float, color: Tuple[float, float, float]) -> viewer.Model:
    u = np.array(u)
    v = np.array(v)
    center = np.array(center)
    angle1 %= np.pi * 2
    angle2 %= np.pi * 2
    if angle2 < angle1:
        angle2 += np.pi * 2
    n = int(np.ceil((angle2 - angle1) * 3))
    vertices = [
        viewer.Vertex(*(center + u * np.cos(a) + v * np.sin(a)))
        for a in np.linspace(angle1, angle2, n + 1)
    ]
    center_vertex = viewer.Vertex(*center)
    return viewer.Model(
        triangles=[
            viewer.Triangle(
                center_vertex,
                vertices[i],
                vertices[i + 1],
                color,
            )
            for i in range(n)
        ]
    )


@dataclass
class CorrelationMeasurementBoundary(CorrelationPiece):
    t: float
    qubit: int

    def measured_value(self):
        return stab_results[self.loc()]

    def loc(self):
        x, y = qubit_loc_map[self.qubit]
        return SpaceTimeLocation(t=self.t, x=x, y=y)

    def model(self) -> viewer.Model:
        loc = self.loc()
        if (loc.x + loc.y) % 2 == 1:
            s = +0.5
        else:
            s = -0.5

        return viewer.Model(
            edges=[
                viewer.Edge(
                    viewer.Vertex(loc.x + 0.5, loc.y + s, loc.t),
                    viewer.Vertex(loc.x - 0.5, loc.y - s, loc.t),
                    color=(0, 0, 1),
                )
            ]
        )


def line_patch(x1, y1, x2, y2, w):
    dx = x2 - x1
    dy = y2 - y1
    d = (dx**2 + dy**2)**0.5
    px = -dy / d * w
    py = dx / d * w
    return [
        (x1 + px, y1 + py),
        (x1 - px, y1 - py),
        (x2 - px, y2 - py),
        (x2 + px, y2 + py),
    ]


@dataclass
class CorrelationSurface:
    pieces: List[CorrelationPiece] = field(default_factory=list)

    def parity(self) -> bool:
        result = False
        for p in self.pieces:
            if p.measured_value():
                result = not result
        return result

    def model(self) -> viewer.Model:
        model = viewer.Model()
        for piece in self.pieces:
            model += piece.model()
        return model

    @staticmethod
    def measurement_line(y: int, t: int) -> Iterator[CorrelationPiece]:
        for loc in qubit_loc_line(y):
            yield CorrelationMeasurementBoundary(t=t, qubit=loc_qubit_map[loc])

    @staticmethod
    def stabilizer_patch(time: int,
                         stabilizers: Iterable[Stabilizer]) -> List[SpacelikeCorrelationSquare]:
        return [
            SpacelikeCorrelationSquare(t=time, stabilizer=s)
            for s in stabilizers
        ]

    @staticmethod
    def stabilizer_patch_boundary(*,
                                  time: int,
                                  time2: Optional[int] = None,
                                  initial_boundary: Iterable[Tuple[int, int]] = (),
                                  stabilizers: Iterable[Stabilizer]) -> List[TimelikeCorrelationSquare]:
        qubits = set(initial_boundary)
        for s in stabilizers:
            for q in s.qubit_locs:
                if q in qubits:
                    qubits.remove(q)
                else:
                    qubits.add(q)

        return [
            TimelikeCorrelationSquare(t, x, y)
            for x, y in qubits
            for t in range(time, time + 1 if time2 is None else time2)
        ]


def qubit_loc_line(y: int) -> List[Tuple[int, int]]:
    return [(i, y) for i in range(code_distance)]


def stabilizer_square(x_min: Optional[int] = None,
                      y_min: Optional[int] = None,
                      x_max: Optional[int] = None,
                      y_max: Optional[int] = None) -> List[Stabilizer]:
    if x_min is None:
        x_min = -1
    if y_min is None:
        y_min = -1
    if x_max is None:
        x_max = code_distance + 1
    if y_max is None:
        y_max = code_distance + 1
    result = []
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            if (i + j) % 2 == 0:
                k = (i + 0.5, j + 0.5)
                if k in loc_stabilizer_map:
                    result.append(loc_stabilizer_map[k])
    return result


def chain() -> Tuple[Mapping[int, List[Tuple[int, int]]], int, CorrelationSurface]:
    reach_bar = qubit_loc_square(x_min=2.5, x_max=3.5, y_min=1.5)
    v_link = qubit_loc_square(x_min=2.5, x_max=3.5, y_min=1.5, y_max=2.5)
    u_bar = [
        *qubit_loc_square(x_min=0.5, x_max=1.5, y_max=4.5),
        *qubit_loc_square(x_min=0.5, x_max=5.5, y_min=3.5, y_max=4.5),
        *qubit_loc_square(x_min=4.5, x_max=5.5, y_max=4.5),
    ]
    erasures = defaultdict(list)
    erasures[5] += reach_bar
    erasures[6] += v_link
    erasures[7] += v_link
    erasures[8] += v_link + u_bar
    erasures[9] += v_link
    erasures[10] += v_link
    erasures[11] += v_link
    erasures[12] += reach_bar

    surface = CorrelationSurface()
    loop = [
        *stabilizer_square(x_min=0, x_max=2, y_max=5),
        *stabilizer_square(x_min=4, x_max=6, y_max=5),
        *stabilizer_square(x_min=2, x_max=4, y_min=3, y_max=5),
    ]
    surface.pieces.extend(surface.stabilizer_patch(time=6, stabilizers=loop))
    surface.pieces.extend(surface.stabilizer_patch(time=10, stabilizers=loop))

    # Draw timelike correlation surface boundaries.
    # surface.pieces.extend(surface.stabilizer_patch_boundary(
    #     time=6,
    #     time2=10,
    #     stabilizers=loop,
    #     initial_boundary=qubit_loc_line(0)))
    # surface.pieces.extend(surface.stabilizer_patch_boundary(
    #     time=0,
    #     time2=6,
    #     stabilizers=[],
    #     initial_boundary=qubit_loc_line(0)))
    # surface.pieces.extend(surface.stabilizer_patch_boundary(
    #     time=10,
    #     time2=17,
    #     stabilizers=[],
    #     initial_boundary=qubit_loc_line(0)))

    surface.pieces.extend(surface.measurement_line(t=17, y=0))
    return erasures, 17, surface


def advance():
    tt = 0

    def tick(n):
        nonlocal tt
        for _ in range(n):
            erase_locs(erasures[tt])
            measure_patch_stabilizers(float(tt))
            tt += 1

    for k in range(10):
        tt = 0
        stab_results.clear()
        erasures, duration, surface = chain()

        init_logical_zero()
        tick(duration)
        measure_patch(tt)
        print(surface.parity())

        if k == 0:
            model = viewer.Model()

            for s in loc_stabilizer_map.values():
                model += s.model()

            for t, vs in erasures.items():
                for x, y in vs:
                    model.add_cube(center=(x, y, t), diameter=1, color=(1, 0, 0))

            # for loc in qubit_loc_map.values():
            #     x, y = loc
            #     p = patches.Rectangle(
            #         (y, 0),
            #         width=0.01,
            #         alpha=0.9,
            #         height=tt,
            #         color='black')
            #     ax.add_patch(p)
            #     art3d.pathpatch_2d_to_3d(p, z=x, zdir="x")

            model += surface.model()
            model = model.transformed(lambda v: viewer.Vertex(v.x, v.z, v.y))
            model.show()


advance()
