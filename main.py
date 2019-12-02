import abc
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Iterator, Optional, Iterable, Mapping, Sequence, Set
import numpy as np
import viewer


from chp_sim import ChpSimulator, MeasureResult


@dataclass(unsafe_hash=True)
class Loc:
    x: float
    y: float


@dataclass(unsafe_hash=True)
class SpaceTimeLocation:
    t: float
    x: float
    y: float


@dataclass
class Stabilizer:
    axis: bool
    qubit_locs: List[Loc]
    x: float
    y: float

    def model(self) -> viewer.Model:
        pts = sorted(self.qubit_locs, key=lambda pt: math.atan2(pt.y - self.y, pt.x - self.x))
        color = (0.8, 0.8, 0.8) if self.axis else (0.4, 0.4, 0.4)
        z = -0.1
        if len(pts) == 2:
            cx = (pts[0].x + pts[1].x) / 2
            cy = (pts[0].y + pts[1].y) / 2
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
                    *[viewer.Vertex(pt.x, pt.y, z) for pt in pts],
                    color=color,
                )
            ])


@dataclass
class SurfaceSimulator:
    chp_sim: ChpSimulator
    time: float = 0.0
    loc_qubit_map: Dict[Loc, int] = field(default_factory=dict)
    qubit_loc_map: Dict[int, Loc] = field(default_factory=dict)
    record: Dict[SpaceTimeLocation, MeasureResult] = field(default_factory=dict)

    def qalloc(self, loc: Loc) -> int:
        i = len(self.loc_qubit_map)
        self.loc_qubit_map[loc] = i
        self.qubit_loc_map[len(self.qubit_loc_map)] = loc
        return i

    def hadamard(self, loc: Loc):
        self.chp_sim.hadamard(self.loc_qubit_map[loc])

    def phase(self, loc: Loc):
        self.chp_sim.phase(self.loc_qubit_map[loc])

    def erase(self, loc: Loc):
        if random.random() < 0.5:
            self.z(loc)
        if random.random() < 0.5:
            self.x(loc)

    def erase_all(self, locs: Iterable[Loc]):
        for loc in locs:
            self.erase(loc)

    def x(self, loc: Loc):
        q = self.loc_qubit_map[loc]
        self.chp_sim.hadamard(q)
        self.chp_sim.phase(q)
        self.chp_sim.phase(q)
        self.chp_sim.hadamard(q)

    def z(self, loc: Loc):
        q = self.loc_qubit_map[loc]
        self.chp_sim.phase(q)
        self.chp_sim.phase(q)

    def reset(self, loc: Loc):
        q = self.loc_qubit_map[loc]
        if self.chp_sim.measure(q):
            self.chp_sim.hadamard(q)
            self.chp_sim.phase(q)
            self.chp_sim.phase(q)
            self.chp_sim.hadamard(q)

    def measure(self, loc: Loc, record: bool = False) -> MeasureResult:
        q = self.loc_qubit_map[loc]
        result = self.chp_sim.measure(q)
        if record:
            ev = SpaceTimeLocation(t=self.time, x=loc.x, y=loc.y)
            self.record[ev] = result
        return result

    def cnot(self, control: Loc, target: Loc):
        c = self.loc_qubit_map[control]
        t = self.loc_qubit_map[target]
        self.chp_sim.cnot(c, t)

    def measure_stabilizer(self, stabilizer: Stabilizer) -> MeasureResult:
        if stabilizer.axis:
            for q in stabilizer.qubit_locs:
                self.hadamard(q)
        for q in stabilizer.qubit_locs[1:]:
            self.cnot(q, stabilizer.qubit_locs[0])

        result = self.measure(stabilizer.qubit_locs[0])

        for q in stabilizer.qubit_locs[1:]:
            self.cnot(q, stabilizer.qubit_locs[0])
        if stabilizer.axis:
            for q in stabilizer.qubit_locs:
                self.hadamard(q)

        self.record[SpaceTimeLocation(t=self.time,
                                      x=stabilizer.x,
                                      y=stabilizer.y)] = result
        return result


@dataclass
class LatticeSurgeryPatch:
    code_distance: int
    _loc_stabilizer_map: Optional[Dict[Loc, Stabilizer]] = None

    def new_big_enough_simulator(self) -> 'SurfaceSimulator':
        d = self.code_distance
        result = SurfaceSimulator(
            chp_sim=ChpSimulator(num_qubits=d**2))

        for loc in self.all_qubit_locs():
            result.qalloc(loc)

        return result

    def all_qubit_locs(self) -> Set[Loc]:
        return {loc
                for stabilizer in self.loc_stabilizer_map.values()
                for loc in stabilizer.qubit_locs}

    @property
    def loc_stabilizer_map(self) -> Dict[Loc, Stabilizer]:
        if self._loc_stabilizer_map is None:
            self._loc_stabilizer_map = self._make_patch_stabilizers()
        return self._loc_stabilizer_map

    def qubit_loc_line(self, y: int) -> List[Loc]:
        return [Loc(i, y) for i in range(self.code_distance)]

    def measurement_line_correlation_pieces(self, y: int, t: int) -> Iterator['CorrelationPiece']:
        for loc in self.qubit_loc_line(y):
            yield CorrelationMeasurementBoundary(t=t, qubit_loc=loc)

    def sub_stabilizer_square(self,
                              *,
                              x_min: Optional[int] = None,
                              y_min: Optional[int] = None,
                              x_max: Optional[int] = None,
                              y_max: Optional[int] = None) -> List[Stabilizer]:
        if x_min is None:
            x_min = -1
        if y_min is None:
            y_min = -1
        if x_max is None:
            x_max = self.code_distance + 1
        if y_max is None:
            y_max = self.code_distance + 1
        result = []
        m = self.loc_stabilizer_map
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if (i + j) % 2 == 0:
                    k = Loc(i + 0.5, j + 0.5)
                    if k in m:
                        result.append(m[k])
        return result

    def sub_data_qubit_square(self,
                              *,
                              x_min: Optional[float] = None,
                              x_max: Optional[float] = None,
                              y_min: Optional[float] = None,
                              y_max: Optional[float] = None) -> List[Loc]:
        if x_min is None:
            x_min = -0.5
        if y_min is None:
            y_min = -0.5
        if x_max is None:
            x_max = self.code_distance - 0.5
        if y_max is None:
            y_max = self.code_distance - 0.5
        assert x_min % 1 == x_max % 1 == y_min % 1 == y_max % 1 == 0.5
        result = []
        for x in range(int(x_min + 0.5), int(x_max + 0.5)):
            for y in range(int(y_min + 0.5), int(y_max + 0.5)):
                result.append(Loc(x, y))
        return result

    def measure_patch_stabilizers(self, sim: SurfaceSimulator):
        for stabilizer in self.loc_stabilizer_map.values():
            sim.measure_stabilizer(stabilizer)

    def init_logical_zero(self, sim: SurfaceSimulator):
        for q in self.all_qubit_locs():
            sim.reset(q)

    def measure_patch(self, sim: SurfaceSimulator):
        for q in self.all_qubit_locs():
            sim.measure(q, record=True)

    def _make_patch_stabilizers(self) -> Dict[Loc, Stabilizer]:
        result = {}
        d = self.code_distance
        for i in range(-1, d):
            for j in range(-1, d):
                axis = bool((i + j) % 2)
                if i == -1 or i == d - 1:
                    if axis:
                        continue
                if j == -1 or j == d - 1:
                    if not axis:
                        continue
                qubits = [
                    (i, j),
                    (i, j + 1),
                    (i + 1, j),
                    (i + 1, j + 1)
                ]
                qubit_locs = [Loc(x, y) for (x, y) in qubits if 0 <= x < d and 0 <= y < d]
                s = Stabilizer(axis=axis, qubit_locs=qubit_locs, x=i+0.5, y=j+0.5)
                result[Loc(s.x, s.y)] = s
        return result


def _validate_stabilizers_commute(stabilizers: List[Stabilizer]):
    for s1 in stabilizers:
        for s2 in stabilizers:
            if s1.axis != s2.axis:
                if len(set(s1.qubit_locs) & set(s2.qubit_locs)) % 2 != 0:
                    raise ValueError(f'Anti-commuting stabilizers:\n'
                                     f'{s1}\n'
                                     f'{s2}\n'
                                     f'{s1.qubit_locs}\n'
                                     f'{s2.qubit_locs}')


class CorrelationPiece(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def measured_value(self, sim: SurfaceSimulator) -> bool:
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

    def measured_value(self, sim: SurfaceSimulator) -> bool:
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

    def measured_value(self, sim: SurfaceSimulator) -> bool:
        return sim.record[self.loc()].value

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
    qubit_loc: Loc

    def measured_value(self, sim: SurfaceSimulator) -> bool:
        return bool(sim.record[self.loc()])

    def loc(self):
        return SpaceTimeLocation(t=self.t, x=self.qubit_loc.x, y=self.qubit_loc.y)

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

    def parity(self, sim: SurfaceSimulator) -> bool:
        result = False
        for p in self.pieces:
            if p.measured_value(sim):
                result = not result
        return result

    def model(self) -> viewer.Model:
        model = viewer.Model()
        for piece in self.pieces:
            model += piece.model()
        return model

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


@dataclass
class LatticeSurgerySituation:
    erasures: Dict[float, List[Loc]]
    duration: int
    logical_measurement: CorrelationSurface
    patch: LatticeSurgeryPatch

    def simulate_situation(self) -> bool:
        sim = self.patch.new_big_enough_simulator()

        self.patch.init_logical_zero(sim)
        for _ in range(self.duration):
            sim.erase_all(self.erasures[sim.time])
            self.patch.measure_patch_stabilizers(sim)
            sim.time += 1
        self.patch.measure_patch(sim)
        return self.logical_measurement.parity(sim)

    def show(self):
        model = viewer.Model()

        # Draw patch as checkerboard at the bottom.
        for stabilizer in self.patch.loc_stabilizer_map.values():
            model += stabilizer.model()

        # Draw erasures as red cubes.
        for t, vs in self.erasures.items():
            for pt in vs:
                model.add_cube(center=(pt.x, pt.y, t), diameter=1, color=(1, 0, 0))

        # Draw correlation surface.
        model += self.logical_measurement.model()

        model.transformed(lambda v: viewer.Vertex(v.x, v.z, v.y)).show()


def chain_link_situation(code_distance: int) -> LatticeSurgerySituation:
    patch = LatticeSurgeryPatch(code_distance)

    reach_bar = patch.sub_data_qubit_square(x_min=2.5, x_max=3.5, y_min=1.5)
    v_link = patch.sub_data_qubit_square(x_min=2.5, x_max=3.5, y_min=1.5, y_max=2.5)
    u_bar = [
        *patch.sub_data_qubit_square(x_min=0.5, x_max=1.5, y_max=4.5),
        *patch.sub_data_qubit_square(x_min=0.5, x_max=5.5, y_min=3.5, y_max=4.5),
        *patch.sub_data_qubit_square(x_min=4.5, x_max=5.5, y_max=4.5),
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
        *patch.sub_stabilizer_square(x_min=0, x_max=2, y_max=5),
        *patch.sub_stabilizer_square(x_min=4, x_max=6, y_max=5),
        *patch.sub_stabilizer_square(x_min=2, x_max=4, y_min=3, y_max=5),
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

    surface.pieces.extend(patch.measurement_line_correlation_pieces(t=17, y=0))
    return LatticeSurgerySituation(
        erasures=erasures,
        duration=17,
        logical_measurement=surface,
        patch=patch)


def main():
    situation = chain_link_situation(code_distance=7)
    print(situation.simulate_situation())
    situation.show()
    for _ in range(9):
        print(situation.simulate_situation())


if __name__ == '__main__':
    main()
