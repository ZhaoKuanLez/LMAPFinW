import math
import numpy as np
from typing import (
    Tuple,
    List,
    Dict
)

from env.constants import (
    VECTORS,
    FREE
)
from env.utils import plus
_NDArray = np.ndarray


class AStar:
    base_map = None
    optimal_path = None
    optimal_path_map = None

    def __init__(self):
        self._open    = dict()
        self._close   = dict()

        self.start     = None
        self.end       = None

    def set_map(self, map_: _NDArray):
        self.base_map = map_
        self.optimal_path_map = np.zeros_like(self.base_map)

    @staticmethod
    def _h(a: Tuple, b: Tuple, mode='m'):
        """计算H值

        mode:
            'm'-曼哈顿距离（默认），
            'e'-欧式距离。
        """
        if mode == 'm':
            return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))
        elif mode == 'e':
            return int(math.hypot(a[0] - b[0], a[1] - b[1]))
        else:
            raise ValueError(f'Argument `mode` must be one of \'m\'、\'e\'.')

    @staticmethod
    def _h_euclidean(a: Tuple, b: Tuple) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def optimal_path_between(
        self,
        start: Tuple,
        end: Tuple,
        mode: str = 'm'
    ) -> List:
        """计算起点到达终点的最短路径

        Args:
            start: 起点
            end  : 终点
            mode : 'm'曼哈顿距离（默认），'e'欧式距离。
        """
        self.start, self.end = start, end
        self.base_map[tuple(self.start)] = self.base_map[tuple(self.end)] = FREE

        g, h = 0, self._h(self.start, self.end, mode)
        self._open.update({tuple(self.start): {'F': g + h, 'G': g, 'H': h}})

        come_from = {}
        finished = False
        while self._open.__len__() != 0:
            position = self._get_lowest_f_position()
            position_score = self._open[position]

            self._open.pop(position)
            self._close.update({position: position_score})

            neighbours = self.gather_neighbours(position)
            for neighbour in list(neighbours):

                if neighbour[0] == self.end[0] and neighbour[1] == self.end[1]:
                    come_from[neighbour] = position
                    finished = True
                    break

                if neighbour in self._close:
                    continue

                g = position_score['G'] + 1
                h = self._h(position, self.end, mode)
                f = g + h

                if neighbour in self._open:
                    if g < self._open[neighbour]['G']:
                        self._open[neighbour] = {'F': f, 'G': g, 'H': h}
                        come_from.update({neighbour: position})
                else:
                    self._open.update({neighbour: {'F': f, 'G': g, 'H': h}})
                    come_from.update({neighbour: position})
            if finished:
                break
        self.optimal_path = self._get_optimal_path(come_from)
        self.get_optimal_map()
        return self.optimal_path

    def gather_neighbours(self, position: Tuple) -> List:
        """收集当前位置邻居的坐标"""
        neighbours = set()
        for vector in VECTORS:
            x, y = neighbour = plus(position, vector)
            condition_1 = (0 <= x <= (self.base_map.shape[0] - 1)
                           and 0 <= y <= (self.base_map.shape[1] - 1))
            condition_2 = self.base_map[x, y] == FREE if condition_1 else False
            condition_3 = not (x == position[0] and y == position[1])
            if condition_1 and condition_2 and condition_3:
                neighbours.add(neighbour)
        return list(neighbours)

    def get_optimal_map(self):
        """生成最短路径的地图"""
        if self.optimal_path is None:
            raise RuntimeError("The path should be calculated first!")
        for index, position in enumerate(self.optimal_path):
            position = tuple(position)
            self.optimal_path_map[position] = (len(self.optimal_path) - index)
        return self.optimal_path_map

    def _get_lowest_f_position(self):
        fscore = math.inf
        fscore_position = None

        for position, position_info in self._open.items():
            if position_info['F'] < fscore:
                fscore = position_info['F']
                fscore_position = position
        return fscore_position

    def _get_optimal_path(self, come_from: Dict) -> List:
        optimal_path = [self.end]
        point = tuple(self.end)
        while True:
            point = come_from[point]
            optimal_path.append(point)
            if point[0] == self.start[0] and point[1] == self.start[1]:
                break
        optimal_path.reverse()
        return optimal_path
