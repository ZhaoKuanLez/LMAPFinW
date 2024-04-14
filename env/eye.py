"""
Development Logs
    - 230614: Boost maps for robot observation.
"""
import numpy as np
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from env._world import World
from env.constants import FREE
from env.utils import get_distance
_NDArray = np.ndarray


class Eye(object):

    world: Optional[World] = None

    def __init__(self, fov_size: Union[Tuple, int] = (9, 9)):
        if isinstance(fov_size, int):
            self.fov_size = (fov_size, fov_size)
        elif isinstance(fov_size, tuple):
            self.fov_size = fov_size
        else:
            raise TypeError(f"fov_size type must be one of int or tuple.")
        self.shape = tuple((5, *self.fov_size))

    def set_world(self, world: World):
        self.world = world

    def observe(self, robot_id: int) -> Dict[str, Any]:
        fov_rect = self._get_fov_rect(robot_id)

        obs_map = self._sparse_obs(robot_id, fov_rect)
        # goal_map = self._sparse_goal(robot_id, fov_rect)
        astar_map = self._sparse_astar(robot_id, fov_rect)
        robots_map, robots = self._sparse_robots(fov_rect)
        # shelves_map = self._sparse_shelves(fov_rect)
        # 230614: Add robot 3 steps A* prediction maps to FOV observation.
        # pred_maps = self._sparse_pred_maps(list(robots.keys()), fov_rect)
        # maps = [obs_map, goal_map, robots_map, astar_map, shelves_map]
        maps = [obs_map, robots_map, astar_map]
        # maps.extend(pred_maps)
        maps = np.stack(maps)

        distances: Tuple = get_distance(
            self.world.get_pos_of(robot_id), self.world.get_goal_of(robot_id)
        )
        return {
            'center': robot_id,
            'pos': self.world.get_pos_of(robot_id),
            'maps'  : maps,
            'robots': robots,
            'move_status': self.world.robots[robot_id].move_status,
            'work_status': self.world.robots[robot_id].work_status,
            'distances': distances,
        }

    def _get_fov_rect(self, robot_id: int) -> Tuple[int, ...]:
        x, y = self.world.get_pos_of(robot_id)
        h, w = self.fov_size

        u = int(x - (h // 2))
        l = int(y - (w // 2))
        b = int(u + h)
        r = int(l + w)

        return u, b, l, r

    def _sparse_obs(self, robot_id: int, rect: Iterable) -> _NDArray:
        obs_map = np.zeros(shape=self.fov_size).astype(int)
        u, b, l, r = rect

        # 标记障碍
        for x in range(u, b):
            for y in range(l, r):
                if not self._out_of_map((x, y)):
                    if self.world.base_map[x, y] != FREE:
                        obs_map[x - u, y - l] = 1
                    else:
                        obs_map[x - u, y - l] = 0
                else:
                    obs_map[x - u, y - l] = 1

        # 标记机器人目标位置为空闲
        x, y = self.world.get_goal_of(robot_id)
        if (u <= x <= b) and (l <= y <= r):
            obs_map[x - u - 1, y - l - 1] = 0

        return obs_map.astype(int)

    def _out_of_map(self, pos: Tuple) -> bool:
        return (
            pos[0] < 0
            or pos[0] > self.world.mapper.shape[0] - 1
            or pos[1] < 0
            or pos[1] > self.world.mapper.shape[1] - 1
        )

    def _sparse_goal(self, robot_id: int, rect: Iterable) -> _NDArray:
        x, y = pos = self.world.get_pos_of(robot_id)
        u, b, l, r = rect
        goal_map = np.zeros(shape=self.fov_size).astype(int)
        goal = self.world.get_goal_of(robot_id)

        # 目标在FOV之内，直接标记。
        if (u <= goal[0] <= b) and (l <= goal[1] <= r):
            goal_map[goal[0] - u - 1, goal[1] - l - 1] = 1
            return goal_map

        # 目标在FOV之外，投影至边界。
        else:
            delta_x = goal[0] - pos[0]
            delta_y = goal[1] - pos[1]

            # 边界映射点
            ppx = None
            ppy = None

            # 情况1: 12或6点钟方向
            if delta_y == 0:
                ppy = y
                ppx = u if delta_x < 0 else b

            # 情况2: 9或3点钟方向
            elif delta_x == 0:
                ppx = x
                ppy = l if delta_y < 0 else r

            # 情况3: 第一象限
            elif (delta_x < 0) and (delta_y > 0):
                if abs(delta_x) / abs(delta_y) > 1:
                    ppx = u
                    ppy = int(y + round(abs(delta_y * (x - u) / delta_x)))
                    ppy = ppy if ppy <= r else r
                elif abs(delta_x) / abs(delta_y) == 1:
                    ppx = u
                    ppy = r
                elif abs(delta_x) / abs(delta_y) < 1:
                    ppx = int(x - round(abs(delta_x * (r - y) / delta_y)))
                    ppy = r
                    ppx = ppx if ppx >= u else u

            # 情况4: 第二象限
            elif (delta_x < 0) and (delta_y < 0):
                if abs(delta_x) / abs(delta_y) > 1:
                    ppx = u
                    ppy = int(y - round(abs(delta_y * (x - u) / delta_x)))
                    ppy = ppy if ppy >= l else l
                elif abs(delta_x) / abs(delta_y) == 1:
                    ppx = u
                    ppy = l
                elif abs(delta_x) / abs(delta_y) < 1:
                    ppx = int(x - round(abs(delta_x * (y - l) / delta_y)))
                    ppy = l
                    ppx = ppx if ppx >= u else u

            # 情况5: 第三象限
            elif (delta_x > 0) and (delta_y < 0):
                if abs(delta_x) / abs(delta_y) < 1:
                    ppx = int(x + round(abs(delta_x * (y - l) / delta_y)))
                    ppy = l
                    ppx = ppx if ppx <= b else b
                elif abs(delta_x) / abs(delta_y) == 1:
                    ppx = b
                    ppy = l
                elif abs(delta_x) / abs(delta_y) > 1:
                    ppx = b
                    ppy = int(y - round(abs(delta_y * (b - x) / delta_x)))
                    ppy = ppy if ppy >= l else l

            # 情况6: 第四象限
            elif (delta_x > 0) and (delta_y > 0):
                if abs(delta_x) / abs(delta_y) > 1:
                    ppx = b
                    ppy = int(y + round(abs(delta_y * (b - x) / delta_x)))
                    ppy = ppy if ppy <= y else y
                elif abs(delta_x) / abs(delta_y) == 1:
                    ppx = b
                    ppy = r
                elif abs(delta_x) / abs(delta_y) < 1:
                    ppx = int(x + round(abs(delta_x * (r - y) / delta_y)))
                    ppy = r
                    ppx = ppx if ppx <= b else b
            else:
                raise ValueError("未知错误！")

            if ppx is not None and ppy is not None:
                try:
                    goal_map[ppx - u - 1, ppy - l - 1] = 1
                except IndexError:
                    print("Index error occurred during projection.")
                return goal_map.astype(int)

    def _sparse_astar(self, robot_id: int, rect: Iterable) -> _NDArray:
        astar_map = np.zeros(self.fov_size).astype(int)
        u, b, l, r = rect
        opt_map = self.world.robots[robot_id].astar_map

        for x in range(u, b):
            for y in range(l, r):
                if not self._out_of_map((x, y)):
                    astar_map[x - u, y - l] = opt_map[x, y]

        return astar_map.astype(int)

    def _sparse_robots(self, rect: Iterable) -> Tuple[_NDArray, Dict]:
        robots_map: _NDArray = np.zeros(self.fov_size).astype(int)
        robots: Dict[int, tuple] = {}
        u, b, l, r = rect

        for x in range(u, b):
            for y in range(l, r):
                if (not self._out_of_map((x, y))
                        and self.world.robots_map[x, y] != 0):
                    other = self.world.robots_map[x, y]
                    if other > 0:
                        robots_map[x - u, y - l] = 1
                        robots.update({int(other): tuple((x, y))})
        return robots_map.astype(int), robots

    def _sparse_shelves(self, rect: Iterable) -> _NDArray:
        shelves_map: _NDArray = np.zeros(self.fov_size).astype(int)
        u, b, l, r = rect

        for x in range(u, b):
            for y in range(l, r):
                if (not self._out_of_map((x, y))
                        and self.world.shelves_map[x, y] != 0):
                    shelves_map[x - u, y - l] = 1

        return shelves_map.astype(int)

    # 230614: Generate predictive maps
    def _sparse_pred_maps(self, friends_id: List[int],
                          rect: Iterable) -> Tuple[_NDArray, ...]:
        pred_1 = np.zeros(self.fov_size).astype(int)
        pred_2 = np.zeros(self.fov_size).astype(int)
        pred_3 = np.zeros(self.fov_size).astype(int)

        friends_pos:  Dict[int, tuple] = {ID: self.world.get_pos_of(ID)
                                          for ID in friends_id}
        friends_path: Dict[int, List[tuple]] = {
            ID: self.world.robots[ID].astar_path for ID in friends_id
        }

        heads = friends_pos.copy()
        u, b, l, r = rect

        def get_next_pos(robot_id, position) -> Optional[tuple]:
            """Get robot next position, if no-exist, return None"""
            _path = friends_path[robot_id]
            if position not in _path:
                return None
            position_idx = _path.index(position)
            next_pos_idx = position_idx + 1
            if next_pos_idx >= len(_path):
                return None
            return _path[next_pos_idx]

        def out_of_fov(position) -> bool:
            """Check if the position is within FOV"""
            _x, _y = position
            if _x < u or _x >= b or _y < l or _y >= r:
                return True
            return False

        def mark_pos_on_map(absolute_pos, pred_map) -> None:
            _x, _y = absolute_pos
            relative_pos = tuple((_x - u, _y - l))
            pred_map[relative_pos] = 1

        def mark(head, map_) -> Dict[int, tuple]:
            """
            Marked the predicted step pos onto the map and Update head pointers
            """
            _head = {}
            for robot, pos in head.items():
                if pos is None:
                    _head[robot] = None
                    continue
                next_pos = get_next_pos(robot, pos)
                if next_pos is None:
                    _head[robot] = None
                    continue
                if out_of_fov(next_pos):
                    _head[robot] = next_pos
                    continue

                mark_pos_on_map(next_pos, map_)
                _head[robot] = next_pos

            return _head

        # Generating predictive maps
        heads = mark(heads, pred_1)  # step-1
        heads = mark(heads, pred_2)  # step-2
        mark(heads, pred_3)          # step-3

        return pred_1, pred_2, pred_3

    def observe_group(self, group_robot_id: Optional[List] = None) -> Dict:
        if group_robot_id is None:
            group_robot_id = self.world.robots_id

        src_observations: Dict[int, dict] = {
            robot_id: self.observe(robot_id)
            for robot_id in self.world.robots_id
        }

        # Communication(Shared observation with each other.)
        observations_: Dict[int, Any] = {}
        robots_map: Dict[int, _NDArray] = {
            robot_id: src_observations[robot_id]['maps']
            for robot_id in self.world.robots_id
        }

        for robot_id in self.world.robots_id:
            observation: Dict[str, Any] = {}
            src_obs = src_observations[robot_id]
            observation['center'] = src_obs['center']
            observation['pos'] = self.world.get_pos_of(robot_id)
            observation['maps'] = src_obs['maps']
            observation['work_status'] = src_obs['work_status']
            observation['move_status'] = src_obs['move_status']
            observation['distances'] = src_obs['distances']
            observation['period'] = self._get_period_code(robot_id)
            observation['robots'] = {}

            src_leaves: Dict[int, tuple] = src_obs['robots']
            if observation['center'] in src_leaves:
                src_leaves.pop(observation['center'])

            for leaf in src_leaves:
                leaf_pos: Tuple = src_leaves[leaf]
                leaf_map: _NDArray = robots_map[leaf]
                ws = src_observations[leaf]['work_status']
                # ms = src_observations[leaf]['move_status']
                dist = get_distance(leaf_pos, self.world.get_goal_of(leaf))
                observation['robots'].update(
                    {
                        leaf:
                            {
                                'pos': leaf_pos, 'map': leaf_map,
                                'period': self._get_period_code(leaf),
                                'status': [ws], 'distances': list(dist)
                            }
                    }
                )
            observations_[robot_id] = observation

        observations: Dict[int, dict] = {}
        for robot_id in group_robot_id:
            observations.update({robot_id: observations_[robot_id]})

        return observations

    def _get_period_code(self, robot_id: int) -> List:
        if self.world.robots[robot_id].period_is(13):
            return [1, 0, 0, 0]
        elif self.world.robots[robot_id].period_is(22):
            return [1, 1, 0, 0]
        elif self.world.robots[robot_id].period_is(31):
            return [1, 1, 1, 0]
        elif self.world.robots[robot_id].period_is(40):
            return [1, 1, 1, 1]
