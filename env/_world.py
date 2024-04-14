import random
import numpy as np
from copy import copy, deepcopy
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from env.astar import AStar
from env.mapper import Mapper
from env._picker import Picker
from env._robot import Robot
from env.constants import (
    FREE,
    MOVE,
    OBSTACLE,
    ROBOT_SPOT,
    TEST_MODE,
    VECTORS,
    WORK,
)
from env.utils import (
    get_keys_of,
    plus,
    minus,
    match_direction,
)
_NDArray = np.ndarray


class World(object):
    pickers: Optional[Dict[int, Picker]] = None
    base_map: Optional[_NDArray] = None
    shelves_map: Optional[_NDArray] = None
    robots_map: Optional[_NDArray] = None
    goals_map: Optional[Dict[Tuple, List]] = None

    init_shelves_pos: Optional[Dict[int, Tuple]] = None
    init_robots_pos: Optional[Dict[int, Tuple]] = None
    init_robots_goal: Optional[Dict[int, Tuple]] = None

    unload_cost: int = 1
    random_init: bool = False

    def __init__(self, mapper: Mapper, robots_num: int):
        self.mapper = mapper
        self.robots_num = robots_num
        self.robots_id = [robot_id
                          for robot_id in range(1, self.robots_num + 1)]
        self.robots = None

    def reset(self, **kwargs) -> None:
        # Generate all maps
        if kwargs:
            arguments = ['rows', 'cols', 'length', 'pickers']
            input_args = {}
            for arg in arguments:
                if arg in kwargs:
                    input_args.update({arg: kwargs.get(arg)})
            self.mapper.generate(**input_args)
            if 'unload_cost' in kwargs:
                self.unload_cost = kwargs.get('unload_cost')
            if 'random_init' in kwargs:
                self.random_init = kwargs.get('random_init')
        else:
            self.mapper.generate()

        self.base_map = self.mapper.base_map.copy()
        self.robots: Dict[int, Robot] = {robot_id: deepcopy(Robot())
                                         for robot_id in self.robots_id}
        self.pickers = {
            picker_id: deepcopy(Picker(id_=picker_id, pos=picker_pos))
            for picker_id, picker_pos in self.mapper.pickers_spot_pos.items()
        }

        self.shelves_map = np.zeros_like(self.base_map).astype(int)
        self.robots_map = np.zeros_like(self.base_map).astype(int)
        self.goals_map = {}
        for x in range(self.mapper.shape[0]):
            for y in range(self.mapper.shape[1]):
                self.goals_map.update({(x, y): []})

        # Initialization
        self.init_shelves_pos = {}
        self.init_robots_pos = {}
        self.init_robots_goal = {}

        # Loading shelves, robots, and robots' goal
        self._load_shelves()
        self._load_robots()
        self._load_goals()

    def _load_shelves(self) -> None:
        for shelf_id, shelf_pos in self.mapper.shelves_spot_pos.items():
            self.shelves_map[shelf_pos] = shelf_id
            self.init_shelves_pos.update({shelf_id: shelf_pos})

    def _load_robots(self) -> None:
        # Random initialization
        if self.random_init:
            # 确定边界
            x_up, y_left = 1, 1
            x_down = self.base_map.shape[0] - 2
            y_right = self.base_map.shape[1] - 1
            # 计算所有可选择位置（拣货台、货架、十字路口不允许选择）
            optional_spots = {}
            index = 0
            for x in range(x_up, x_down):
                for y in range(y_left, y_right):
                    pos = (x, y)
                    # Condition1 and 2
                    if (pos in self.mapper.pickers_spot_pos.values()
                            or pos in self.mapper.shelves_spot_pos.values()):
                        continue
                    # Condition3
                    if self.is_at_intersection(pos):
                        continue
                    optional_spots.update({index: (x, y)})
                    index += 1
            # 选择出有效位置
            choice_spots = []
            for _ in range(self.robots_num):
                choice_index = random.sample(
                    list(optional_spots.keys()),
                    1
                )[0]
                choice_spot = optional_spots[choice_index]
                choice_spots.append(choice_spot)
                optional_spots.pop(choice_index)
                # 将周围位置（包括自身位置）从可选择位置中移除
                for neighbor in self.collect_adj_pos(choice_spot):
                    if neighbor not in optional_spots.values():
                        continue
                    pos_index = get_keys_of(neighbor, optional_spots)[0]
                    optional_spots.pop(pos_index)
                    # 如果在通道中，则该通道中所有位置从可选择位置中移除
                    # （一个通道只允许生成一个机器人）
                    if not self.is_inside_corridor(choice_spot):
                        continue
                    corridor_points, _ = self.collect_corridor_pos(choice_spot)
                    for corridor_point in corridor_points:
                        if corridor_point not in optional_spots.values():
                            continue
                        point_id = get_keys_of(
                            corridor_point,
                            optional_spots
                        )[0]
                        optional_spots.pop(point_id)

            # 为每个机器人指定初始位置
            for ID, init_pos in zip(self.robots_id, choice_spots):
                self.robots[ID].id = ID
                self.robots[ID].pos_history = []
                self.robots[ID].move(init_pos, WORK['detach'])
                self.robots_map[init_pos] = ID
                self.init_robots_pos[ID] = init_pos
        # 顺序初始化
        else:
            robots_spot: Dict[int, tuple] = self.mapper.robots_spot_pos
            for ID in list(range(1, self.robots_num + 1)):
                init_pos = robots_spot[ID]
                self.robots[ID].id = ID
                self.robots[ID].pos_history = []
                self.robots[ID].move(init_pos, WORK['detach'])
                self.robots[ID].move(init_pos, WORK['detach'])
                self.robots_map[init_pos] = ID
                self.init_robots_pos.update({ID: init_pos})

    def _load_goals(self) -> None:
        selected_shelves_id = np.random.choice(
            list(self.init_shelves_pos.keys()), self.robots_num,
            False
        )

        for robot_id, shelf_id in enumerate(selected_shelves_id, start=1):
            goal = self.init_shelves_pos[shelf_id]
            self.robots[robot_id].goal = goal
            self.goals_map[goal].append(robot_id)
            self.init_robots_goal.update({robot_id: goal})
            self.astar_plan(robot_id)

    def get_pos_of(self, robot_id: int,
                   others: bool = False) -> Union[tuple, dict]:
        if not others:
            return self.robots[robot_id].pos
        else:
            others = list(copy(self.robots_id))
            others.remove(robot_id)
            return {i: self.get_pos_of(i) for i in others}

    def get_goal_of(self, robot_id: int,
                    others: bool = False) -> Union[tuple, dict]:
        if not others:
            return self.robots[robot_id].goal
        else:
            others = list(copy(self.robots_id))
            others.remove(robot_id)
            return {i: self.get_goal_of(i) for i in others}

    def astar_plan(self, robot_id: int) -> None:
        obs_map: _NDArray = self.get_obs_map(robot_id=robot_id, end2free=True)
        planner: AStar = AStar()
        planner.set_map(obs_map)
        planner.optimal_path_between(self.get_pos_of(robot_id),
                                     self.get_goal_of(robot_id))
        self.robots[robot_id].astar_path = planner.optimal_path
        self.robots[robot_id].astar_map = planner.optimal_path_map

    def get_obs_map(self, robot_id: int = 0, source: bool = False,
                    end2free: bool = False) -> _NDArray:
        """生成障碍地图

        Args:
            robot_id: 机器人ID，如果是0则标记全部机器人位置为0
            source: 是否生成原始障碍地图，不考虑机器人。
            end2free: T将目标位置标记为空闲，否则F为障碍；

        Returns:
            01 形式的障碍地图

        Notes:
            - ODrM*：规划器对地图的要球，规划路径的起点必须为空闲，而终点可以是障碍。
            - ODrM*：其会做出协调行为；
        """
        obs_map = np.zeros(self.mapper.shape).astype(int)
        # 计算障碍
        obs_part_1 = self.base_map > FREE
        obs_part_2 = self.base_map < ROBOT_SPOT
        obstacles = np.argwhere(np.logical_and(obs_part_1, obs_part_2)).tolist()
        # 标记
        for obs_pos in obstacles:
            obs_map[tuple(obs_pos)] = OBSTACLE
        if source:
            return obs_map

        # 标记机器人
        if robot_id == 0:
            robots_pos = [self.get_pos_of(i) for i in self.robots_id]
            robots_goal = [self.get_goal_of(i) for i in self.robots_id]
        else:
            robots_pos = [self.get_pos_of(robot_id)]
            robots_goal = [self.get_goal_of(robot_id)]
        for obs_pos in robots_pos:
            obs_map[obs_pos] = FREE
        if end2free:
            for goal in robots_goal:
                obs_map[goal] = FREE
        return obs_map

    def set_goals(self, robot_group: List[int]) -> None:
        # 参数检查
        if robot_group is None:
            raise ValueError("No robots request new goals.")

        # 找位置
        new_goals = {}
        for robot_id in robot_group:
            new_goal: Optional[tuple] = None
            # Case-1: 刚连接上货架，准备去拣货台 --> 拣货台
            if self.robots[robot_id].period_is(22):
                # 统计每个拣货台被选择的次数，选择次数最小的一个
                pickers_pos: Dict[int, tuple] = self.mapper.pickers_spot_pos
                pickers_followers: Dict[tuple, int] = {
                    pickers_pos[picker_id]: 0
                    for picker_id in pickers_pos.keys()
                }
                others_id: List[int] = list(copy(self.robots_id))
                others_id.remove(robot_id)
                for other_id in others_id:
                    other_goal = self.robots[other_id].goal
                    if other_goal in pickers_pos.values():
                        pickers_followers[other_goal] += 1
                if 0 in pickers_followers.values():  # 有拣货台没被选择
                    new_goal: tuple = random.choice(
                        get_keys_of(0, pickers_followers)
                    )
                else:  # 重复选择
                    new_goal: tuple = random.choice(
                        get_keys_of(
                            min(pickers_followers.values()),
                            pickers_followers
                        )
                    )

            # Case-2: 刚完成卸货， 准备回货架原位 --> 货架原位
            elif self.robots[robot_id].period_is(40):
                new_goal = self.robots[robot_id].shelf_home

            # Case-3: 正在卸货的机器人，目标设置为其原位
            elif self.robots[robot_id].period_is(31):
                new_goal = self.robots[robot_id].picker_pos

            # Case-4: 机器人空闲，准备前去拉货架 --> 新货架位置
            elif self.robots[robot_id].period_is(13):
                # 寻找空闲（没被连接）的货架
                free_shelves_id: List[int] = []
                busy_shelves_id: List[int] = []
                for i in self.robots_id:  # 被连接的货架
                    shelf_id = self.robots[i].shelf_id
                    if shelf_id is not None and shelf_id not in busy_shelves_id:
                        busy_shelves_id.append(shelf_id)
                    goal = self.robots[i].goal
                    if goal in self.init_shelves_pos.values():
                        shelf_id = get_keys_of(goal, self.init_shelves_pos)[0]
                        if shelf_id not in busy_shelves_id:
                            busy_shelves_id.append(shelf_id)

                for shelf_id in self.init_shelves_pos.keys():
                    if (shelf_id not in busy_shelves_id
                            and shelf_id not in new_goals.values()):
                        free_shelves_id.append(shelf_id)
                if len(free_shelves_id) == 0:
                    raise RuntimeError("There are no optional shelves.")
                tgt_shelf_id = random.choice(free_shelves_id)
                tgt_shelf_pos = self.init_shelves_pos[tgt_shelf_id]
                new_goal = tgt_shelf_pos
                if self.shelves_map[tgt_shelf_pos] < 1:
                    raise RuntimeError(
                        f"Set goal for R{robot_id:2d}, "
                        f"no shelf exists (={self.shelves_map[tgt_shelf_pos]}) "
                        f"for new shelf location {new_goal}."
                    )

            assert new_goal is not None, f"The new goal(={new_goal}) is None."
            # 移除旧目标
            last_goal: tuple = self.get_goal_of(robot_id)
            self.goals_map[last_goal].remove(robot_id)
            # 放置新目标
            new_goals.update({robot_id: new_goal})
            self.goals_map[new_goal].append(robot_id)
            self.robots[robot_id].goal = new_goal
            self.astar_plan(robot_id)
            if TEST_MODE:
                print('\t' * 2, f"- Set new goal{new_goal} for R{robot_id:2d}.")

    def collision_check(self, actions: Dict[int, int]) -> Tuple[dict, dict]:
        """碰撞检查及协调

        Returns:
            新位置、状态
        """
        # 'normal': 0, 'goal_done': 1, 'task_done': 2, 'err_env': -1,
        # 'err_line': -2, 'err_point': -3,

        _N, _TD, _GD, _EE, _EL, _EP = (MOVE['normal'], MOVE['task_done'],
                                       MOVE['goal_done'], MOVE['err_env'],
                                       MOVE['err_line'], MOVE['err_point'])

        rest_robots_id: List[int] = list(copy(self.robots_id))
        now_positions: Dict[int, tuple] = {i: self.get_pos_of(i)
                                           for i in rest_robots_id}
        ask_positions: Dict[int, tuple] = {
            i: plus(now_positions[i], match_direction(actions[i]))
            for i in rest_robots_id
        }
        goals: Dict[int, tuple] = {i: self.get_goal_of(i)
                                   for i in rest_robots_id}

        new_positions: Dict[int, tuple] = {}
        new_status: Dict[int, int] = {}

        def record(id_: int, pos_: tuple, status_: int) -> None:
            if TEST_MODE:
                print('\t' * 3, f"Approve R{id_:2d} position as {pos_}, "
                                f"move status as {get_keys_of(status_, MOVE)}.")
            new_positions[id_] = pos_
            new_status[id_] = status_
            if id_ in rest_robots_id:
                rest_robots_id.remove(id_)

        def reject_except(id_: int, pos_: tuple, status_: int) -> None:
            """递归拒绝对该位置的请求（该位置已批准给该机器人）"""
            # 找出还在请求新位置的机器人
            asking: Dict[int, tuple] = {}
            for i in self.robots_id:
                if i not in new_positions.keys():
                    asking[i] = ask_positions[i]
            # 找出请求该位置的机器人
            _followers: List[int] = get_keys_of(pos_, asking)
            if id_ in _followers:
                _followers.remove(id_)
            # 拒绝请求
            for _follower in _followers:
                _now_pos: tuple = now_positions[_follower]
                record(_follower, _now_pos, status_)
                if TEST_MODE:
                    print(
                        '\t' * 3,
                        f"Reject R{_follower: 2d} ask {pos_}, "
                        f"Approve as {_now_pos}."
                    )
                reject_except(_follower, _now_pos, status_)

        # 获取障碍地图
        static_obs_map: _NDArray = self.get_static_obs()
        if TEST_MODE:
            print('\t' + "+ Collision checking:")
            print('\t' * 2, f"Robots Now: {now_positions}")
            print('\t' * 2, f"Robots Ask: {ask_positions}")
            print('\t' * 2, '-' * 66)

        # Case-1: 静态障碍碰撞 ---------------------------------------------------
        # 1-1 越界 ------------///
        outs: List[int] = []
        if TEST_MODE:
            print('\t' * 2, f"- Out of bounds: {rest_robots_id}")
        for robot_id in copy(rest_robots_id):
            if self.out_of_map(ask_positions[robot_id]):
                now_pos: tuple = now_positions[robot_id]
                record(robot_id, now_pos, _EE)
                outs.append(robot_id)
                if TEST_MODE:
                    print('\t' * 3,
                          f"> R{robot_id:02d}-{now_positions[robot_id]}"
                          f", Request out of bounds "
                          f"{ask_positions[robot_id]} |-> "
                          f"STOP {new_positions[robot_id]}.")
                reject_except(robot_id, now_pos, _EE)
        if TEST_MODE:
            if outs:
                print('\t' * 2, f"  {outs}")
            else:
                print('\t' * 2, "   ==> None.")

        # *** 批准请求自身位置的机器人 *** -----------------------------------------
        if TEST_MODE:
            print('\t' * 2, f"- Ask self-pos: {rest_robots_id}")
        ask_self: List[int] = []
        if TEST_MODE:
            print('\t' * 3, f"- Unloading:")
        for robot_id in copy(rest_robots_id):
            now_pos: tuple = now_positions[robot_id]
            if 0 < self.robots[robot_id].unload_time <= self.unload_cost:
                record(robot_id, now_pos, _N)
                ask_self.append(robot_id)
                reject_except(robot_id, now_pos, _EE)
        if TEST_MODE:
            print('\t' * 3, f"- Main:")
        for robot_id in copy(rest_robots_id):
            ask_pos: tuple = ask_positions[robot_id]
            now_pos: tuple = now_positions[robot_id]
            goal: tuple = goals[robot_id]
            if ask_pos == now_pos:
                if ask_pos == goal:
                    record(robot_id, now_pos, _GD)
                else:
                    record(robot_id, now_pos, _N)  # 2023-06-30 _EE -> _N
                    reject_except(robot_id, now_pos, _EE)
                ask_self.append(robot_id)
            # elif 0 < self.robots[robot_id].unload_time <= self.unload_cost:
            #     record(robot_id, now_pos, _N)
            #     ask_self.append(robot_id)
        if TEST_MODE:
            if ask_self:
                print('\t' * 2, f"  {ask_self}")
            else:
                print('\t' * 2, "   ==> None.")

        # 1-3 请求进入静态障碍区域 -------------------------------///
        if TEST_MODE:
            print('\t' * 2,
                  f"- Collide with static obstacle:：{rest_robots_id}")
        move_in_obs: List[int] = []
        for robot_id in copy(rest_robots_id):
            ask_pos: tuple = ask_positions[robot_id]
            now_pos: tuple = now_positions[robot_id]
            goal: tuple = goals[robot_id]
            if static_obs_map[ask_pos] == OBSTACLE:
                if ask_pos == goal:
                    record(robot_id, ask_pos, _GD)
                    move_in_obs.append(robot_id)
                else:
                    record(robot_id, now_pos, _EE)
                    reject_except(robot_id, now_pos, _EE)
            elif static_obs_map[ask_pos] == FREE:
                if ask_pos in self.init_shelves_pos.values():
                    if ask_pos == goal:
                        record(robot_id, ask_pos, _GD)
                    else:
                        record(robot_id, now_pos, _EE)
                        reject_except(robot_id, now_pos, _EE)

        if TEST_MODE:
            if move_in_obs:
                print('\t' * 2, f"  {move_in_obs}")
            else:
                print('\t' * 2, "   ==> None.")

        # Case-2: 边碰撞 --------------------------------------------------------
        if TEST_MODE:
            print('\t' * 2, f"- LINE Collision: {rest_robots_id}")
        edge_collision = []
        for robot_id in copy(rest_robots_id):
            ask_pos: tuple = ask_positions[robot_id]
            now_pos: tuple = now_positions[robot_id]
            exist_robot: int = int(self.robots_map[ask_pos])
            if not exist_robot or exist_robot == robot_id:
                continue

            friend, friend_ask = exist_robot, ask_positions[exist_robot]
            # 发生边碰撞
            if friend_ask == now_pos:
                if robot_id not in edge_collision:
                    edge_collision.append(robot_id)
                if friend not in edge_collision:
                    edge_collision.append(friend)

                if robot_id not in new_status.keys():
                    record(robot_id, now_pos, _EL)
                if friend not in new_status.keys():
                    record(friend, now_positions[friend], _EL)
        if TEST_MODE:
            if edge_collision:
                print('\t' * 2, f"  {edge_collision}")
            else:
                print('\t' * 2, "   ==> None.")

        # Case-3: 点碰撞 --------------------------------------------------------
        if TEST_MODE:
            print('\t' * 2, f"- POINT Collision: {rest_robots_id}")
        point_collision = []
        for robot_id in copy(rest_robots_id):
            ask_pos: tuple = ask_positions[robot_id]
            now_pos: tuple = now_positions[robot_id]
            goal: tuple = goals[robot_id]

            others_ask = copy(ask_positions)
            others_ask.pop(robot_id)
            for _id in new_positions.keys():
                others_ask.pop(_id)

            if ask_pos in new_positions.values():
                point_collision.append(robot_id)
                record(robot_id, now_pos, _EP)
            elif ask_pos in others_ask.values():
                competitors = get_keys_of(ask_pos, others_ask)
                if robot_id < min(competitors):
                    status = _GD if ask_pos == goal else _N
                    record(robot_id, ask_pos, status)
                else:
                    record(robot_id, now_pos, _EP)
        if TEST_MODE:
            if point_collision:
                print('\t' * 2, f"  {point_collision}")
            else:
                print('\t' * 2, "   ==> None.")

        # Case-4: 正常移动
        if TEST_MODE:
            print('\t' * 2, f"- Normal Move: {rest_robots_id}")
        normal = []
        for robot_id in copy(rest_robots_id):
            ask_pos: tuple = ask_positions[robot_id]
            goal: tuple = goals[robot_id]
            status = _GD if ask_pos == goal else _N
            record(robot_id, ask_pos, status)
            normal.append(robot_id)
        if TEST_MODE:
            if normal:
                print('\t' * 2, f"  {normal}")
            else:
                print('\t' * 2, "   ==> None.")

        assert not rest_robots_id, f"还有机器人没有被碰撞检查{rest_robots_id}"
        if TEST_MODE:
            new_positions_ = {}
            sorted_new = sorted(new_positions.items(), key=lambda x: x[0])
            for new_pos_ in sorted_new:
                new_positions_.update({new_pos_[0]: new_pos_[1]})
            print('\t' * 2, '-' * 66)
            print('\t' * 2, f"Robots Now: {now_positions}")
            print('\t' * 2, f"Permitted : {new_positions_}")
        return new_positions, new_status

    def out_of_map(self, position: tuple) -> bool:
        """越界检查

        Notes:
            计算边界时，要去除墙体

        Returns:
            越界T/未越界F
        """
        x, y = position
        # 计算边界（右、下）
        R = self.mapper.shape[1] - 1
        B = self.mapper.shape[0] - 1

        return x <= 0 or x >= B or y <= 0 or y >= R

    def get_shelves(self) -> List[tuple]:
        """获取实时货架的位置坐标"""
        shelves = np.argwhere(self.shelves_map > 0).tolist()
        return [tuple(shelf) for shelf in shelves]

    def get_static_obs(self) -> _NDArray:
        obs_map = self.get_obs_map()
        for robot_id in self.robots_id:
            if self.robots[robot_id].shelf_id is not None:
                home = self.robots[robot_id].shelf_home
                obs_map[home] = FREE
        return obs_map

    def mark_adj_pos(self, center: tuple) -> list:
        def is_obstacle(position) -> bool:
            if 0 < self.base_map[position] <= 2:
                return True
            return False

        adj_pos: list = [None] * 4
        vectors = VECTORS.copy()
        vectors.remove((0, 0))

        if is_obstacle(center):  # obstacle
            return adj_pos

        for index, vector in enumerate(vectors):
            neighbour = plus(center, vector)
            if is_obstacle(neighbour):
                continue
            adj_pos[index] = neighbour
        return adj_pos

    def collect_adj_pos(self, center: tuple,
                        return_src_pos: bool = False,
                        contain_center: bool = True) -> List[tuple]:
        """Collect valid neighbours coordinate for the center pos.

        Warnings: arg `center` can't is `None`.
        """
        obstacle_map = self.get_obs_map(source=True)
        if center is None:
            raise ValueError(f"Argument `center` can't be None.")
        all_adj_pos = [plus(center, direct) for direct in VECTORS]
        if return_src_pos:
            return all_adj_pos
        # Filter out invalid positions.
        for adj_pos in all_adj_pos.copy():
            # case 1: The adjacent is out of world.
            if self.out_of_map(adj_pos):
                all_adj_pos.remove(adj_pos)
                continue
            # case 2: The adjacent position is obstacle.
            if obstacle_map[adj_pos] != FREE:
                all_adj_pos.remove(adj_pos)

        if not contain_center and center in all_adj_pos:
            all_adj_pos.remove(center)

        return all_adj_pos

    def is_inside_corridor(self, position: tuple) -> int:
        """Check if the robot is in a corridor

        Returns: 0-outside, others are corridors id.
        """
        # Read position info from corridor map.
        info = self.mapper.corridors_map[position]
        corridor_id, _ = info

        if corridor_id > 0:
            return corridor_id
        return 0

    def is_at_intersection(self, position: tuple) -> int:
        """Checks if the robot is at an intersection

        Returns:
            0-is not at intersection, 4-outside intersection, 3-inside intersection.
        """
        # Read position info from corridor map.
        info = self.mapper.corridors_map[position]
        _, status = info

        # _CR_OUT, _CR_IN, _EP, _IN, _FREE, _OBS = 4, 3, 2, 1, 0, -1
        if status == 4 or status == 3:
            return status
        return 0

    def scan_corridor(self, corridor_id,
                      skip_pos: Optional[tuple] = None
                      ) -> Tuple[Dict[str, dict], bool]:
        """Scan the robot info of corridor

        Returns:
            robots, is_clean
        """
        points = self.mapper.corridors[corridor_id]['all'].copy()

        robots_pos = {}  # {id: pos}
        normal_rs: Dict[int, tuple] = {}  # moved into the corridor normally
        leaved_rs: Dict[int, tuple] = {}  # just leaved a shelf in the corridor
        born_rs  : Dict[int, tuple] = {}  # born in the corridor
        is_clean: bool = False

        # check every point in corridor
        for point in points:
            if ((skip_pos is not None and point == skip_pos)
                    or (not self.robots_map[point])):
                continue

            robot_id = self.robots_map[point]
            last_pos = self.query_last_pos(robot_id)
            item = {robot_id: point}
            robots_pos.update(item)

            if last_pos is not None:
                # 1: Normal move into the corridor
                if last_pos not in self.init_shelves_pos.values():
                    normal_rs.update(item)
                # 2: Just leaved a shelf
                else:
                    leaved_rs.update(item)
            else:
                # 3: born in the corridor
                born_rs.update(item)
        # robots
        robots = {
            'all': robots_pos,
            'normal': normal_rs,
            'leaved': leaved_rs,
            'born': born_rs
        }

        # if the corridor is clean
        if not robots_pos:
            is_clean = True

        return robots, is_clean

    def scan_intersection(self, corridor_id,
                          skip_inse=None) -> Tuple[Dict[int, tuple], bool]:
        intersections = self.mapper.corridors[corridor_id]['crossroads']
        robots: Dict[int, tuple] = {}

        for intersection in intersections:
            if intersection == skip_inse:
                continue
            exist_robot = self.robots_map[intersection]
            if exist_robot:
                robots.update({exist_robot: intersection})

        is_clean: bool = False
        if not robots:
            is_clean = True

        return robots, is_clean

    def collect_corridor_pos(self, position: tuple) -> Tuple[List[tuple], ...]:
        """Collect all position of the corridor

        Warnings:
            You must ensure the `position` is in one corridor,
            here hasn't check process for it.

        Args:
            position: One of pos in the corridor.

        Returns:
            corridor's points & ends coordinate
        """
        neighbours: List[tuple] = self.collect_adj_pos(
            position, contain_center=False
        )
        corridor_points: List[tuple] = [position]
        corridor_ends: List[tuple] = []

        for neighbour in neighbours:
            direct = minus(neighbour, position)
            i = 0
            while True:
                check_pos = tuple((position[0] + i * direct[0],
                                   position[1] + i * direct[1]))
                if self.out_of_map(check_pos):
                    break
                check_pos_vp = self.collect_adj_pos(check_pos)
                if len(check_pos_vp) == 3:
                    corridor_points.append(check_pos)
                elif len(check_pos_vp) == 5:
                    corridor_ends.append(check_pos)
                    break
                i += 1

        return corridor_points, corridor_ends

    def query_last_pos(self, robot_id: int) -> Optional[tuple]:
        """Query the robot's last unique position"""
        history: List[tuple] = self.robots[robot_id].pos_history.copy()
        history.reverse()
        curr_pos = history.pop(0)  # del current pos
        if not history:
            return None
        # Get the robot's position, and check if it is in the history.
        # if curr_pos not in history:
        #     return None
        # Finding the last unique position.
        for past_pos in history:
            if past_pos != curr_pos:
                return past_pos
        return None

    def collect_robots_in(self, positions: List[tuple]) -> Dict[tuple, dict]:
        """Collect info about robot in the positions

        Returns:
            {
                'position':
                 {'id': id, 'last_pos': last_pos, 'direction': direction}
            }
        """
        robots = {}
        for position in positions:
            robot_id = self.robots_map[position]
            if not robot_id:
                continue
            # position exists robot
            robot_pos = self.get_pos_of(robot_id)
            info = {'id': robot_id}
            # Get `last pos` and `last direction`
            last_pos = self.query_last_pos(robot_id)
            if last_pos is not None:
                info.update({'last_pos': last_pos})
                last_dir = minus(robot_pos, last_pos)
                info.update({'last_dir': last_dir})
            robots[position] = info
        return robots


if __name__ == '__main__':
    w = World(Mapper(), 8)
    w.random_init = True
    w.reset()
    for idx, picker in w.pickers.items():
        print(f"{idx:02d}: {picker}")
    # for pos, robots in w.goals_map.items():
    #     if not robots:
    #         continue
    #     else:
    #         print(f"{pos}: {robots}")
    import matplotlib.pyplot as plt

    plt.imshow(w.get_obs_map(source=True))
    plt.show()
