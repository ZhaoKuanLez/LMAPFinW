import numpy as np
import pygame as pg
from abc import ABC
from gym import Env
from pygame import (
    Surface,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import env.rendering as rendering
from env.mapper import Mapper
from env.eye import Eye
from env._world import World
from env.utils import (
    adapt_screen_size,
    assign_colors,
    get_keys_of,
)
from env.constants import (
    ACTIONS,
    COLORS,
    MOVE,
    REWARDS,
    STAGE,
    TEST_MODE,
    WINDOW_LIMIT,
    WORK,
)
_NDArray = np.ndarray


class MAPFinW(Env, ABC, object):

    metadata = {
        'render_mode': ['human', 'rgb_array'],
        'render_fps': 30
    }
    world: Optional[World] = None
    unloading: List = []

    w_cargos: int = 0
    w_rewards: float = 0
    w_tasks: int = 0
    w_goals: int = 0

    # 渲染
    screen = None
    clock = None
    screen_size: Optional[tuple] = None
    cell_size: Optional[int] = None
    colors: Dict[int, _NDArray] = None

    print_log: bool = False

    def __init__(self, mapper: Mapper, eye: Eye, robots_num: int = 12,
                 unload_cost: int = 3, render_mode: Optional[str] = None):
        self.mapper = mapper
        self.eye = eye
        self.robots_num = robots_num
        self.unload_cost = unload_cost
        self.render_mode = render_mode

    def reset(self, **kwargs) -> Dict:
        """重置环境

        Args:
            kwargs: 'rows': int, 'cols': int, 'length': int, 'pickers': int,
                    'print_log': bool

        Returns:
            群体观测
        """
        self.world = World(mapper=self.mapper, robots_num=self.robots_num)

        world_args = {'unload_cost': self.unload_cost}
        if kwargs is not None:
            # 'rows', 'cols', 'length', 'pickers', 'print_log'
            if 'rows' in kwargs:
                world_args['rows'] = kwargs.get('rows')
            if 'cols' in kwargs:
                world_args['cols'] = kwargs.get('cols')
            if 'length' in kwargs:
                world_args['length'] = kwargs.get('length')
            if 'pickers' in kwargs:
                world_args['pickers'] = kwargs.get('pickers')
            if 'print_log' in kwargs:
                self.print_log = kwargs.get('print_log')
            if 'random_init' in kwargs:
                world_args['random_init'] = kwargs.get('random_init')
        self.world.reset(**world_args)
        self.eye.set_world(self.world)

        self.w_cargos, self.w_rewards = 0, 0
        self.w_tasks, self.w_goals = 0, 0
        self.unloading = []
        self.screen_size, self.cell_size = adapt_screen_size(
            WINDOW_LIMIT,
            self.world.mapper.shape
        )
        self.colors = assign_colors(self.robots_num)
        return self.eye.observe_group()

    def step(self, actions: dict, index: int = 0) -> Tuple[dict, ...]:
        """执行一步

        Args:
            actions: 所有机器人的动作
            index: 时间步索引

        Returns:
            新观测值、每个机器人的奖励、移动状态、其他信息
        """
        _STOP = ACTIONS['stop']
        _GD = MOVE['goal_done']
        _TD = MOVE['task_done']
        _ACH = WORK['attach']
        _DTH = WORK['detach']

        # 统计参数
        goals_done = {robot: False for robot in self.world.robots_id}
        tasks_done = {robot: False for robot in self.world.robots_id}

        # 检查参数
        for robot_id in self.world.robots_id:
            if robot_id not in actions:
                actions[robot_id] = _STOP
            if actions[robot_id] not in ACTIONS.values():
                raise ValueError(f"Action-{actions[robot_id]} "
                                 f"does not in Action-Space.")

        # 碰撞检查
        now_positions = {i: self.world.get_pos_of(i)
                         for i in self.world.robots_id}
        new_positions, new_status = self.world.collision_check(actions)
        self.w_goals += list(
            np.array(list(new_status.values())) > 0
        ).count(True)

        # 对 Stage-31 做特殊处理
        for robot_id in self.unloading:
            new_status[robot_id] = _GD
        # 将完成任务的机器人状态修正为 task_done
        for robot_id in self.world.robots_id:
            reach_goal = new_status[robot_id] == _GD
            home_stage = self.world.robots[robot_id].period_is(40)
            no_unload = self.world.robots[robot_id].unload_time == 0
            if reach_goal and no_unload:
                goals_done[robot_id] = True
            if reach_goal and home_stage:
                tasks_done[robot_id] = True
                new_status[robot_id] = _TD
                self.w_tasks += 1

        # 执行动作
        if TEST_MODE:
            print('\t', "+ Execute action: ")
        request_goals: List[int] = []  # 请求新目标的机器人ID
        for robot_id in self.world.robots_id:
            now_pos: tuple = self.world.get_pos_of(robot_id)
            new_pos: tuple = new_positions[robot_id]
            status: int = new_status[robot_id]
            if robot_id not in self.unloading:
                self.world.robots[robot_id].move(new_pos, status)
                self.world.robots[robot_id].move_status = status
                self.world.robots_map[now_pos] = 0
                self.world.robots_map[new_pos] = robot_id
                if self.world.robots[robot_id].work_status == _ACH:
                    shelf_id = self.world.robots[robot_id].shelf_id
                    self.world.shelves_map[now_pos] = 0
                    self.world.shelves_map[new_pos] = shelf_id

            # 状态转移
            if status == _GD:  # Stage 13/22/31
                robot = self.world.robots[robot_id]
                # Stage 13 -> 22
                if robot.period_is(13):
                    self.world.robots[robot_id].period = STAGE[22]
                    self.world.robots[robot_id].work_status = _ACH
                    # 连接货架
                    shelf_id: int = self.world.shelves_map[new_pos]
                    if shelf_id <= 0:
                        raise RuntimeError(f"所处目标位置{new_pos}没有货架！")
                    self.world.robots[robot_id].shelf_id = shelf_id
                    self.world.robots[robot_id].shelf_home = new_pos

                    request_goals.append(robot_id)
                # Stage 22 -> 31
                elif robot.period_is(22):
                    self.world.robots[robot_id].period = STAGE[31]
                    # 占据拣货台
                    pickers = self.world.mapper.pickers_spot_pos
                    if new_pos not in pickers.values():
                        raise RuntimeError(f"所处目标位置{new_pos}不是拣货台！")
                    picker_id = int(get_keys_of(new_pos, pickers)[0])
                    self.world.robots[robot_id].picker_id = picker_id
                    self.world.robots[robot_id].picker_pos = new_pos
                    # 开始卸货
                    self.unloading.append(robot_id)
                    self.world.robots[robot_id].unload_time += 1
                    self.world.pickers[picker_id].pick()

                    request_goals.append(robot_id)
                # Stage 31 -> 40
                elif robot.period_is(31):
                    if robot.unload_time >= self.unload_cost - 1:
                        self.world.robots[robot_id].period = STAGE[40]
                        self.world.robots[robot_id].unload_time = 0
                        self.unloading.remove(robot_id)
                        # 清除机器人保存的拣货台信息
                        self.world.robots[robot_id].picker_id = None
                        self.world.robots[robot_id].picker_pos = None
                        # 请求回家
                        request_goals.append(robot_id)
                    else:
                        self.world.robots[robot_id].unload_time += 1
                else:
                    raise RuntimeError("未知状态！")

            # 负载状态切换
            if status == _TD:
                # 分离货架
                self.world.robots[robot_id].work_status = _DTH
                self.world.robots[robot_id].passed_cargos += 1
                self.w_cargos += 1
                # 重置机器人的部分参数
                self.world.robots[robot_id].reset()

                request_goals.append(robot_id)

            if TEST_MODE or self.print_log:
                old_pos = now_positions[robot_id]
                goal = self.world.get_goal_of(robot_id)
                robot = self.world.robots[robot_id]
                shf_id = robot.shelf_id
                shf_pos = robot.shelf_home
                picker_id = robot.picker_id
                picker_pos = robot.picker_pos
                print('\t' * 2, end='')
                print(f"> R{robot_id:2d} ", end='')
                print(f"- P({old_pos[0]:02d}, {old_pos[1]:02d})->", end='')
                print(f"({new_pos[0]:02d}, {new_pos[1]:02d}) ", end='')
                print(f"- G({goal[0]:02d}, {goal[1]:02d}) ", end='')
                print(f"- MS({get_keys_of(status, MOVE)[0]:<9}) ", end='')
                print(f"- WS({get_keys_of(robot.work_status, WORK)[0]:<6}) ",
                      end='')
                if shf_id is not None:
                    print(f"- SHF(id: {robot.shelf_id:3d}, ", end='')
                    print(f"home: ({shf_pos[0]:02d}, {shf_pos[1]:02d})) ",
                          end='')
                if picker_id is not None:
                    print(f"- PIK(id: {robot.picker_id:2d}, ", end='')
                    print(f"pos: ({picker_pos[0]:2d}, {picker_pos[1]:2d})) ",
                          end='')
                print(f"- Ul: {robot.unload_time:d} ", end='')
                print(f"- Cy: {list(robot.period.values())} ", end='')
                print(f"-")

        # 计算奖励
        rewards = {}
        for robot_id in self.world.robots_id:
            if robot_id not in self.unloading:
                reward = self.return_reward(robot_id)
                rewards[robot_id] = reward
                self.w_rewards += reward
            else:
                reward = 0
                rewards[robot_id] = reward
                self.w_rewards += reward

        # 放置新目标
        if request_goals:
            self.world.set_goals(request_goals)

        # 返回
        observations: dict = self.eye.observe_group()
        infos = {
            'goals_done': goals_done,
            'tasks_done': tasks_done,
            'group_reward': self.w_rewards,
            'group_cargos': self.w_cargos,
            'group_goals': self.w_goals,
            'group_tasks': self.w_tasks,
            'team_status': new_status,
            'pickers_pressure': len(self.unloading) / len(self.world.pickers)
        }
        return observations, rewards, new_status, infos

    def render(self, episode: int = 0, step: int = 0, *, reward: float = 0,
               cargos: int = 0, tasks: int = 0, goals: int = 0,
               connect: bool = False,
               pickers_pressure: float = 0, graph=None) -> Optional[_NDArray]:
        if self.screen is None:
            pg.init()
            if self.render_mode == 'human':
                pg.display.init()
                self.screen = pg.display.set_mode(size=self.screen_size)
            else:
                self.screen = Surface(self.screen_size)
            self.screen.fill('white')
            pg.display.set_caption('MAPF in Warehouse')
        if self.clock is None:
            self.clock = pg.time.Clock()

        # 绘制当前帧
        surf = Surface(self.screen_size)
        surf.fill('white')

        _SIZE = {'screen': self.screen_size, 'cell': self.cell_size}
        rendering.draw_grid(_SIZE, surf)
        rendering.draw_wall(_SIZE, surf)
        rendering.draw_tick(_SIZE, self.world.mapper.shape, surf)
        rendering.draw_wait(
            self.world.mapper.wait_spots, COLORS['wait'], self.cell_size, surf
        )
        rendering.draw_spots(
            self.world.mapper.robots_spot_pos, COLORS['shelf_spot'],
            self.cell_size, surf
        )
        rendering.draw_spots(
            self.world.mapper.shelves_spot_pos, COLORS['shelf_spot'],
            self.cell_size, surf
        )
        rendering.draw_pickers(self.world.pickers, self.cell_size, surf)
        rendering.rectangle(
            [self.cell_size, self.cell_size,
             self.screen_size[0] - 2 * self.cell_size,
             self.screen_size[1] - 2 * self.cell_size], surf, (0, 0, 0)
        )
        if connect:
            starts = [self.world.get_pos_of(i) for i in self.world.robots_id]
            ends = [self.world.get_goal_of(i) for i in self.world.robots_id]
            rendering.guide_line(starts, ends, self.cell_size, surf)
        if graph:
            rendering.graph(graph, self.cell_size, surf)
        rendering.draw_robots(
            {i: self.world.get_pos_of(i) for i in self.world.robots_id},
            self.colors, self.cell_size, surf
        )
        shelves = self.world.get_shelves()
        rendering.draw_shelves(shelves, self.cell_size, surf)
        rendering.draw_goals(
            {i: self.world.get_goal_of(i) for i in self.world.robots_id},
            self.colors, self.cell_size, surf
        )

        information: Dict[str, Any] = {
            'rows': self.world.mapper.rows, 'cols': self.world.mapper.cols,
            'episode': episode, 'step': step, 'reward': reward,
            'cargo': cargos, 'goal': goals ,
            'pbr': pickers_pressure * 100,  # picker-busy-rate
            'robot': self.robots_num,
            'picker': len(self.world.mapper.pickers_spot_pos),
            'shelf': len(self.world.init_shelves_pos),
            'shape': self.world.mapper.shape,
            'unload_cost': self.unload_cost,
        }
        rendering.dynamical_info(information, self.cell_size,
                                 self.screen_size, surf)

        self.screen.blit(surf, (0, 0))
        # Keep window is showing. --Open it when Test, and Close last part.
        # rendering.render_test()

        if self.render_mode == 'human':
            pg.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pg.display.flip()
        elif self.render_mode == 'rgb_array':
            return np.transpose(
                np.array(pg.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            self.screen = None
            pg.display.quit()
            pg.quit()
            return 

    def return_reward(self, robot_id: int) -> float:
        move_status = self.world.robots[robot_id].move_status
        if move_status == MOVE['normal']:
            return REWARDS['normal']
        elif (
                move_status == MOVE['err_env']
                or move_status == MOVE['err_line']
                or move_status == MOVE['err_point']
        ):
            return REWARDS['normal'] + REWARDS['errors']
        elif move_status == MOVE['goal_done']:
            if robot_id in self.unloading:
                return REWARDS['unloading']
            else:
                return REWARDS['normal'] + REWARDS['goal_done']
        elif move_status == MOVE['task_done']:
            return REWARDS['normal'] + REWARDS['task_done']
        else:
            raise ValueError(f"R{robot_id:2d}'s move state {move_status} "
                             f"does not exist in Action-Space!")


if __name__ == '__main__':
    mapf = MAPFinW(
        mapper=Mapper(),
        eye=Eye(),
        render_mode='human'
    )
    mapf.reset()
    mapf.render()
