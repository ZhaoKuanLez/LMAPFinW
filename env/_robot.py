import numpy as np
from copy import copy
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from colorama import (
    Fore,
    Style
)

from env.constants import (
    MOVE,
    PERIOD,
    VECTORS,
    WORK,
)
from env.utils import (
    get_keys_of,
    maintain,
    match_action,
    minus,
    plus,
)

_NDArray = np.ndarray


class Robot(object):
    pos: Optional[tuple] = None
    pos_history: Optional[List[tuple]] = []
    goal: Optional[tuple] = None
    directory: Optional[tuple] = None
    dir_history: Optional[List[tuple]] = [(None, None)]
    act_history: Optional[List[int]] = []

    astar_path: Optional[List] = None
    astar_map: Optional[_NDArray] = None

    period: Dict = copy(PERIOD)
    move_status: Optional[int] = MOVE['normal']
    work_status: Optional[int] = WORK['detach']

    shelf_id: Optional[int] = None
    shelf_home: Optional[tuple] = None
    picker_id: Optional[tuple] = None
    picker_pos: Optional[tuple] = None
    unload_time: Optional[int] = 0

    waiting: Optional[bool] = None
    waiting_pos: Optional[tuple] = None
    waiting_time: Optional[int] = 0

    odometers: Optional[int] = 0
    passed_cargos: Optional[int] = 0

    def __init__(self, id_=None):
        self.id = id_

    def reset(self):
        self.period = copy(PERIOD)
        self.unload_time = 0
        self.shelf_id = None
        self.shelf_home = None
        self.picker_id = None
        self.picker_pos = None
        self.work_status = WORK['detach']

    def move(self, to_pos: Tuple, move_status: int) -> None:
        if to_pos is None:
            raise ValueError(f"New position is None!")

        if self.odometers:
            neighbors = [plus(self.pos, vector) for vector in VECTORS]
            if to_pos not in neighbors:
                raise ValueError(
                    f"The new position {to_pos} is not a neighbor of "
                    f"the current position {neighbors}."
                )
        # 更新
        if self.odometers:
            dir_ = minus(to_pos, self.pos)
            self.dir_history.append(dir_)
            self.dir_history = maintain(self.dir_history)
            self.act_history.append(match_action(dir_))
            self.act_history = maintain(self.act_history)
        self.pos = to_pos
        self.pos_history.append(to_pos)
        self.pos_history = maintain(self.pos_history)
        self.move_status = move_status
        self.odometers += 1

    def period_is(self, code: int, return_onehot=False) -> Union[bool, list]:
        period = self.period

        if code == 13:
            condition_1 = period['idle'] == True
            condition_2 = period['to_picker'] == False
            condition_3 = period['unloading'] == False
            condition_4 = period['to_home'] == False
            if return_onehot:
                return [1, 0, 0, 0]
            return all((condition_1, condition_2, condition_3, condition_4))

        elif code == 22:
            condition_1 = period['idle'] == True
            condition_2 = period['to_picker'] == True
            condition_3 = period['unloading'] == False
            condition_4 = period['to_home'] == False
            if return_onehot:
                return [1, 1, 0, 0]
            return all((condition_1, condition_2, condition_3, condition_4))

        elif code == 31:
            condition_1 = period['idle'] == True
            condition_2 = period['to_picker'] == True
            condition_3 = period['unloading'] == True
            condition_4 = period['to_home'] == False
            if return_onehot:
                return [1, 1, 1, 0]
            return all((condition_1, condition_2, condition_3, condition_4))

        elif code == 40:
            condition_1 = period['idle'] == True
            condition_2 = period['to_picker'] == True
            condition_3 = period['unloading'] == True
            condition_4 = period['to_home'] == True
            if return_onehot:
                return [1, 1, 1, 1]
            return all((condition_1, condition_2, condition_3, condition_4))

        else:
            raise ValueError(f'Stage code {code} does not exist.')

    def __str__(self):
        print(Style.BRIGHT + Fore.CYAN + f"Robot {self.id}" + Style.RESET_ALL)
        print('-' * 20)
        if self.move_status is not None and self.work_status is not None:
            work_status = get_keys_of(self.work_status, WORK)[0]
            move_status = get_keys_of(self.move_status, MOVE)[0]
            return (
                f"position: {self.pos}\n"
                f"pos_history: {self.pos_history}\n"
                f"goal: {self.goal}\n"
                f"period: {self.period}\n"
                f"shelf: ID-{self.shelf_id}, home-{self.shelf_home}\n"
                f"picker: ID-{self.picker_id}, pos-{self.picker_pos}\n"
                f"move status: {move_status}\n"
                f"work status: {work_status}\n\n"
            )
        else:
            return (
                f"position: {self.pos}\n"
                f"pos_history: {self.pos_history}\n"
                f"goal: {self.goal}\n"
                f"period: {self.period}\n"
                f"shelf: ID-{self.shelf_id}, home-{self.shelf_home}\n"
                f"picker: ID-{self.picker_id}, pos-{self.picker_pos}\n\n"
            )
