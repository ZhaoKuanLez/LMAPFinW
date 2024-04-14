import math
import os
import numpy as np
from copy import copy
from matplotlib.colors import hsv_to_rgb
from datetime import (
    datetime,
    timedelta,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)


def plus(a: Tuple, b: Tuple) -> Tuple:
    """Addition of Tuples 'a + b'"""
    return tuple((a[0] + b[0], a[1] + b[1]))


def minus(a: Tuple, b: Tuple) -> Tuple:
    """Subtraction of Tuples 'a - b'"""
    return tuple((a[0] - b[0], a[1] - b[1]))


def get_keys_of(value_, dict_: Dict) -> List:
    """在字典中，根据值查询键（注意：返回值中的键值可能存在重复）"""
    return [k for k, v in dict_.items() if v == value_]


def match_direction(action: int) -> tuple:
    """转换动作为基本移动向量"""
    comparison_table = {
        0: (0,  0),
        1: (0, -1),
        2: (-1, 0),
        3: (0,  1),
        4: (1,  0)
    }
    if action in comparison_table.keys():
        return tuple(comparison_table[action])
    else:
        raise KeyError(f'Action-{action} is not in '
                       f'Action-Space {comparison_table}。')


def match_action(direction: Tuple) -> int:
    """将移动向量转换为动作"""
    comparison_table = {
        (0,  0): 0,
        (0, -1): 1,
        (-1, 0): 2,
        (0,  1): 3,
        (1,  0): 4
    }
    if direction in comparison_table.keys():
        return int(comparison_table[direction])
    else:
        raise KeyError(f'Move vector {direction} is not in '
                       f'Move-Vector-Space {comparison_table}。')


def adapt_screen_size(limits: Tuple, map_shape: tuple) -> Tuple[tuple, int]:
    """自适应窗口尺寸

    Args:
        limits: 窗口的宽、高极限尺寸
        map_shape: 地图形状

    Returns:
        窗口尺寸（宽, 高）、正方形单元格尺寸
    """
    W, H = limits
    rows, cols = map_shape

    # 尝试最宽适应
    cell_size = int(W / cols)
    if cell_size < 2:
        raise ValueError(f"The window width is not enough, "
                         f"the cell size is {cell_size}")
    if cell_size % 2 != 0:  # 调整为偶数
        cell_size -= 1
    width = cell_size * cols
    height = cell_size * rows
    if height > H:
        # 最高适应
        cell_size = int(H / rows)
        if cell_size < 2:
            raise ValueError(f"The window height is not enough, "
                             f"the cell size is {cell_size}")
        if cell_size % 2 != 0:  # 调整为偶数
            cell_size -= 1
        width = cell_size * cols
        height = cell_size * rows

    return (width, height), cell_size


def assign_colors(robots_num: int) -> Dict[int, np.ndarray]:
    """给团队每个成员指定唯一的颜色"""
    if robots_num < 1:
        raise ValueError(f"The robots num must gt 1.")
    return {
        i + 1: hsv_to_rgb(np.array([i / robots_num, 1, 1])) * 255
        for i in range(robots_num)
    }


def print_flag(
        flag: str,
        num: int,
        *,
        info: Optional[str] = None,
        pos: str = 'm',
) -> None:
    """打印行分隔符"""
    if info:
        if len(info) >= num:
            raise ValueError(f"标记数量过少")
    if pos not in ['l', 'm', 'r']:
        raise ValueError(f"信息位置只能是左（r）、中（m）、右（r）的其中之一")

    if info:
        if pos == 'm':
            print(f"{info:{flag}^{num}}")
        elif pos == 'l':
            print(f"{info:{flag}<{num}}")
        elif pos == 'r':
            print(f"{info:{flag}>{num}}")
    else:
        print(f'{flag}' * num)


def get_repeat_positions(pos, print_info=False):
    if pos is None or pos == []:
        raise ValueError("要检测的列表为空！")
    if print_info:
        print(f"Robot Num: {len(pos)}")
        print(f"Diff pos : {len(set(pos))}")
    IDs = [ID for ID in range(1, len(pos) + 1)]
    positions = {
        ID: pos[ID - 1]
        for ID in IDs
    }
    repeats = {}
    for ID in IDs:

        robot_pos = positions[ID]
        others_pos: Dict = copy(positions)
        others_pos.pop(ID)
        repeat_ids = [ID]
        for other_id, other_pos in others_pos.items():
            if tuple(robot_pos) == tuple(other_pos):
                repeat_ids.append(other_id)
        if len(repeat_ids) > 1:
            repeat_ids.reverse()
            repeats.update({robot_pos: repeat_ids})
    if print_info:
        print_flag('-', 10)
        print("Repeat info: ")
        if repeats:
            for overlap_pos, overlap_ids in repeats.items():
                print(f"\tPos: {overlap_pos}", end='')
                print(f"\tIDs: {overlap_ids}")
        else:
            print("\bNo Overlap！")
    if len(repeats) > 0:
        return repeats
    else:
        return None


def clean_document(document_path: str) -> None:
    files_path = [
        os.path.join(document_path, file)
        for file in os.listdir(document_path)
    ]
    for file in files_path:
        if os.path.isfile(file):
            os.remove(file)


def format_time(second_time: int) -> str:
    s = timedelta(seconds=int(second_time))
    d = datetime(1, 1, 1) + s
    t = ''
    if d.day - 1:
        t += f'{(d.day - 1):d}d '
        t += f'{d.hour:d}:'
    else:
        t += f'{d.hour:d}:'
    t += f"{d.minute:d}:{d.second:d}"

    return t


def get_distance(from_: tuple, to_: tuple) -> Tuple[float, ...]:
    """计算两点之间的距离

    Return:
        delta_x, delta_y, distance
    """
    delta_x = to_[0] - from_[0]
    delta_y = to_[1] - from_[1]
    distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
    return delta_x, delta_y, distance


def get_graph(observations, robot_id=None):
    if robot_id is None or robot_id == 0:
        return None
    observation = observations[robot_id]

    center_pos = observation['pos']
    leaves = observation['robots']

    leaves_pos = []
    for leaf_id, leaf_info in leaves.items():
        leaves_pos.append(leaf_info['pos'])

    return {'pos': center_pos, 'leaves': leaves_pos}


def get_grads(network):
    return [param.grad for param in network.parameters()]


def maintain(record_list: list, max_len: int = 30):
    while True:
        if len(record_list) > max_len:
            record_list.pop(0)
        else:
            return record_list


if __name__ == '__main__':
    res = get_distance((0, 0), (3, 4))
    print(res)
