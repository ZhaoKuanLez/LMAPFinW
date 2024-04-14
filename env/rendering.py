import pygame
from platform import platform
from pygame import (
    gfxdraw,
    Rect,
    Surface,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from env.constants import (
    COLORS,
)
from env._picker import Picker


def draw_grid(size: dict, surf: Surface):
    screen_size, cell_size = size['screen'], size['cell']
    color = COLORS['grid']
    # 横线
    hl_num = int(screen_size[1] / cell_size) + 1
    ys = [y * cell_size for y in range(hl_num)]
    for y in ys:
        gfxdraw.hline(surf, 0, screen_size[0], y, color)

    # 竖线
    vl_num = int(screen_size[0] / cell_size) + 1
    xs = [x * cell_size for x in range(vl_num)]
    for x in xs:
        gfxdraw.vline(surf, x, 0, screen_size[1], color)


def draw_wall(size: dict, surf: Surface) -> None:
    color = COLORS['wall']
    screen_size, cell_size = size['screen'], size['cell']
    # Up
    gfxdraw.box(surf, [0, 0, screen_size[0], cell_size], color)
    # Bottom
    gfxdraw.box(surf, [0, screen_size[1] - cell_size, screen_size[0],
                       cell_size], color)
    # Left
    gfxdraw.box(surf, [0, 0, cell_size, screen_size[1]], color)
    # Right
    gfxdraw.box(surf,
                [screen_size[0] - cell_size, 0, cell_size, screen_size[1]],
                color)


def draw_tick(size: dict, shape: tuple, surf: Surface) -> None:
    """绘制刻度"""
    screen_size, cell_size = size['screen'], size['cell']
    f_color = COLORS['tick']
    b_color = COLORS['wall']

    if 'macOS' in platform():
        font = pygame.font.SysFont('PingFang', 10)
    elif 'Linux' in platform():
        font = pygame.font.SysFont('DejaVu Sans Mono', 10)
    else:
        font = pygame.font.Font(None, 10)

    h_ticks = Surface((screen_size[0], cell_size))
    h_ticks.fill(b_color)
    v_ticks = Surface((cell_size, screen_size[1]))
    v_ticks.fill(b_color)

    offset = int(cell_size / 2)
    hts_x = [(i * cell_size + offset) for i in range(shape[1] - 1)]
    hts_y = [offset] * hts_x.__len__()

    vts_y = [(i * cell_size + offset) for i in range(shape[0] - 1)]
    vts_x = [offset] * vts_y.__len__()

    i = 0
    for center in zip(hts_x, hts_y):
        if i == 0:
            i += 1
            continue
        tick = font.render(str(i), True, f_color, b_color)
        tick_rect = tick.get_rect()
        tick_rect.center = center
        h_ticks.blit(tick, tick_rect)
        i += 1

    j = 0
    for center in zip(vts_x, vts_y):
        if j == 0:
            j += 1
            continue
        tick = font.render(str(j), True, f_color, b_color)
        tick_rect = tick.get_rect()
        tick_rect.center = center
        v_ticks.blit(tick, tick_rect)
        j += 1

    surf.blit(h_ticks, (0, 0))
    surf.blit(v_ticks, (0, 0))


def _draw_spot(center: tuple, color: tuple, surf: Surface, cell_size: int,
               text: Optional[str] = None, text_color: Optional[tuple] = None,
               with_bound: bool = False):
    font = None
    if text:
        if 'macOS' in platform():
            font = pygame.font.SysFont('PingFang', 10, bold=True)
        elif 'Linux' in platform():
            font = pygame.font.SysFont('DejaVu Sans Mono',
                                       10, bold=True)
        else:
            font = pygame.font.Font(None, 10)

    _surf = Surface((cell_size, cell_size))
    _surf.fill(color)
    if with_bound:
        gfxdraw.rectangle(_surf, _surf.get_rect(), (0, 0, 0))

    rect = _surf.get_rect()
    rect.center = center
    surf.blit(_surf, rect)

    if text is not None:
        if text_color is None:
            text_color = 'black'
        text_render = font.render(text, True, text_color, color)
        text_rect = text_render.get_rect()
        text_rect.center = center
        surf.blit(text_render, text_rect)


def draw_spots(spots: dict, color: tuple, cell_size: int,
               surf: Surface) -> None:
    offset = cell_size / 2
    for _, robot_spot in spots.items():
        x, y = robot_spot
        spot_center = (y * cell_size + offset, x * cell_size + offset)
        _draw_spot(spot_center, color, surf, cell_size)


def draw_wait(wait_spots: Dict[int, List[tuple]], color: tuple, cell_size: int,
              surf: Surface) -> None:
    def trans(tick_pos: Tuple[int, ...]) -> Tuple[int, int]:
        offset = int(cell_size / 2)
        x, y = tick_pos
        return y * cell_size + offset, x * cell_size + offset
    for _, pos_lst in wait_spots.items():
        for src_pos in pos_lst:
            _draw_spot(trans(src_pos), color, surf, cell_size)


def draw_pickers(pickers: Dict[int, Picker], cell_size: int,
                 surf: Surface):
    offset = cell_size / 2
    for _, picker in pickers.items():
        cargos = picker.cargos
        x, y = picker.pos
        picker_center = (y * cell_size + offset, x * cell_size + offset)
        _draw_spot(picker_center, COLORS['picker'], surf, cell_size,
                   str(cargos), (255, 255, 255), True)


def rectangle(rect: Union[Rect, tuple, list], surf: Surface,
              color: tuple = (0, 0, 0)) -> None:
    gfxdraw.rectangle(surf, rect, color)


def guide_line(starts: list, ends: list, cell_size: int, surf: Surface):
    """机器人与目标位置间的连接线

    Notes:
        机器人顺序，起点、终点必须一致
    """

    def trans(tick_pos: Tuple[int, ...]) -> Tuple[int, int]:
        offset = int(cell_size / 2)
        x, y = tick_pos
        return y * cell_size + offset, x * cell_size + offset

    for start, end in zip(starts, ends):
        pygame.draw.line(surf, COLORS['connect'], trans(start), trans(end))


def graph(data: dict, cell_size: int, surf: Surface) -> None:
    """绘制图

    data: {'center': pos, 'leaves': {'robot_id': pos}}
    """

    def trans(tick_pos: Tuple[int, ...]) -> Tuple[int, int]:
        offset = int(cell_size / 2)
        x, y = tick_pos
        return y * cell_size + offset, x * cell_size + offset

    center_pos: Tuple[int, int] = trans(data['pos'])
    leaves_pos: List[Tuple[float, float]] = []
    # for leaf_id in data['robots'].keys():
    #     leaves_pos.append(trans(data['robots'][leaf_id]['pos']))
    for leaf_pos in data['leaves']:
        leaves_pos.append(trans(leaf_pos))

    for leaf_pos in leaves_pos:
        pygame.draw.line(surf, COLORS['graph'], center_pos, leaf_pos, width=4)


def draw_robots(robots_pos: dict, colors: dict, cell_size: int, surf: Surface):
    def trans(tick_pos: Tuple[int, ...]) -> Tuple[int, int]:
        offset = int(cell_size / 2)
        x, y = tick_pos
        return y * cell_size + offset, x * cell_size + offset

    def robot(id_: int, color: tuple, center: Tuple[int, int]):
        font, bold = None, True
        if 'macOS' in platform():
            font = pygame.font.SysFont('PingFang', 8, bold)
        elif 'Linux' in platform():
            font = pygame.font.SysFont('DejaVu Sans Mono', 8, bold)
        else:
            font = pygame.font.Font(None, 8)
        r = int(cell_size / 2) - 1

        gfxdraw.filled_circle(surf, center[0], center[1], r, color)
        gfxdraw.aacircle(surf, center[0], center[1], r, (0, 0, 0))

        id_text = font.render(str(id_), True, (0, 0, 0), color)
        id_rect = id_text.get_rect()
        id_rect.center = center
        surf.blit(id_text, id_rect)

    for robot_id, robot_pos in robots_pos.items():
        robot(robot_id, colors[robot_id], trans(robot_pos))


def draw_shelves(shelves_pos: List[tuple], cell_size: int, surf: Surface):
    def trans(tick_pos: Tuple[int, ...]) -> Tuple[int, int]:
        offset = int(cell_size / 2)
        x, y = tick_pos
        return y * cell_size + offset, x * cell_size + offset

    def shelf(center: Tuple[int, int]):
        shelf_surf = Surface((cell_size, cell_size))
        shelf_surf.fill(COLORS['shelf'])
        shelf_surf.set_alpha(128)

        gfxdraw.rectangle(shelf_surf,
                          [0, 0, cell_size, cell_size], (0, 0, 0))
        gfxdraw.line(shelf_surf, 0, 0,
                     cell_size, cell_size, (0, 0, 0))
        gfxdraw.line(shelf_surf, cell_size,
                     0, 0, cell_size, (0, 0, 0))

        shelf_rect = shelf_surf.get_rect()
        shelf_rect.center = center
        surf.blit(shelf_surf, shelf_rect)

    for shelf_pos in shelves_pos:
        shelf(trans(shelf_pos))


def draw_goals(goals: dict, colors: dict, cell_size: int, surf: Surface):
    def trans(tick_pos: Tuple[int, ...]) -> Tuple[int, int]:
        offset = int(cell_size / 2)
        x, y = tick_pos
        return y * cell_size + offset, x * cell_size + offset

    def goal(center: Tuple[int, int], color: tuple):
        goal_surf = Surface((cell_size, cell_size))
        goal_surf.fill(color)
        goal_surf.set_alpha(128)

        goal_rect = goal_surf.get_rect()
        goal_rect.center = center
        surf.blit(goal_surf, goal_rect)

    for robot_id, robot_goal in goals.items():
        goal(trans(robot_goal), colors[robot_id])


def dynamical_info(infos: dict, cell_size: int, screen_size: tuple,
                   surf: Surface):
    """添加动态信息到界面

    Args:
        infos: 'step', 'episode', 'reward', 'cargo', 'task', 'goal'
        cell_size: 单元格尺寸
        screen_size: 窗口尺寸
        surf: 帧
    """
    key_bold, value_bold = False, True
    if 'macOS' in platform():
        font = pygame.font.SysFont('PingFang', 10, key_bold)
        font_b = pygame.font.SysFont('PingFang', 10, value_bold)
    elif 'Linux' in platform():
        font = pygame.font.SysFont('DejaVu Sans Mono', 10, key_bold)
        font_b = pygame.font.SysFont('DejaVu Sans Mono',
                                     10, value_bold)
    else:
        font = font_b = pygame.font.Font(None, 10)

    def blit_kv(key: str, key_pos: tuple, value: str, value_pos: tuple):
        key_text = font.render('| ' + key + ': ', True,
                               'black', COLORS['wall'])
        # value_text = font_b.render(value, True, 'black', COLORS['wall'])
        value_text = font_b.render(value, True,
                                   'red', 'white')
        surf.blit(key_text, key_pos)
        surf.blit(value_text, value_pos)

    cols = infos['cols']
    if cols <= 10:
        distances_1 = [1, 3, 7, 9, 11, 14, 16, 18, 21, 23, 25, 27, 31]
    elif 10 < cols <= 15:
        distances_1 = [1, 3,  7, 10, 12, 16, 19, 22, 25, 28, 30, 32, 45]
    else:
        distances_1 = [1, 3, 7, 9, 11, 14, 16, 18, 21, 23, 25, 27, 31]
    y = screen_size[1] - cell_size

    ep_k_pos = (distances_1[0] * cell_size, y)
    ep_v_pos = (distances_1[1] * cell_size, y)
    blit_kv('EP.', ep_k_pos, f"{infos['episode']:10d}", ep_v_pos)

    st_k_pos = (distances_1[2] * cell_size, y)
    st_v_pos = (distances_1[3] * cell_size, y)
    blit_kv('STEP', st_k_pos, f"{infos['step']:3d}", st_v_pos)

    r_k_pos = (distances_1[4] * cell_size, y)
    r_v_pos = (distances_1[5] * cell_size, y)
    blit_kv('Reward', r_k_pos, f"{infos['reward']:4.2f}", r_v_pos)

    c_k_pos = (distances_1[6] * cell_size, y)
    c_v_pos = (distances_1[7] * cell_size, y)
    blit_kv('Cargo', c_k_pos, f"{infos['cargo']:6d}", c_v_pos)

    g_k_pos = (distances_1[8] * cell_size, y)
    g_v_pos = (distances_1[9] * cell_size, y)
    blit_kv('Goal', g_k_pos, f"{infos['goal']:6d}", g_v_pos)

    t_k_pos = (distances_1[10] * cell_size, y)
    t_v_pos = (distances_1[11] * cell_size, y)
    blit_kv('PBR', t_k_pos, f"{infos['pbr']:6.2f}%", t_v_pos)

    static_info = f"Robot: {infos['robot']}  Picker: {infos['picker']}  " \
                  f"Shelf: {infos['shelf']}  " \
                  f"Unload: {infos['unload_cost']}  shape: {infos['shape']}"
    si_surf = font.render(static_info, True,
                          'black', COLORS['wall'])
    surf.blit(si_surf, (distances_1[-1] * cell_size, y))

    font_rc = pygame.font.SysFont('PingFang', 10)
    rc_surf = Surface((cell_size, 4 * cell_size))
    rc_surf.fill(COLORS['wall'])
    rs = font_rc.render(' Rs', True, 'black', COLORS['wall'])
    rc_surf.blit(rs, (0, 0))
    rs_v = font_rc.render(f"{infos['rows']:3d}",
                          True, 'black', 'white')
    rc_surf.blit(rs_v, (0, cell_size))
    cs = font_rc.render(' Cs', True, 'black', COLORS['wall'])
    rc_surf.blit(cs, (0, 2 * cell_size))
    cs_v = font_rc.render(f"{infos['cols']:3d}", True,
                          'black', 'white')
    rc_surf.blit(cs_v, (0, 3 * cell_size))
    surf.blit(rc_surf,
              (screen_size[0] - cell_size, screen_size[1] - 5 * cell_size))


def render_test():
    keep = True
    while keep:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keep = False
        pygame.display.update()
