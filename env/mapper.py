import numpy as np
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
from warnings import warn

from env.constants import (
    BLOCK_WIDTH,
    OBSTACLE,
    PICKER_WIDTH,
    PICK_SPOT,
    ROBOT_SPOT,
    SHELF_SPOT,
    WALL_WIDTH,
    WAIT_SPOT,
)

_NDArray = np.ndarray


class Mapper(object):
    """地图生成器

    可指定的初始化参数：
        rc- 货架组列数（必须是偶数且大于等于2）；
        bb - 货架组间距；
        bp - 货架组与拣货台的间距；
        bw - 货架组与墙体的间距；
        pw - 拣货台与墙体的间距

    """
    robot_cols: int = 2  # rc
    block_block: int = 2  # bb
    block_picker: int = 5  # bp
    block_wall: int = 2  # bw
    picker_wall: int = 0  # pw
    _block_width: int = 2

    rows: Optional[int] = None
    cols: Optional[int] = None
    len_: Optional[int] = None

    pickers_spot_num: Optional[int] = None
    pickers_spot_pos: Optional[Dict[int, tuple]] = None

    shelves_spot_num: Optional[int] = None
    shelves_spot_pos: Optional[Dict[int, tuple]] = None

    robots_spot_num: Optional[int] = None
    robots_spot_pos: Optional[Dict[int, tuple]] = None

    base_map: Optional[_NDArray] = None
    shape: Optional[tuple] = None

    corridors_map: Optional[Dict[tuple, Any]] = None
    corridors: Optional[dict] = None
    intersections: Optional[Dict[str, tuple]] = None

    def __init__(self, **kwargs):
        def _data_check(data_name: str, data, lower_lim):
            if data < lower_lim:
                raise ValueError(f"{data_name}(in={data}) must gt {lower_lim}!")

        if 'rc' in kwargs:
            self.robot_cols = kwargs.get('rc')
        if self.robot_cols % 2 != 0:
            raise ValueError(f"robot_cols(={self.robot_cols}) must be even!")
        _data_check('robot_cols', self.robot_cols, 2)
        if 'bb' in kwargs:
            self.block_block = kwargs.get('bb')
        _data_check('block_block', self.block_block, 1)
        if 'bp' in kwargs:
            self.block_picker = kwargs.get('bp')
        _data_check('block_picker', self.block_picker, 1)
        if 'bw' in kwargs:
            self.block_wall = kwargs.get('bw')
        _data_check('block_wall', self.block_wall, 1)
        if 'pw' in kwargs:
            self.picker_wall = kwargs.get('pw')
        _data_check('picker_wall', self.picker_wall, 0)

    def generate(self, rows: int = 3, cols: int = 10, len_: int = 5,
                 pickers: int = 10) -> Tuple[_NDArray, dict, dict, dict]:
        """Generate the base map

        Args:
            rows: the shelf rows
            cols: the shelf colum
            len_: the shelf length
            pickers: the num of pick station

        Returns:
            base_map, corridors_map, intersections, corridors
        """
        # check parameters
        if rows < 1 or cols < 1 or len_ < 1 or pickers < 1:
            raise ValueError(f"parameters error: r-{rows}，c-{cols}，len={len_}，"
                             f"picker={pickers}")
        else:
            self.rows, self.cols = rows, cols
            self.len_, self.pickers_spot_num = len_, pickers

        # compute map size
        self.shape = self.get_map_size()

        # generate map
        self.base_map = np.zeros(self.shape).astype(int)
        self._add_walls()
        self._add_shelves_spot()
        self._add_robots_spot()
        self._add_pickers_spot()
        self._add_wait_spots()

        # compute intersections and corridors
        self.corridors_map, self.intersections, self.corridors \
            = self.gather_info()

        return (self.base_map, self.corridors_map,
                self.intersections, self.corridors)

    def gather_info(self) -> Tuple[dict, dict, dict]:
        """Gather intersections and corridors

        Data Form:
            - `intersection`: List[position: tuple]
            - `corridor`: Dict['endpoints': dict, 'inside': list, 'crossroads': list, 'vector': tuple, 'trend': tuple],
              here, endpoints: "endpoint intersection", inside: "inside points", crossroads: 'crossroad points'.
            - `corridors_map`: Dict[pos: tuple, info: list], the length of info is 2, the info[1] is corridor id
              (the pos is inside the corridor, otherwise equal -1), and the info[2] is status, there are 6 case:
              first, inside a corridor, value is 1, endpoint, value is 2, free area, value is 0, obstacle,
              value is -1, inner crossroad point, value is 3, outside crossroad point, value is 4.

        Returns:
            corridors_map, intersections, corridors
        """
        _showing_on_map = False  # WARN: If finished test, ensure its value is False.
        _CR_OUT, _CR_IN, _EP, _IN, _FREE, _OBS = 4, 3, 2, 1, 0, -1

        def put(coords, flag, showing=True):
            for coord in coords:
                if showing:
                    self.base_map[coord] = flag

        # Initializer corridor map
        corridors_map: Dict[tuple, List[int, int]] = {}
        for x in range(self.base_map.shape[0]):
            for y in range(self.base_map.shape[1]):
                position = tuple((x, y))
                # FREE = 0 OBSTACLE = 1 SHELF_SPOT = 2 PICK_SPOT = 3 ROBOT_SPOT = 4 WAIT_SPOT = 5
                if 0 < self.base_map[position] < 3:
                    status = [-1, _OBS]
                else:
                    status = [-1, _FREE]
                corridors_map[position] = status

        def write_to_map(pos_s: List[tuple], corridor_index: int = -1, flag: int = -1):
            """Write position info to corridor map"""
            for pos in pos_s:
                corridors_map[pos] = [corridor_index, flag]

        # Compute All Intersections' position
        intersections: Dict[str, List[tuple]] = {
            'all': [], 'inside': [], 'outside': []
        }
        # First, we will get left-up point coordination
        anchor_u = (WALL_WIDTH + self.block_wall - 1)
        anchor_l = (WALL_WIDTH + int(self.robot_cols / 2)
                    + self.block_wall - 1)
        xs = [anchor_u + i * (self.len_ + 1) for i in range(self.rows + 1)]
        ys = [anchor_l + i * (self._block_width + 1)
              for i in range(self.cols + 1)]
        out_x_u, out_x_d, out_y_l, out_y_r = xs[0], xs[-1], ys[0], ys[-1]
        for x in xs:
            coordinates = [(x, y) for y in ys]
            intersections['all'].extend(coordinates)
        for position in intersections['all']:
            x, y = position
            if x == out_x_u or x == out_x_d or y == out_y_l or y == out_y_r:
                intersections['outside'].append(tuple(position))
            else:
                intersections['inside'].append(tuple(position))
        write_to_map(intersections['outside'], flag=_CR_OUT)
        write_to_map(intersections['inside'], flag=_CR_IN)

        # Second, we will compute all corridors and update corridors map.
        corridors: Dict[int, Any] = {}
        corridor_id: int = 1
        # 2-1: Vertical corridors
        corridors_anchor = []
        anchor_u = (WALL_WIDTH + self._block_width)
        anchor_l = (WALL_WIDTH + int(self.robot_cols / 2) + self.len_ - 1)
        xs = [anchor_u + i * (self.len_ + 1) for i in range(self.rows)]
        ys = [anchor_l + i * (self._block_width + 1)
              for i in range(self.cols - 1)]
        for x in xs:
            coordinates = [(x, y) for y in ys]
            corridors_anchor.extend(coordinates)
        for anchor in corridors_anchor:
            x, y = anchor
            endpoints: List[tuple] = [tuple(anchor),
                                      (x + self.len_ - 1, y)]  # ep-1
            write_to_map(endpoints, corridor_id, flag=_EP)
            put(endpoints, 10, _showing_on_map)

            crossroads: List[tuple] = [(x - 1, y), (x + self.len_, y)]
            put(crossroads, 15, _showing_on_map)

            inside_points: List[tuple] = []
            for i in range(1, self.len_ - 1):
                x_ = x + i
                inside_points.append(tuple((x_, y)))
            write_to_map(inside_points, corridor_id, flag=_IN)
            put(inside_points, 20, _showing_on_map)

            corridor_info = {'endpoints': endpoints, 'inside': inside_points,
                             'all': endpoints + inside_points,
                             'crossroads': crossroads, 'vector': (0, 0),
                             'trend': (1, 0)}
            corridors[corridor_id] = corridor_info
            corridor_id += 1
        # 2-2: Horizontal corridors
        corridors_anchor = []
        anchor_u = (WALL_WIDTH + self._block_width + self.len_)
        anchor_l = (WALL_WIDTH + int(self.robot_cols / 2) + self.block_wall)
        xs = [anchor_u + i * (self.len_ + 1) for i in range(self.rows - 1)]
        ys = [anchor_l + i * (self._block_width + 1) for i in range(self.cols)]
        for x in xs:
            coordinates = [(x, y) for y in ys]
            corridors_anchor.extend(coordinates)
        for anchor in corridors_anchor:
            x, y = anchor
            endpoints: List[tuple] = [tuple(anchor),
                                      (x, y + self._block_width - 1)]  # ep-1
            write_to_map(endpoints, corridor_id, flag=_EP)
            put(endpoints, 10, _showing_on_map)

            crossroads: List[tuple] = [(x, y - 1), (x, y + self._block_width)]
            put(crossroads, 15, _showing_on_map)
            inside_points: List[tuple] = []

            corridor_info = {'endpoints': endpoints, 'inside': inside_points,
                             'all': endpoints + inside_points,
                             'crossroads': crossroads, 'vector': (0, 0),
                             'trend': (0, 1)}
            corridors[corridor_id] = corridor_info
            corridor_id += 1

        return corridors_map, intersections, corridors

    def get_map_size(self) -> Tuple[int, int]:
        w = (2 * WALL_WIDTH + self.robot_cols + 2 * self.block_wall
             + (self.cols - 1) * (BLOCK_WIDTH + self.block_block) + BLOCK_WIDTH)
        h = (2 * WALL_WIDTH + self.block_wall
             + (self.rows - 1) * (self.len_ + self.block_block)
             + self.len_ + self.block_picker + PICKER_WIDTH)
        return h, w

    def _add_walls(self) -> None:
        self.base_map[0, :] = self.base_map[-1, :] = OBSTACLE
        self.base_map[:, 0] = self.base_map[:, -1] = OBSTACLE

    def _add_shelves_spot(self) -> None:
        """标记货架位"""
        anchor_u = WALL_WIDTH + self.block_wall
        anchor_l = WALL_WIDTH + int(self.robot_cols / 2) + self.block_wall

        block_heads_x, block_heads_y = [], []
        self.shelves_spot_num = 0
        self.shelves_spot_pos = {}

        for _ in range(self.rows):
            block_heads_x.extend([anchor_u + r for r in range(self.len_)])
            anchor_u += (self.len_ + self.block_block)
        for _ in range(self.cols):
            block_heads_y.extend([anchor_l + c for c in range(BLOCK_WIDTH)])
            anchor_l += (BLOCK_WIDTH + self.block_block)

        shelves_spots_pos = []
        for x in block_heads_x:
            for y in block_heads_y:
                shelves_spots_pos.append((x, y))

        self.shelves_spot_num = len(shelves_spots_pos)
        for id_, shelf_spot_pos in enumerate(shelves_spots_pos, start=1):
            self.base_map[shelf_spot_pos] = SHELF_SPOT
            self.shelves_spot_pos.update({id_: shelf_spot_pos})

    def _add_robots_spot(self) -> None:
        anchor_u = WALL_WIDTH
        anchor1_l = WALL_WIDTH
        half_spots_col = int(self.robot_cols / 2)
        anchor2_l = self.shape[1] - WALL_WIDTH - half_spots_col

        # 获取所有位置的 y 坐标
        offset_y = list(range(half_spots_col))
        y_l = [anchor1_l + offset for offset in offset_y]
        y_r = [anchor2_l + offset for offset in offset_y]
        ys = y_l + y_r

        # 获取所有位置的 x 坐标
        offset_x = list(range(
            self.shape[0] - WALL_WIDTH - PICKER_WIDTH - self.block_picker + 1
        ))
        xs = [anchor_u + offset for offset in offset_x]

        # 在地图中标记
        self.robots_spot_num = 0
        self.robots_spot_pos = {}

        idx = 1
        for x in xs:
            for y in ys:
                self.base_map[tuple((x, y))] = ROBOT_SPOT
                self.robots_spot_pos.update({idx: tuple((x, y))})
                idx += 1
                self.robots_spot_num += 1

    def _add_pickers_spot(self) -> None:
        def get_pickers_coord(x: int, width: int, num: int):
            """计算拣货台坐标

            每一个拣货台的左侧和右侧必须留出一个空位，所以每个拣货台占据的位置为2。
            需要确保下边（去除墙体）长度大于等于 3*(N+2)个单元格。

            Returns:
                拣货台坐标（从左到右）
            """
            real_width = width - 2 * WALL_WIDTH
            if 3 * num > real_width:
                raise ValueError(f"拣货台（{self.pickers_spot_num}）太多!")
            elif num / real_width > 0.8:
                raise ValueError(f"拣货台（{self.pickers_spot_num}）太多（>0.8）!")
            elif num < 0:
                raise ValueError(f"拣货台（{self.pickers_spot_num}）不能少于0！")
            elif num == 0:
                return None
            else:
                gap = int(real_width / num)
                if (real_width - (num - 1) * gap - 1) >= 0 and gap >= 2:
                    head = int((real_width - (num - 1) * gap - 1) / 2 + 1)
                    y = list(range(head, real_width + 1, gap))[0: num]
                else:
                    head = int((real_width - num) / 2 + 1)
                    y = list(range(head, width))[0: num]
                return list(zip([x] * len(y), y))

        anchor_u = self.shape[0] - WALL_WIDTH - PICKER_WIDTH
        pickers_pos = get_pickers_coord(anchor_u, self.shape[1],
                                        self.pickers_spot_num)
        self.pickers_spot_pos: Dict[int, tuple] = {
            id_: pos for id_, pos in enumerate(pickers_pos, start=1)
        }
        if pickers_pos is not None:
            for picker in pickers_pos:
                self.base_map[picker] = PICK_SPOT
        else:
            warn(f"Failed to add picker spot!")

    def _add_wait_spots(self) -> None:
        self.wait_spots: Dict[int, List[tuple]] = {
            picker_id: [] for picker_id in self.pickers_spot_pos.keys()
        }  # 不包括拣货台本身位置

        for x_ in range(1, 3):
            for picker_id, picker_pos in self.pickers_spot_pos.items():
                x = picker_pos[0] - x_
                y = picker_pos[1]
                self.base_map[x, y] = WAIT_SPOT
                self.wait_spots[picker_id].append(tuple((x, y)))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pprint import pprint as pr

    mapper = Mapper(rc=2, bb=1, bp=3, bw=2, pw=1)
    mapper.generate(rows=3, cols=10, len_=5, pickers=5)
    print(f"Map size: {mapper.shape}")
    print("\nCorridors map:")
    pr(mapper.corridors_map)
    print("\nCorridors:")
    pr(mapper.corridors)
    print("\nIntersections:")
    pr(mapper.intersections)
    # print(f"Shelf num: {mapper.shelves_spot_num}")
    # print(f"R-pos-{mapper.robots_spot_num}: {mapper.robots_spot_pos}")
    # print(f"S-pos-{mapper.shelves_spot_num}: {mapper.shelves_spot_pos}")
    # print(f"P-pos-{mapper.pickers_spot_num}: {mapper.pickers_spot_pos}")
    plt.imshow(mapper.base_map)
    plt.show()
