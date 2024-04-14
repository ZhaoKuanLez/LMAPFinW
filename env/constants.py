# TEST_MODE = True
TEST_MODE = False
# --------------------------------------------
BLOCK_WIDTH = 2
WALL_WIDTH = 1
PICKER_WIDTH = 1

FREE = 0
OBSTACLE = 1
SHELF_SPOT = 2
PICK_SPOT = 3
ROBOT_SPOT = 4
WAIT_SPOT = 5

REWARDS = {
    'normal': -0.1,
    'errors': -2,
    'unloading': 0,
    'goal_done': 10,
    'task_done': 30
}

WINDOW_LIMIT = (1000, 1000)
ACTIONS = {
    'stop': 0,
    'left': 1,
    'up': 2,
    'right': 3,
    'down': 4
}

# 渲染
COLORS = {
    'grid': (255, 255, 255),
    'wall': (217, 217, 217),
    'wait': (245, 245, 245),
    'bond': (0, 0, 0),
    'tick': (0, 0, 0),
    'shelf': (255, 255, 255),
    'picker': (0, 0, 0),
    'robot_spot': (242, 242, 242),
    'shelf_spot': (217, 217, 217),
    'connect': (0, 255, 0),
    'graph': (255, 0, 0),
    'dy_info': (0, 0, 255),
}

# 机器人
MOVE = {
    'normal': 0,
    'goal_done': 1,
    'task_done': 2,
    'err_env': -1,
    'err_line': -2,
    'err_point': -3,
}

WORK = {
    'detach': 0,
    'attach': 1
}

PERIOD = {
    'idle': True,
    'to_picker': False,
    'unloading': False,
    'to_home': False
}

STAGE = {
    13: {
        'idle': True, 'to_picker': False,
        'unloading': False, 'to_home': False
    },
    22: {
        'idle': True, 'to_picker': True,
        'unloading': False, 'to_home': False
    },
    31: {
        'idle': True, 'to_picker': True,
        'unloading': True, 'to_home': False
    },
    40: {
        'idle': True, 'to_picker': True,
        'unloading': True, 'to_home': True
    }
}

VECTORS = [
    (0, 0),
    (0, -1),
    (-1, 0),
    (0, 1),
    (1, 0)
]
