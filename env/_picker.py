from colorama import (
    Fore,
    Style,
)
from typing import Tuple


class Picker(object):

    def __init__(self, id_: int, pos: Tuple):
        self.id = id_
        self._pos = pos
        self._cargos = 0

    @property
    def pos(self):
        return self._pos

    @property
    def cargos(self):
        return self._cargos

    def pick(self):
        self._cargos += 1

    def reset(self):
        self._cargos = 0

    def __str__(self):
        return (f"Picker({self.id:02d})-({self.pos[0]:02d}, {self.pos[1]:02d})"
                f" received cargos :" +
                Style.BRIGHT + Fore.RED + f"{self._cargos}" + Style.RESET_ALL)
