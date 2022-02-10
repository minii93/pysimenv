from typing import Union
from pysimenv.core.base import BaseObject


class PurePNG2dim(BaseObject):
    def __init__(self, N: float = 3.0, interval: Union[int, float] = -1):
        super(PurePNG2dim, self).__init__(interval=interval)
        self.N = N

    # implement
    def evaluate(self, V_M, omega):
        """
        :param V_M: speed of the missile
        :param omega: LOS rate
        :return: acceleration command a_M
        """
        a_M = self.N*V_M*omega
        return a_M
