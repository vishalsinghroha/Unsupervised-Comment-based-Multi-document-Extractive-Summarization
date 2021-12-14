"""Module with main parts of NSGA-II algorithm.
It contains individual definition"""


class Individual(object):
    """Represents one individual"""

    def __init__(self):
        self.rank = None
        self.crowding_distance = None
        self.dominated_solutions = set()
        self.features = None
        self.objectives = None
        self.dominates = None
        self.K = None           #No. of ones in a solution vector
        self.label = None
        #self.cluster_center = None
        #self.cluster_center_pos=None

    def set_objectives(self, objectives):
        self.objectives = objectives
