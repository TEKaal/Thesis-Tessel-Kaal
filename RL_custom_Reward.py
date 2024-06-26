import numpy as np

from pymgrid.microgrid.reward_shaping.base import BaseRewardShaper

class GridUsagePenaltyShaper(BaseRewardShaper):
    yaml_tag = u"!GridUsagePenaltyShaper"
    def __call__(self, step_info):
        grid_usage = self.sum_module_val(step_info, 'grid', 'provided_energy')
        if grid_usage > 0:  # Check if there is any grid usage
            return 50000000  # Halve the reward if the grid was used
        return 0  # Otherwise, return the full reward
