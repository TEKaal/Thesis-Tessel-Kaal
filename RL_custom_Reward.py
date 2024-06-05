import numpy as np

from pymgrid.microgrid.reward_shaping.base import BaseRewardShaper

class GridUsagePenaltyShaper(BaseRewardShaper):
    """
    Reward is inversely proportional to the percentage of load that is met by the main grid.

    Return a value in [-1, 1]. Value of 1 implies no grid energy was used, while -1 implies all load was met by the grid.

    Use in a config with:
    microgrid:
        attributes:
            reward_shaping_func: !GridUsagePenaltyShaper {}
    """
    yaml_tag = u"!GridUsagePenaltyShaper"
    def __call__(self, step_info):
        # grid_usage = self.sum_module_val(step_info, 'grid', 'provided_energy')
        # met_load = self.sum_module_val(step_info, 'load', 'absorbed_energy')
        #
        # print("grid usage", grid_usage)
        # print(met_load)
        # try:
        #     percent_grid_usage = grid_usage / met_load
        # except ZeroDivisionError:
        #     return 1.0  # Max reward if no load
        #
        # reward = -2 * percent_grid_usage + 1
        # reward = np.clip(reward, -1, 1)  # Ensure the reward is within the intended range
        #
        # assert -1 <= reward <= 1 or np.isclose(reward, 1) or np.isclose(reward, -1)
        # return reward
        grid_usage = self.sum_module_val(step_info, 'grid', 'provided_energy')
        if grid_usage > 0:  # Check if there is any grid usage
            return 50000000  # Halve the reward if the grid was used
        return 0  # Otherwise, return the full reward
