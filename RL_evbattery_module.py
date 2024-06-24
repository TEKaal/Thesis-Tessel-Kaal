import numpy as np
import yaml

from pymgrid.microgrid import DEFAULT_HORIZON
from pymgrid.modules.base import BaseTimeSeriesMicrogridModule
from pymgrid.modules import BatteryModule


class EVBatteryModule(BaseTimeSeriesMicrogridModule, BatteryModule):
    """
    A battery module with time series data and availability per timestep.

    Parameters
    ----------
    time_series : array-like, shape (n_steps, )
        Time series of battery availability or other relevant data.

    min_capacity : float
        Minimum energy that must be contained in the battery.

    max_capacity : float
        Maximum energy that can be contained in the battery.
        If ``soc=1``, capacity is at this maximum.

    max_charge : float
        Maximum amount the battery can be charged in one step.

    max_discharge : float
        Maximum amount the battery can be discharged in one step.

    efficiency : float
        Efficiency of the battery.

    battery_cost_cycle : float, default 0.0
        Marginal cost of charging and discharging.

    battery_transition_model : callable or None, default None
        Function to model the battery's transition.

    init_charge : float or None, default None
        Initial charge of the battery.

    init_soc : float or None, default None
        Initial state of charge of the battery.

    forecaster : callable, float, "oracle", or None, default None.
        Function that gives a forecast n-steps ahead.

    forecast_horizon : int.
        Number of steps in the future to forecast. If forecaster is None, ignored and 0 is returned.

    forecaster_increase_uncertainty : bool, default False
        Whether to increase uncertainty for farther-out dates if using a GaussianNoiseForecaster. Ignored otherwise..

    raise_errors : bool, default False
        Whether to raise errors if bounds are exceeded in an action.
        If False, actions are clipped to the limit possible.

    """

    module_type = ('battery', 'controllable')
    yaml_tag = u"!EVBatteryModule"
    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader

    state_components = np.array(["battery_availability"], dtype=object)

    def __init__(self,
                 time_series,
                 min_capacity,
                 max_capacity,
                 max_charge,
                 max_discharge,
                 efficiency,
                 battery_cost_cycle=0.0,
                 battery_transition_model=None,
                 init_charge=None,
                 init_soc=None,
                 forecaster=None,
                 forecast_horizon=DEFAULT_HORIZON,
                 forecaster_increase_uncertainty=False,
                 forecaster_relative_noise=False,
                 initial_step=0,
                 final_step=-1,
                 raise_errors=False):

        # Initialize BaseTimeSeriesMicrogridModule
        BaseTimeSeriesMicrogridModule.__init__(self,
                                               time_series=time_series,
                                               raise_errors=raise_errors,
                                               forecaster=forecaster,
                                               forecast_horizon=forecast_horizon,
                                               forecaster_increase_uncertainty=forecaster_increase_uncertainty,
                                               forecaster_relative_noise=forecaster_relative_noise,
                                               initial_step=initial_step,
                                               final_step=final_step)

        # Initialize BatteryModule with only its relevant parameters
        BatteryModule.__init__(self,
                               min_capacity=min_capacity,
                               max_capacity=max_capacity,
                               max_charge=max_charge,
                               max_discharge=max_discharge,
                               efficiency=efficiency,
                               battery_cost_cycle=battery_cost_cycle,
                               battery_transition_model=battery_transition_model,
                               init_charge=init_charge,
                               init_soc=init_soc,
                               initial_step=initial_step,
                               raise_errors=raise_errors)

        self.name = ('battery', None)

    def update(self, external_energy_change, as_source=False, as_sink=False):
        if not self.is_available():
            return 0.0, self._done(), {'availability': 0}

        reward, done, info = BatteryModule.update(self, external_energy_change, as_source, as_sink)
        self._update_step()
        return reward, done, info

    def is_available(self):
        """
        Check if the battery is available at the current timestep.

        Returns
        -------
        available : bool
            True if the battery is available, False otherwise.
        """
        return self._time_series[self._current_step] > 0

    @property
    def max_consumption(self):
        return self.current_load if self.is_available() else 0.0

    @property
    def max_production(self):
        return super().max_production if self.is_available() else 0.0

    def sample_action(self, strict_bound=False):
        return np.array([])

    @property
    def is_sink(self):
        return True

    @property
    def is_source(self):
        return True

    @property
    def current_obs(self):
        return np.array([self._time_series[self._current_step]])
