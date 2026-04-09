import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class HydrogenEnv(gym.Env):

    def __init__(self, price_file, scenario="grid"):
        super().__init__()

        self.prices = pd.read_csv(price_file)["price"].values

        # remove NaN / inf
        self.prices = np.nan_to_num(self.prices, nan=0.0, posinf=300, neginf=-50)

        # clip extreme spikes (VERY IMPORTANT)
        self.prices = np.clip(self.prices, -50, 300)

        # normalize
        self.prices = (self.prices - self.prices.mean()) / self.prices.std()

        self.T = len(self.prices)

        self.scenario = scenario
        self.ppa_price = 45.0  # €/MWh fixed contract

        # Plant parameters
        self.max_power = 10.0        # MW
        self.efficiency = 0.7        # hydrogen per MW
        self.storage_max = 500.0
        self.demand = 5.0            # hourly demand

        # observation = [storage, price]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([self.storage_max, 500.0]),
            dtype=np.float32
        )

        # action = electrolyzer load fraction
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.reset()

    # --------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.storage = 200.0
        return self._get_state(), {}

    # --------------------------------------------------

    def _get_price(self):
        if self.scenario == "ppa":
            return self.ppa_price
        return self.prices[self.t]

    def _get_state(self):
        return np.array([self.storage, self._get_price()],
                        dtype=np.float32)

    # --------------------------------------------------

    def step(self, action):

        load = float(action[0])
        power = load * self.max_power
        price = self._get_price()

        hydrogen_produced = power * self.efficiency

        # storage dynamics
        self.storage += hydrogen_produced - self.demand
        self.storage = np.clip(self.storage, 0, self.storage_max)

        electricity_cost = power * price

        penalty = 0
        if self.storage <= 1:
            penalty = 200

        reward = -(electricity_cost + penalty)

        self.t += 1
        done = self.t >= self.T - 1

        info = {
            "cost": electricity_cost,
            "storage": self.storage,
            "price": price,
            "power": power
        }

        return self._get_state(), reward, done, False, info
