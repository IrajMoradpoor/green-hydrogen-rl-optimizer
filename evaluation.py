import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from hydrogen_env import HydrogenEnv


def evaluate(scenario):

    env = HydrogenEnv(
        r"C:\Users\moradpi1\Desktop\innovation postdoc\green-hydrogen-rl-optimizer\Prices.csv",
        scenario=scenario
    )

    model = PPO.load(f"models/ppo_{scenario}")

    obs, _ = env.reset()

    prices = []
    storage = []
    power = []

    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        prices.append(info["price"])
        storage.append(info["storage"])
        power.append(info["power"])

    plt.figure(figsize=(10,6))
    plt.plot(prices, label="Price")
    plt.plot(storage, label="Storage")
    plt.plot(power, label="Electrolyzer Power")
    plt.legend()
    plt.title(f"RL Operation — {scenario.upper()} Scenario")
    plt.savefig(f"results/{scenario}_operation.png")
    plt.show()