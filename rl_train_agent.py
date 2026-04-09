from stable_baselines3 import PPO
from hydrogen_env import HydrogenEnv


def train_model(scenario):

    env = HydrogenEnv(
        price_file= r"C:\Users\moradpi1\Desktop\innovation postdoc\green-hydrogen-rl-optimizer\Prices.csv",
        scenario=scenario
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64
    )

    model.learn(total_timesteps=120_000)

    model.save(f"models/ppo_{scenario}")

    print(f"Training finished for {scenario}")
