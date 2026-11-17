"""Evalúa la Q-Table congelada (ε = 0): prueba (200 ep.), línea base aleatoria y validación cruzada (5 bloques)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from flappy_qlearning import CONFIG, TEST_SEED0, CROSS_SEED0, QLearningAgent, RandomAgent, StateDiscretizer, evaluate

path = sys.argv[1] if len(sys.argv) > 1 else "models/qtable_flappy.npy"
agent = QLearningAgent(CONFIG); agent.Q = np.load(path); disc = StateDiscretizer(CONFIG)
seeds = range(TEST_SEED0, TEST_SEED0 + CONFIG["n_test"])
res = {"Q-Learning": evaluate(agent, disc, seeds, CONFIG["max_steps_test"]),
       "Aleatorio": evaluate(RandomAgent(CONFIG["seed"]), disc, seeds, CONFIG["max_steps_test"])}
print(pd.DataFrame({k: dict(media=v.score.mean(), mediana=v.score.median(), maximo=v.score.max(),
                            desv_std=v.score.std(), supervivencia=1 - v.choque.mean()) for k, v in res.items()}).T.round(2))
bl = [evaluate(agent, disc, range(CROSS_SEED0 + b * 100, CROSS_SEED0 + b * 100 + 50), CONFIG["max_steps_test"]).score.mean() for b in range(5)]
print("Validación cruzada (5 bloques):", [round(x, 1) for x in bl], f"| CV = {np.std(bl, ddof=1) / np.mean(bl):.1%}")
