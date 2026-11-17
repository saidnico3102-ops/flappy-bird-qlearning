"""Genera docs/flappy_agente.gif con el agente entrenado jugando (política greedy)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle
from flappy_qlearning import CONFIG, TEST_SEED0, FlappyBirdEnv, QLearningAgent, StateDiscretizer

agent = QLearningAgent(CONFIG); agent.Q = np.load("models/qtable_flappy.npy"); disc = StateDiscretizer(CONFIG)

def draw(ax, s):
    E = FlappyBirdEnv; ax.clear(); ax.set_xlim(0, E.W); ax.set_ylim(E.H, 0); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), E.W, E.GROUND, color="#70c5ce")); ax.add_patch(Rectangle((0, E.GROUND), E.W, E.H - E.GROUND, color="#ded895"))
    for x, cy, _ in s["pipes"]:
        ax.add_patch(Rectangle((x, 0), E.PIPE_W, cy - E.GAP / 2, color="#5aa02c", ec="k"))
        ax.add_patch(Rectangle((x, cy + E.GAP / 2), E.PIPE_W, E.GROUND - cy - E.GAP / 2, color="#5aa02c", ec="k"))
    ax.add_patch(Circle((E.BIRD_X, s["y"]), E.BIRD_R, color="#f7d51d", ec="k", zorder=5))
    ax.text(E.W / 2, 50, str(s["score"]), ha="center", fontsize=22, weight="bold", color="white")

env = FlappyBirdEnv(max_steps=900); s = disc.discretize(env.reset(TEST_SEED0)); frames = [env.snapshot()]
while True:
    obs, r, done, info = env.step(agent.select_action(s, greedy=True)); s = disc.discretize(obs); frames.append(env.snapshot())
    if done: break
fig, ax = plt.subplots(figsize=(2.9, 5.1))
anim = FuncAnimation(fig, lambda k: draw(ax, frames[k * 3]), frames=len(frames) // 3, interval=40)
anim.save("docs/flappy_agente.gif", writer=PillowWriter(fps=25)); print("GIF:", len(frames), "pasos, puntaje", info["score"])
