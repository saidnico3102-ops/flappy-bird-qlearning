"""Entrena el agente (10 000 episodios, ~1-3 min en CPU) y guarda la Q-Table en models/."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from flappy_qlearning import CONFIG, train

agent, log, val_log, final_Q = train(CONFIG, CONFIG["n_train"], CONFIG["eval_every"], CONFIG["eval_episodes"])
best = val_log.loc[val_log["media"].idxmax()]
print(f"Mejor checkpoint: episodio {int(best.episodio)} | validación = {best['media']:.1f}")
os.makedirs("models", exist_ok=True)
agent.save("models/qtable_flappy.pkl"); np.save("models/qtable_flappy.npy", agent.Q)
log.to_csv("models/log_entrenamiento.csv", index=False); val_log.to_csv("models/log_validacion.csv", index=False)
print("Guardado en models/")
