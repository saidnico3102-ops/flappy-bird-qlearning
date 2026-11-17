"""Verifica que models/qtable_flappy.npy reproduce los resultados del informe (Tabla 9)."""
import os, numpy as np
from flappy_qlearning import CONFIG, QLearningAgent, StateDiscretizer, evaluate

RUTA = os.path.join(os.path.dirname(__file__), "..", "models", "qtable_flappy.npy")

def test_forma_de_la_qtable():
    q = np.load(RUTA); assert q.shape == (15, 20, 12, 2) and q.dtype == np.float64

def test_desempeno_semillas_de_prueba():
    ag = QLearningAgent(CONFIG); ag.Q = np.load(RUTA)
    r = evaluate(ag, StateDiscretizer(CONFIG), range(10_000, 10_020), CONFIG["max_steps_test"])
    assert r.score.mean() > 30 and r.score.max() >= 100      # muy por encima del agente aleatorio (0)
