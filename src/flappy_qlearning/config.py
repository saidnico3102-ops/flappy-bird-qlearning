"""Hiperparámetros y semillas del experimento (Tabla 6 del informe)."""
SEED = 42
CONFIG = dict(
    alpha=0.1, gamma=0.95,                      # tasa de aprendizaje y factor de descuento
    eps0=1.0, eps_min=0.01, decay=0.995,        # política ε-greedy con decaimiento por episodio
    bins=(15, 20, 12),                          # discretización (dx, dy, vel)
    q_init=2.0,                                 # inicialización optimista de la Q-Table
    dy_range=128.0,                             # px que cubre el rango [-1, 1] de dy
    n_train=10_000, n_test=200,                 # episodios de entrenamiento / prueba
    eval_every=500, eval_episodes=50,           # validación periódica (greedy)
    max_steps_train=5_000, max_steps_val=5_000, max_steps_test=10_000,
    seed=SEED,
)
VAL_SEED0, TEST_SEED0, CROSS_SEED0 = 20_000, 10_000, 50_000   # bloques de semillas independientes
