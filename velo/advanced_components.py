"""
VELO v11+ Advanced Components
"""
import random
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.mixture import GaussianMixture


class FormSequenceNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.head_win = nn.Linear(hidden_dim, 1)
        self.head_place = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        return self.head_win(h_last).squeeze(-1), self.head_place(h_last).squeeze(-1)


class FormSequenceModel:
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, lr: float = 1e-3, n_epochs: int = 15, batch_size: int = 64, device: str = "cpu"):
        if not HAS_TORCH:
            raise ImportError("torch required")
        self.device = torch.device(device)
        self.net = FormSequenceNet(input_dim, hidden_dim, num_layers).to(self.device)
        self.lr, self.n_epochs, self.batch_size = lr, n_epochs, batch_size
    def fit(self, X_seq, y_win, y_place):
        self.net.train()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()
        X = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
        yw = torch.tensor(y_win, dtype=torch.float32, device=self.device)
        yp = torch.tensor(y_place, dtype=torch.float32, device=self.device)
        n, idx = X.shape[0], np.arange(X.shape[0])
        for epoch in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                opt.zero_grad()
                win_logit, place_logit = self.net(X[batch_idx])
                loss = bce(win_logit, yw[batch_idx]) + bce(place_logit, yp[batch_idx])
                loss.backward()
                opt.step()
        return self
    @torch.no_grad()
    def predict_proba(self, X_seq):
        self.net.eval()
        X = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
        win_logit, place_logit = self.net(X)
        return torch.sigmoid(win_logit).cpu().numpy(), torch.sigmoid(place_logit).cpu().numpy()


class IntentModel:
    def __init__(self, n_states: int = 3, random_state: int = 1337):
        self.n_states = n_states
        self.gmm = GaussianMixture(n_components=n_states, covariance_type="full", random_state=random_state)
        self.state_order_ = None
    def fit(self, X_intent):
        self.gmm.fit(X_intent)
        self.state_order_ = np.argsort(self.gmm.means_[:, 0]).tolist()
        return self
    def intent_scores(self, X_intent):
        if self.state_order_ is None:
            raise RuntimeError("Not fitted")
        resp = self.gmm.predict_proba(X_intent)[:, self.state_order_]
        if self.n_states == 3:
            p_off, p_neutral, p_on = resp[:, 0], resp[:, 1], resp[:, 2]
        else:
            p_on, p_off = resp[:, -1], resp[:, 0]
            p_neutral = np.clip(1.0 - (p_on + p_off), 0, 1)
        return {"p_off": p_off, "p_neutral": p_neutral, "p_on": p_on, "scalar_intent": p_on - p_off}


class GAStakingOptimizer:
    def __init__(self, population_size=40, n_generations=60, elite_frac=0.2, mutation_prob=0.2, mutation_scale=0.25, random_state=1337):
        self.population_size, self.n_generations = population_size, n_generations
        self.elite_frac, self.mutation_prob, self.mutation_scale = elite_frac, mutation_prob, mutation_scale
        self.rs = np.random.RandomState(random_state)
        self.best_theta_, self.best_fitness_ = None, -np.inf
    def _random_theta(self):
        return {"k_core": 0.5 + self.rs.rand(), "k_value": 0.3 + self.rs.rand() * 0.7, "k_chaos": 0.1 + self.rs.rand() * 0.4, "alpha_edge": 0.5 + self.rs.rand(), "max_frac_bank": 0.08 + self.rs.rand() * 0.12}
    def _mutate(self, theta):
        new = theta.copy()
        for k in new:
            if self.rs.rand() < self.mutation_prob:
                new[k] += self.rs.randn() * self.mutation_scale * abs(new[k])
                new[k] = max(0, new[k])
        return new
    def fit(self, historical_decisions, fitness_func):
        population = [self._random_theta() for _ in range(self.population_size)]
        for gen in range(self.n_generations):
            scores = [fitness_func(theta, historical_decisions) for theta in population]
            sorted_idx = np.argsort(scores)[::-1]
            elite_count = max(1, int(self.elite_frac * self.population_size))
            elite = [population[i] for i in sorted_idx[:elite_count]]
            if scores[sorted_idx[0]] > self.best_fitness_:
                self.best_fitness_, self.best_theta_ = scores[sorted_idx[0]], population[sorted_idx[0]].copy()
            new_pop = elite.copy()
            while len(new_pop) < self.population_size:
                new_pop.append(self._mutate(elite[self.rs.randint(len(elite))]))
            population = new_pop
        return self


class RLStakingAgent:
    def __init__(self, buckets, n_odds_bins=10, alpha=0.1, gamma=0.95, epsilon=0.1, random_state=1337):
        self.buckets, self.n_odds_bins = buckets, n_odds_bins
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rs = np.random.RandomState(random_state)
        self.q_table = {(b, o): 0.0 for b in buckets for o in range(n_odds_bins)}
    def _discretize_odds(self, odds):
        if odds < 1.5: return 0
        elif odds < 3.0: return 1
        elif odds < 5.0: return 2
        elif odds < 10.0: return 3
        else: return min(self.n_odds_bins - 1, 4 + int((odds - 10) / 10))
    def stake_for_runner(self, bucket, odds, bank, train=False):
        state = (bucket, self._discretize_odds(odds))
        if state not in self.q_table:
            self.q_table[state] = 0.0
        q_value = self.q_table[state]
        stake_frac = max(0, min(0.10, q_value))
        return stake_frac * bank, state
    def update(self, bucket, odds, stake_frac, reward):
        state = (bucket, self._discretize_odds(odds))
        if state not in self.q_table:
            self.q_table[state] = 0.0
        self.q_table[state] += self.alpha * (reward - self.q_table[state])
