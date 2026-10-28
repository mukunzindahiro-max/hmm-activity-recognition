import json
from pathlib import Path

import numpy as np

LOG_2PI = np.log(2 * np.pi)


def logsumexp(a: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return out if keepdims else np.squeeze(out, axis=axis)


class GaussianHMM:
    def __init__(self, states: list, covariance_type: str = "full", tol: float = 1e-4,
                 max_iter: int = 200, min_covar: float = 1e-3, transition_prior: float = 1.0,
                 verbose: bool = False):
        if covariance_type not in ("full", "diag"):
            raise ValueError("covariance_type must be 'full' or 'diag'")
        self.states = list(states)
        self.n_states = len(states)
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.min_covar = min_covar
        self.transition_prior = transition_prior
        self.verbose = verbose

        self.startprob_ = None
        self.transmat_ = None
        self.means_ = None
        self.covars_ = None
        self.history_ = []
        self.converged_ = False
        self.n_iter_ = 0

    def _state_index(self, labels: np.ndarray) -> np.ndarray:
        lookup = {s: i for i, s in enumerate(self.states)}
        return np.array([lookup[l] for l in labels])

    def _regularise(self, cov: np.ndarray) -> np.ndarray:
        if self.covariance_type == "diag":
            return np.maximum(np.diag(cov), self.min_covar)
        cov = (cov + cov.T) / 2
        return cov + self.min_covar * np.eye(cov.shape[0])

    def _normalise_transitions(self, counts: np.ndarray) -> np.ndarray:
        counts = counts + self.transition_prior
        return counts / counts.sum(axis=1, keepdims=True)

    def fit_supervised(self, sequences: list, label_sequences: list) -> "GaussianHMM":
        X = np.vstack(sequences)
        y = np.concatenate([self._state_index(l) for l in label_sequences])
        n_features = X.shape[1]

        start_counts = np.zeros(self.n_states)
        trans_counts = np.zeros((self.n_states, self.n_states))
        for labels in label_sequences:
            idx = self._state_index(labels)
            start_counts[idx[0]] += 1
            for a, b in zip(idx[:-1], idx[1:]):
                trans_counts[a, b] += 1

        self.startprob_ = (start_counts + 1) / (start_counts + 1).sum()
        self.transmat_ = self._normalise_transitions(trans_counts)

        self.means_ = np.zeros((self.n_states, n_features))
        shape = (self.n_states, n_features) if self.covariance_type == "diag" \
            else (self.n_states, n_features, n_features)
        self.covars_ = np.zeros(shape)
        for k in range(self.n_states):
            Xk = X[y == k]
            self.means_[k] = Xk.mean(axis=0)
            self.covars_[k] = self._regularise(np.cov(Xk, rowvar=False, bias=True))
        return self

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        out = np.empty((n, self.n_states))
        for k in range(self.n_states):
            diff = X - self.means_[k]
            if self.covariance_type == "diag":
                var = self.covars_[k]
                out[:, k] = -0.5 * (d * LOG_2PI + np.sum(np.log(var)) + np.sum(diff ** 2 / var, axis=1))
            else:
                chol = np.linalg.cholesky(self.covars_[k])
                solved = np.linalg.solve(chol, diff.T)
                log_det = 2 * np.sum(np.log(np.diag(chol)))
                out[:, k] = -0.5 * (d * LOG_2PI + log_det + np.sum(solved ** 2, axis=0))
        return out

    def _forward(self, log_b: np.ndarray) -> np.ndarray:
        n = log_b.shape[0]
        log_a = np.log(self.transmat_)
        log_alpha = np.empty((n, self.n_states))
        log_alpha[0] = np.log(self.startprob_) + log_b[0]
        for t in range(1, n):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0) + log_b[t]
        return log_alpha

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        n = log_b.shape[0]
        log_a = np.log(self.transmat_)
        log_beta = np.zeros((n, self.n_states))
        for t in range(n - 2, -1, -1):
            log_beta[t] = logsumexp(log_a + log_b[t + 1] + log_beta[t + 1], axis=1)
        return log_beta

    def score(self, sequences: list) -> float:
        return float(sum(logsumexp(self._forward(self._log_emission(X))[-1]) for X in sequences))

    def _e_step(self, X: np.ndarray) -> tuple:
        log_b = self._log_emission(X)
        log_alpha = self._forward(log_b)
        log_beta = self._backward(log_b)
        log_likelihood = logsumexp(log_alpha[-1])

        gamma = np.exp(log_alpha + log_beta - log_likelihood)

        xi_sum = np.zeros((self.n_states, self.n_states))
        if len(X) > 1:
            log_a = np.log(self.transmat_)
            log_xi = (log_alpha[:-1, :, None] + log_a[None, :, :]
                      + (log_b[1:] + log_beta[1:])[:, None, :] - log_likelihood)
            xi_sum = np.exp(logsumexp(log_xi, axis=0))
        return gamma, xi_sum, log_likelihood

    def _m_step(self, sequences: list, gammas: list, xi_total: np.ndarray) -> None:
        start = sum(g[0] for g in gammas)
        self.startprob_ = (start + 1) / (start + 1).sum()
        self.transmat_ = self._normalise_transitions(xi_total)

        X = np.vstack(sequences)
        G = np.vstack(gammas)
        weights = G.sum(axis=0) + 1e-10
        self.means_ = (G.T @ X) / weights[:, None]
        for k in range(self.n_states):
            diff = X - self.means_[k]
            cov = (G[:, k, None] * diff).T @ diff / weights[k]
            self.covars_[k] = self._regularise(cov)

    def fit(self, sequences: list) -> "GaussianHMM":
        if self.means_ is None:
            raise RuntimeError("Initialise the model with fit_supervised before Baum-Welch")

        self.history_ = []
        self.converged_ = False
        previous = -np.inf
        for iteration in range(1, self.max_iter + 1):
            gammas, xi_total, total_ll = [], np.zeros((self.n_states, self.n_states)), 0.0
            for X in sequences:
                gamma, xi_sum, ll = self._e_step(X)
                gammas.append(gamma)
                xi_total += xi_sum
                total_ll += ll

            self.history_.append(total_ll)
            improvement = total_ll - previous
            if self.verbose:
                print(f"iter {iteration:3d}  log-likelihood {total_ll:12.3f}  delta {improvement:10.5f}")
            if abs(improvement) < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration
                break

            self._m_step(sequences, gammas, xi_total)
            previous = total_ll
        else:
            self.n_iter_ = self.max_iter
        return self

    def viterbi(self, X: np.ndarray) -> tuple:
        log_b = self._log_emission(X)
        log_a = np.log(self.transmat_)
        n = len(X)

        delta = np.empty((n, self.n_states))
        backpointer = np.zeros((n, self.n_states), dtype=int)
        delta[0] = np.log(self.startprob_) + log_b[0]
        for t in range(1, n):
            candidates = delta[t - 1][:, None] + log_a
            backpointer[t] = np.argmax(candidates, axis=0)
            delta[t] = candidates[backpointer[t], np.arange(self.n_states)] + log_b[t]

        path = np.empty(n, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(n - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]
        return np.array(self.states)[path], float(delta[-1, path[-1]])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.viterbi(X)[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._e_step(X)[0]

    def to_dict(self) -> dict:
        return {
            "states": self.states,
            "covariance_type": self.covariance_type,
            "startprob": self.startprob_.tolist(),
            "transmat": self.transmat_.tolist(),
            "means": self.means_.tolist(),
            "covars": self.covars_.tolist(),
            "log_likelihood_history": self.history_,
            "converged": self.converged_,
            "n_iter": self.n_iter_,
            "tol": self.tol,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "GaussianHMM":
        with open(path) as f:
            params = json.load(f)
        model = cls(params["states"], covariance_type=params["covariance_type"], tol=params["tol"])
        model.startprob_ = np.array(params["startprob"])
        model.transmat_ = np.array(params["transmat"])
        model.means_ = np.array(params["means"])
        model.covars_ = np.array(params["covars"])
        model.history_ = params["log_likelihood_history"]
        model.converged_ = params["converged"]
        model.n_iter_ = params["n_iter"]
        return model
