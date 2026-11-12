"""DocMIA white-box attacks (Algorithm 1 of the paper) with the FL, FLLoRA and IG variants.

For every (document, question, answer) pair the attack starts from the target
model, optimises one object with Adam to fit the ground-truth answer, and
records

* the distance travelled  Delta = ||theta - theta_0||_2,
* the number of optimisation steps s until the loss stops improving by at
  least tau (or the step budget S is reached), and
* the answer utility u (ACC) averaged over the optimisation trace.

The optimised object is the final answer-projection layer (FL), LoRA factors on
that layer (FLLoRA; A Kaiming-initialised, B zero, so optimisation starts at
the target), or the document image pixels (IG).  Features are aggregated over the document's questions with
{AVG, MIN, MAX, MED}, standardised, and clustered with KMeans; the cluster with
the larger mean distance is the non-member cluster, and the membership score of
a document is minus its distance to the member centroid.

Pairs are processed in batches for speed only: each pair owns its own copy of
the optimised tensor and its own Adam moments, and stops independently, so the
result equals running Algorithm 1 pair by pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans

from data.audit_io import Candidates
from data.target_model import TinyDocVQA

ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
AGGREGATIONS = ("avg", "min", "max", "med")


@dataclass
class Trajectory:
    loss: np.ndarray      # [P, S]   loss at the start of step s (NaN once stopped)
    utility: np.ndarray   # [P, S]   ACC at the start of step s
    distance: np.ndarray  # [P, S+1] ||theta_s - theta_0||_2 after s updates


def _adam_update(param, grad, m, v, t: int, lr: float) -> None:
    """In-place torch.optim.Adam step (no weight decay) on stacked independent rows."""
    b1, b2 = ADAM_BETAS
    m.mul_(b1).add_(grad, alpha=1 - b1)
    v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
    denom = (v.sqrt() / math.sqrt(1 - b2 ** t)).add_(ADAM_EPS)
    param.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))


def _optimize(params: list[torch.Tensor], objective, steps: int, lr: float,
              tau: float | None) -> Trajectory:
    """Run Algorithm 1's inner loop for a batch of independent pairs.

    ``objective(sub_params, rows)`` returns (loss [a], correct [a], grads list)
    for the active rows.  ``tau=None`` disables early stopping (full trace).
    """
    p = params[0].shape[0]
    init = [x.clone() for x in params]
    m = [torch.zeros_like(x) for x in params]
    v = [torch.zeros_like(x) for x in params]
    loss_tr = np.full((p, steps), np.nan)
    util_tr = np.full((p, steps), np.nan)
    dist_tr = np.full((p, steps + 1), np.nan)
    dist_tr[:, 0] = 0.0
    prev = torch.full((p,), float("inf"), dtype=torch.float64)
    active = torch.arange(p)
    for s in range(steps):
        if active.numel() == 0:
            break
        sub = [x[active] for x in params]
        loss, correct, grads = objective(sub, active)
        loss = loss.detach().to(torch.float64)
        rows = active.numpy()
        loss_tr[rows, s] = loss.numpy()
        util_tr[rows, s] = correct.to(torch.float64).numpy()
        if tau is not None and s > 0:
            keep = (prev[active] - loss) >= tau  # stop once the improvement is below tau
        else:
            keep = torch.ones_like(loss, dtype=torch.bool)
        prev[active] = loss
        active = active[keep]
        if active.numel() == 0:
            break
        sq = torch.zeros(active.numel(), dtype=torch.float64)
        for k in range(len(params)):
            pk, gk = sub[k][keep], grads[k][keep]
            mk, vk = m[k][active], v[k][active]
            _adam_update(pk, gk, mk, vk, s + 1, lr)
            params[k][active] = pk
            m[k][active] = mk
            v[k][active] = vk
            diff = (pk - init[k][active]).reshape(pk.shape[0], -1).to(torch.float64)
            sq += (diff * diff).sum(1)
        dist_tr[active.numpy(), s + 1] = sq.sqrt().numpy()
    return Trajectory(loss_tr, util_tr, dist_tr)


def _concat(parts: list[Trajectory]) -> Trajectory:
    return Trajectory(
        np.concatenate([t.loss for t in parts]),
        np.concatenate([t.utility for t in parts]),
        np.concatenate([t.distance for t in parts]),
    )


def _softmax_grad(logits: torch.Tensor, answers: torch.Tensor):
    logp = torch.log_softmax(logits, dim=1)
    rows = torch.arange(logits.shape[0])
    loss = -logp[rows, answers]
    grad = logp.exp()
    grad[rows, answers] -= 1.0
    correct = logits.argmax(1) == answers
    return loss, correct, grad


@torch.no_grad()
def hidden_states(model: TinyDocVQA, cand: Candidates, batch: int = 256) -> torch.Tensor:
    """Penultimate hidden state h(x, q) of the target for every QA pair."""
    out = []
    for start in range(0, cand.pair_doc.numel(), batch):
        idx = slice(start, start + batch)
        pages = cand.pages[cand.pair_doc[idx]]
        out.append(model.hidden_state(model.encode_page(pages), cand.pair_question[idx]))
    return torch.cat(out)


def fl_trajectories(model: TinyDocVQA, cand: Candidates, lr: float, steps: int,
                    tau: float | None, chunk: int = 1024, h: torch.Tensor | None = None) -> Trajectory:
    """FL: optimise the final answer-projection layer (weight and bias)."""
    h = (hidden_states(model, cand) if h is None else h).to(torch.float64)
    w0 = model.output_layer.weight.detach().to(torch.float64)
    b0 = model.output_layer.bias.detach().to(torch.float64)
    answers = cand.pair_answer
    parts = []
    for start in range(0, h.shape[0], chunk):
        hc, ac = h[start:start + chunk], answers[start:start + chunk]
        n = hc.shape[0]
        params = [w0.expand(n, -1, -1).clone(), b0.expand(n, -1).clone()]

        def objective(sub, rows, hc=hc, ac=ac):
            hr, ar = hc[rows], ac[rows]
            logits = torch.einsum("pah,ph->pa", sub[0], hr) + sub[1]
            loss, correct, g = _softmax_grad(logits, ar)
            return loss, correct, [g[:, :, None] * hr[:, None, :], g]

        parts.append(_optimize(params, objective, steps, lr, tau))
    return _concat(parts)


def fllora_trajectories(model: TinyDocVQA, cand: Candidates, lr: float, steps: int,
                        tau: float | None, rank: int = 4, alpha: float = 4.0, seed: int = 0,
                        chunk: int = 1024, h: torch.Tensor | None = None) -> Trajectory:
    """FLLoRA: optimise Kaiming-initialised LoRA factors attached to the final layer."""
    h = (hidden_states(model, cand) if h is None else h).to(torch.float64)
    w0 = model.output_layer.weight.detach().to(torch.float64)
    b0 = model.output_layer.bias.detach().to(torch.float64)
    num_answers, dim = w0.shape
    scaling = alpha / rank
    gen = torch.Generator().manual_seed(seed)
    total = h.shape[0]
    # LoRA initialisation: A ~ kaiming_uniform_(a=sqrt(5)) (bound 1/sqrt(fan_in)), B = 0, so the
    # adapted model equals the target at the start (Algorithm 1: theta_0 = theta_t).
    lora_a = (torch.rand(total, rank, dim, generator=gen, dtype=torch.float64) * 2 - 1) / math.sqrt(dim)
    lora_b = torch.zeros(total, num_answers, rank, dtype=torch.float64)
    answers = cand.pair_answer
    parts = []
    for start in range(0, total, chunk):
        hc, ac = h[start:start + chunk], answers[start:start + chunk]
        base = hc @ w0.T + b0
        params = [lora_a[start:start + chunk].clone(), lora_b[start:start + chunk].clone()]

        def objective(sub, rows, hc=hc, ac=ac, base=base):
            hr, ar = hc[rows], ac[rows]
            z = torch.einsum("prh,ph->pr", sub[0], hr)
            logits = base[rows] + scaling * torch.einsum("par,pr->pa", sub[1], z)
            loss, correct, g = _softmax_grad(logits, ar)
            grad_b = scaling * g[:, :, None] * z[:, None, :]
            gz = scaling * torch.einsum("par,pa->pr", sub[1], g)
            grad_a = gz[:, :, None] * hr[:, None, :]
            return loss, correct, [grad_a, grad_b]

        parts.append(_optimize(params, objective, steps, lr, tau))
    return _concat(parts)


def ig_trajectories(model: TinyDocVQA, cand: Candidates, lr: float, steps: int,
                    tau: float | None, chunk: int = 384) -> Trajectory:
    """IG: optimise the document image pixels with the target frozen."""
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    parts = []
    total = cand.pair_doc.numel()
    for start in range(0, total, chunk):
        docs = cand.pair_doc[start:start + chunk]
        qc, ac = cand.pair_question[start:start + chunk], cand.pair_answer[start:start + chunk]
        params = [cand.pages[docs].clone()]

        def objective(sub, rows, qc=qc, ac=ac):
            x = sub[0].detach().requires_grad_(True)
            with torch.enable_grad():
                logits = model(x, qc[rows])
                loss = torch.nn.functional.cross_entropy(logits, ac[rows], reduction="none")
                loss.sum().backward()
            return loss.detach(), (logits.detach().argmax(1) == ac[rows]), [x.grad.detach()]

        parts.append(_optimize(params, objective, steps, lr, tau))
    return _concat(parts)


def pair_features(traj: Trajectory, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the stopping rule with threshold tau to a trajectory -> (Delta, s, u)."""
    loss, util, dist = traj.loss, traj.utility, traj.distance
    p, steps = loss.shape
    rows = np.arange(p)
    if steps < 2:
        has, first = np.zeros(p, dtype=bool), np.ones(p, dtype=np.int64)
    else:
        with np.errstate(invalid="ignore"):
            stop = (loss[:, :-1] - loss[:, 1:]) < tau
        has = stop.any(1)
        first = np.argmax(stop, 1) + 1
    n_steps = np.where(has, first, steps)
    delta = dist[rows, n_steps]
    last = np.where(has, first, steps - 1)
    utility = np.nancumsum(util, 1)[rows, last] / (last + 1)
    return delta, n_steps.astype(np.float64), utility


def aggregate(values: np.ndarray, pair_doc: np.ndarray, num_docs: int) -> np.ndarray:
    """Phi_all = {AVG, MIN, MAX, MED} over each document's questions -> [N, 4]."""
    out = np.zeros((num_docs, len(AGGREGATIONS)))
    order = np.argsort(pair_doc, kind="stable")
    bounds = np.searchsorted(pair_doc[order], np.arange(num_docs + 1))
    for d in range(num_docs):
        vals = values[order[bounds[d]:bounds[d + 1]]]
        out[d] = (vals.mean(), vals.min(), vals.max(), np.median(vals))
    return out


def document_features(traj: Trajectory, tau: float, cand: Candidates) -> np.ndarray:
    """F_x = [Phi_all(Delta), Phi_all(s), Phi_all(u)] for every candidate document."""
    delta, steps, utility = pair_features(traj, tau)
    pair_doc = cand.pair_doc.numpy()
    return np.concatenate([aggregate(x, pair_doc, cand.num_docs) for x in (delta, steps, utility)], axis=1)


def cluster_membership(features: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """KMeans(2) on standardised descriptors; larger mean AVG(Delta) cluster = non-members."""
    std = features.std(0)
    z = (features - features.mean(0)) / np.where(std > 0, std, 1.0)
    if np.unique(np.round(z, 12), axis=0).shape[0] < 2:
        return np.zeros(len(features), dtype=np.int64), np.zeros(len(features))
    labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(z)
    if len(np.unique(labels)) < 2:
        return np.zeros(len(features), dtype=np.int64), np.zeros(len(features))
    mean_delta = [features[labels == c, 0].mean() for c in (0, 1)]
    member = int(np.argmin(mean_delta))
    centroid = z[labels == member].mean(0)
    preds = (labels == member).astype(np.int64)
    scores = -np.linalg.norm(z - centroid[None, :], axis=1)
    return preds, scores
