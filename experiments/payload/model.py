"""Uniform comparison matrices, with no symmetry or empirical slope estimation.

All arrays are organized by robot, with one robot per row. The physical model is the exact sampled map
of a known stable vertical compliance servo subject to held uncertain forces.
The model represents vertical load motion and force allocation.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cholesky, expm, solve_continuous_lyapunov
from experiments.payload.config import ModelConfig

GUARD = 1e-10  # Tolerance for rounding in the comparison bounds.


def topology(cfg: ModelConfig):
    W = np.zeros((cfg.n, cfg.n))
    for i, j in cfg.edges:
        if not 0 <= i < cfg.n or not 0 <= j < cfg.n or i == j:
            raise ValueError("Invalid edge")
        W[i, j] = W[j, i] = 1.0
    if np.any(W.sum(1) == 0):
        raise ValueError("An isolated node is unsupported")
    seen = {0}
    for _ in range(cfg.n):
        seen |= {j for i in tuple(seen) for j in np.flatnonzero(W[i])}
    if len(seen) != cfg.n:
        raise ValueError("Graph must be connected")
    return W, np.diag(W.sum(1)) - W, W / W.sum(1)[:, None]


def physical_matrices(cfg):
    P = []
    Q = []
    T = []
    Tinv = []
    rho = []
    beta = []
    cp = []
    alphas = []
    for m, k, d in zip(cfg.mass, cfg.kp, cfg.kd):
        A = np.array([[0.0, 1.0], [-k / m, -d / m]])
        decay = float(min(-np.linalg.eigvals(A).real))
        if decay <= 0:
            raise ValueError("Local servo is not Hurwitz")
        alpha = 0.5 * decay
        H = solve_continuous_lyapunov((A + alpha * np.eye(2)).T, -np.eye(2))
        if np.linalg.eigvalsh(H).min() <= 0:
            raise ValueError("Invalid physical state metric")
        C = cholesky(H, lower=False) if cfg.metric == "lyapunov" else np.eye(2)
        if cfg.metric not in ("lyapunov", "euclidean"):
            raise ValueError("Unknown metric")
        D = np.linalg.inv(C)
        E = expm(cfg.dt * A)
        # The load tension convention gives an altitude increase for a negative force error.
        q = -np.linalg.solve(A, (E - np.eye(2)) @ np.array([0.0, 1.0 / m]))
        P.append(E)
        Q.append(q)
        T.append(C)
        Tinv.append(D)
        # Analytic contraction from A^T H+H A <= -2 alpha H.
        rho.append(
            (
                np.exp(-alpha * cfg.dt)
                if cfg.metric == "lyapunov"
                else np.linalg.norm(E, 2)
            )
            + GUARD
        )
        beta.append(np.linalg.norm(C @ q) + GUARD)
        cp.append(np.linalg.norm(D[0]) + GUARD)
        alphas.append(alpha)
    return tuple(np.array(x) for x in (P, Q, T, Tinv, rho, beta, cp, alphas))


def certificate(cfg: ModelConfig):
    """Compute the allocation and state comparison bounds from the model parameters."""
    W, L, N = topology(cfg)
    n = cfg.n
    cl = np.array(cfg.classes)
    a = np.array(cfg.capability)
    mu = np.array(cfg.mu)
    sc = np.array(cfg.position_scale)
    sigma = np.array(cfg.sigma) if cfg.dynamic else np.zeros(n)
    wmin = a * np.exp(-mu * np.array(cfg.margin_cap)[cl] - sigma)
    wmax = a * np.exp(sigma)
    qlo = 1 / wmax
    qhi = 1 / wmin
    if np.any(cfg.gamma * W.sum(1) * qhi > 1):
        raise ValueError("Step size does not certify nonnegative force iterates")
    Mhi = np.diag(1 - cfg.gamma * W.sum(1) * qlo) + cfg.gamma * W * qhi[None, :]
    Mlo = np.diag(1 - cfg.gamma * W.sum(1) * qhi) + cfg.gamma * W * qlo[None, :]
    Mhi += GUARD * (np.eye(n) + W > 0)
    Mlo = np.maximum(0, Mlo - GUARD * (np.eye(n) + W > 0))
    P, Q, T, Tinv, rho, beta, cp, decay = physical_matrices(cfg)
    ub = cfg.force_max * a / a.sum()
    lb = cfg.force_min * a / a.sum()
    initial_ub = ub.copy()
    initial_lb = lb.copy()
    g = np.zeros(n)
    C = np.zeros((n, n))
    E = np.eye(2)[cl]
    histories = []
    for ell in range(cfg.rounds):
        # Bound each derivative separately, including modes shared across the class.
        # Column h has support only in h and its neighbors.
        R = cfg.gamma * np.abs(L) @ np.diag(mu * qhi * ub) @ E
        b = np.linalg.norm(R, axis=1) + GUARD
        V = cfg.gamma * np.abs(L) @ np.diag((sigma / sc) * qhi * ub * cp)
        V += GUARD * (V > 0)
        g = Mhi @ g + b
        C = Mhi @ C + V
        ub = np.minimum(Mhi @ ub, cfg.force_max)
        lb = np.maximum(Mlo @ lb, np.maximum(0, cfg.force_min - (np.sum(ub) - ub)))
        histories.append(
            dict(
                b=b.tolist(),
                V=V.tolist(),
                g=g.tolist(),
                C=C.tolist(),
                ub=ub.tolist(),
                lb=lb.tolist(),
            )
        )
    H = (
        np.diag(np.array(cfg.theta_bound)[cl])
        + np.diag(np.array(cfg.phi_bound)[cl]) @ N
    )
    state_amp = cfg.position_residual_bound if cfg.dynamic else 0.0
    Jx = np.diag(state_amp / sc * cp)
    Ac = np.diag(rho) + np.diag(beta) @ ((np.eye(n) + H) @ C + Jx)
    bc = beta * ((np.eye(n) + H) @ g)
    Lscore = H @ C + Jx
    direct = H @ g
    G = np.zeros(n)
    Gbar = np.zeros(n)
    Gh = []
    # Score is evaluated at the beginning of each of the steps intervals.
    for k in range(cfg.steps):
        Gbar = np.maximum(Gbar, G)
        Gh.append(G.copy())
        G = Ac @ G + bc
    robot_kappa = direct + Lscore @ Gbar
    if not cfg.dynamic:
        robot_kappa = direct
    robot_kappa += GUARD
    kappa = np.array([robot_kappa[cl == c].max() for c in range(2)])
    support = H @ ub + state_amp + cfg.eta_bound
    actual_min = lb - support
    actual_max = ub + support
    actuator_cert = bool(
        np.all(ub < np.array(cfg.command_cap)[cl])
        and np.all(actual_min > 0)
        and np.all(actual_max < np.array(cfg.actual_force_cap)[cl])
    )
    # Uniform bound on the physical state in the comparison recursion.
    xinit = np.array(
        [
            max(
                np.linalg.norm(T[i] @ np.array([sp, sv]))
                for sp in (-cfg.initial_position_support, cfg.initial_position_support)
                for sv in (-cfg.initial_velocity_support, cfg.initial_velocity_support)
            )
            for i in range(n)
        ]
    )
    input_bound = np.maximum(np.abs(lb - initial_ub), np.abs(ub - initial_lb)) + support
    xbound = xinit.copy()
    xb = xinit.copy()
    for _ in range(cfg.steps):
        xb = rho * xb + beta * input_bound
        xbound = np.maximum(xbound, xb)
    return dict(
        config=cfg.dictionary(),
        certificate_method="analytic uniform derivatives + force envelopes + finite comparison recursion",
        floating_point_note="Formula evaluations padded by 1e-10; not a formal interval-arithmetic proof",
        margin_metric=np.eye(2).tolist(),
        wmin=wmin.tolist(),
        wmax=wmax.tolist(),
        A_allocation=Mhi.tolist(),
        allocation_histories=histories,
        command_upper=ub.tolist(),
        command_lower=lb.tolist(),
        allocation_margin_gain=g.tolist(),
        allocation_state_gain=C.tolist(),
        error_input_gain=H.tolist(),
        error_state_gain=Jx.tolist(),
        physical_decay=decay.tolist(),
        rho=rho.tolist(),
        beta=beta.tolist(),
        position_projection=cp.tolist(),
        A_dt=Ac.tolist(),
        b_dt=bc.tolist(),
        G_history=np.array(Gh).tolist(),
        G_bar=Gbar.tolist(),
        L_dt=Lscore.tolist(),
        L_r_dt=direct.tolist(),
        robot_kappa=robot_kappa.tolist(),
        kappa=kappa.tolist(),
        spectral_radius=float(max(abs(np.linalg.eigvals(Ac)))),
        physical_state_support=xbound.tolist(),
        score_support=support.tolist(),
        actual_force_lower=actual_min.tolist(),
        actual_force_upper=actual_max.tolist(),
        actuator_certificate=actuator_cert,
        physical_model="Independent stable local vertical compliance servos coupled through state-dependent allocation and neighbor force errors; no rigid cable constraints",
    )


@dataclass(frozen=True)
class Noise:
    theta: np.ndarray
    phi: np.ndarray
    poscoef: np.ndarray
    eta: np.ndarray
    initial: np.ndarray


def sample_missions(cfg, count, seed, law="biased"):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    cl = np.array(cfg.classes)
    if law == "biased":
        sh1 = rng.uniform(0.1, 0.8, (count, 1))
        sh2 = rng.uniform(0.7, 1.0, (count, 1))
        ind = rng.random((count, 6))
        th = np.where(
            cl == 0, 0.12 + 0.18 * sh1 + 0.12 * ind, 0.74 + 0.18 * sh2 + 0.06 * ind
        )
        ph = np.where(
            cl == 0, 0.08 + 0.15 * sh1 + 0.10 * ind, 0.58 + 0.28 * sh2 + 0.10 * ind
        )
    elif law == "centered":
        sh = rng.uniform(-1, 1, (count, 1))
        th = 0.65 * sh + 0.35 * rng.uniform(-1, 1, (count, 6))
        ph = 0.5 * sh + 0.5 * rng.uniform(-1, 1, (count, 6))
    elif law == "corners":
        th = rng.choice([-1.0, 1.0], (count, 6))
        ph = rng.choice([-1.0, 1.0], (count, 6))
    else:
        raise ValueError("Unknown noise law")
    eta = (
        rng.uniform(
            0.0 if law == "biased" else -cfg.eta_shared, cfg.eta_shared, (count, 1, 1)
        )
        + rng.uniform(
            0.0 if law == "biased" else -cfg.eta_robot, cfg.eta_robot, (count, 6, 1)
        )
        + rng.uniform(
            0.0 if law == "biased" else -cfg.eta_step,
            cfg.eta_step,
            (count, cfg.n, cfg.steps),
        )
    )
    initial = np.stack(
        [
            rng.uniform(
                -cfg.initial_position_support, cfg.initial_position_support, (count, 6)
            ),
            rng.uniform(
                -cfg.initial_velocity_support, cfg.initial_velocity_support, (count, 6)
            ),
        ],
        axis=-1,
    )
    out = Noise(
        th * np.array(cfg.theta_bound)[cl],
        ph * np.array(cfg.phi_bound)[cl],
        rng.uniform(
            -cfg.position_residual_bound, cfg.position_residual_bound, (count, 6)
        ),
        eta,
        initial,
    )
    for x in (out.theta, out.phi, out.poscoef, out.eta, out.initial):
        x.setflags(write=False)
    return out


def allocation(cfg, r, x, F, collect=False):
    """Run the neighbor exchanges for a single force demand. Reinitialize the force commands at each time step and carry the physical state between steps."""
    a = np.array(cfg.capability)
    cl = np.array(cfg.classes)
    r = np.asarray(r, float)
    if r.shape != (2,) or np.any(r < 0) or np.any(r > cfg.margin_cap):
        raise ValueError("Margin outside domain")
    W, L, N = topology(cfg)
    ex = -np.array(cfg.mu) * r[cl]
    if cfg.dynamic:
        ex = ex + np.array(cfg.sigma) * np.tanh(
            x[:, :, 0] / np.array(cfg.position_scale)
        )
    w = a * np.exp(ex)
    w = np.broadcast_to(w, x.shape[:2])
    u = np.broadcast_to(F * a / a.sum(), x.shape[:2]).copy()
    history = [u.copy()] if collect else []
    for ell in range(cfg.rounds):
        packet = u / w
        # One neighbor exchange per allocation iteration.
        inbox = [sum(packet[:, j] for j in np.flatnonzero(W[i])) for i in range(cfg.n)]
        u = u - cfg.gamma * (W.sum(1) * packet - np.column_stack(inbox))
        if collect:
            history.append(u.copy())
    return u, history


@dataclass
class MissionBatch:
    node_scores: tuple[np.ndarray, ...]
    scores: np.ndarray
    max_position_error: np.ndarray
    rms_position_error: np.ndarray
    max_force_error: np.ndarray
    all_commands: np.ndarray | None
    all_states: np.ndarray | None
    trace: dict


def simulate(cfg, r, noise, keep=False):
    count = len(noise.theta)
    W, L, N = topology(cfg)
    P, Q, T, Tinv, rho, beta, cp, alpha = physical_matrices(cfg)
    x = noise.initial.copy()
    a = np.array(cfg.capability)
    score = np.zeros((count, 6))
    pmax = np.abs(x[:, :, 0])
    psum = np.zeros((count, 6))
    commands = []
    states = []
    tr_u = []
    tr_x = []
    tr_eps = []
    for k in range(cfg.steps):
        F = cfg.force_center + cfg.force_amplitude * np.sin(
            2 * np.pi * (k + 0.5) / cfg.steps
        )
        u, _ = allocation(cfg, r, x, F)
        eps = noise.theta * u + noise.phi * (u @ N.T) + noise.eta[:, :, k]
        if cfg.dynamic:
            eps = eps + noise.poscoef * np.tanh(
                x[:, :, 0] / np.array(cfg.position_scale)
            )
        score = np.maximum(score, np.abs(eps))
        pmax = np.maximum(pmax, np.abs(x[:, :, 0]))
        psum += x[:, :, 0] ** 2
        tr_u.append(u[0].copy())
        tr_x.append(x[0].copy())
        tr_eps.append(eps[0].copy())
        if keep:
            commands.append(u.copy())
            states.append(x.copy())
        if cfg.dynamic:
            deviation = u - F * a / a.sum() + eps
            x = np.einsum("nij,mnj->mni", P, x) + deviation[:, :, None] * Q[None]
    pmax = np.maximum(pmax, np.abs(x[:, :, 0]))
    tr_x.append(x[0].copy())
    return MissionBatch(
        tuple(score[:, i].copy() for i in range(6)),
        score,
        pmax,
        np.sqrt(psum / cfg.steps),
        np.max(score, axis=1),
        np.stack(commands, axis=1) if keep else None,
        np.stack(states, axis=1) if keep else None,
        dict(
            commands=np.stack(tr_u),
            states=np.stack(tr_x),
            residual=np.stack(tr_eps),
            margins=np.array(r),
            time=np.arange(cfg.steps + 1) * cfg.dt,
        ),
    )


def class_scores(batch, cfg):
    cl = np.array(cfg.classes)
    return np.column_stack([batch.scores[:, cl == c].max(1) for c in range(2)])
