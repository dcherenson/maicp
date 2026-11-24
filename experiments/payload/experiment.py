"""Payload transport with feedback and paired sensitivity checks."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import beta

from experiments.payload.calibration import (
    Bus,
    calibrate,
    decimal_slacks,
    margin_update,
)
from experiments.payload.config import ModelConfig
from experiments.payload.model import (
    certificate,
    class_scores,
    physical_matrices,
    sample_missions,
    simulate,
)
from maicp.ranks import binomial_calibration_rank


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def hashes():
    directory = Path(__file__).resolve().parent
    sources = [*directory.glob("*.py"), directory.parents[1] / "maicp" / "ranks.py"]
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(sources)
    }


def ci(k, n, fail):
    return 0.0 if k == 0 else float(beta.ppf(fail, k, n - k + 1))


def experiment(path, cfg, exp):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Do not overwrite registered results")
    path.mkdir(parents=True)
    (path / "data").mkdir()
    cert = certificate(cfg)
    if not cert["actuator_certificate"]:
        raise ValueError("No uniform force capacity certificate")
    reg = dict(
        config=cfg.dictionary(),
        experiment=exp.dictionary(),
        certificate=cert,
        source_hash=hashes(),
        protocol="Fresh iid calibration missions each update; paired MA-ICP/nominal evaluation; no score-based parameter selection",
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
        ),
        guarantee="Per class and per update. Numerical feasibility checked for each deployed policy. No claim of feasibility for every possible calibration batch.",
    )
    write(path / "registration.json", reg)
    rows = []
    updates = []
    per_error = 0.05 / (exp.replicates * (exp.updates + 1) * 2 * 3)
    kappa = np.array(cert["kappa"])
    cl = np.array(cfg.classes)
    for rep in range(exp.replicates):
        r = np.array(exp.initial_margin)
        for j in range(exp.updates + 1):
            noise = sample_missions(
                cfg, exp.evaluation_missions, (exp.seed, 20, rep, j)
            )
            arrays = {}
            for method, rr in [("maicp", r), ("nominal", np.zeros(2))]:
                batch = simulate(cfg, rr, noise)
                scores = class_scores(batch, cfg)
                cover = scores <= rr
                counts = [int(cover[:, c].sum()) for c in range(2)] + [
                    int(cover.all(1).sum())
                ]
                row = dict(
                    replicate=rep,
                    episode=j,
                    method=method,
                    margins=rr.tolist(),
                    missions=exp.evaluation_missions,
                    counts=counts,
                    coverage=[k / exp.evaluation_missions for k in counts],
                    mc_lower=[
                        ci(k, exp.evaluation_missions, per_error) for k in counts
                    ],
                    mean_rms_position=float(batch.rms_position_error.mean()),
                    mean_max_robot_position=float(
                        batch.max_position_error.max(1).mean()
                    ),
                    mean_rms_position_by_class=[
                        float(batch.rms_position_error[:, cl == c].mean())
                        for c in range(2)
                    ],
                    mean_max_force_error=float(batch.max_force_error.mean()),
                    mean_class2_share=float(
                        batch.trace["commands"][:, cl == 1].sum(1).mean()
                        / cfg.force_center
                    ),
                )
                rows.append(row)
                arrays[method + "_robot_scores"] = batch.scores
                arrays[method + "_class_scores"] = scores
                arrays[method + "_margin"] = rr.copy()
                arrays[method + "_rms_position"] = batch.rms_position_error
                arrays[method + "_max_position"] = batch.max_position_error
                if rep == 0 and j in (0, exp.updates):
                    np.savez_compressed(path / f"trace_{method}_{j}.npz", **batch.trace)
            if j < exp.updates:
                calbatch = simulate(
                    cfg,
                    r,
                    sample_missions(
                        cfg, exp.calibration_missions, (exp.seed, 10, rep, j)
                    ),
                )
                bus = Bus()
                cal = calibrate(calbatch, cfg, exp, bus)
                q = cal.quantiles_by_node[0]
                next_by_node = []
                details = []
                for qq in cal.quantiles_by_node:
                    nr, detail = margin_update(r, qq, kappa, cfg, exp)
                    next_by_node.append(nr)
                    details.append(detail)
                assert all(np.array_equal(next_by_node[0], nr) for nr in next_by_node)
                nr = next_by_node[0]
                ranks = [v["rank"] for v in cal.reports]
                exact = np.sort(cal.mission_scores, axis=0)[np.asarray(ranks) - 1, np.arange(len(ranks))]
                assert np.all(q >= exact)
                updates.append(
                    dict(
                        replicate=rep,
                        from_episode=j,
                        old=r.tolist(),
                        new=nr.tolist(),
                        quantiles=q.tolist(),
                        exact_quantiles=exact.tolist(),
                        quantile_gap=(q - exact).tolist(),
                        slacks=[float(v) for v in decimal_slacks(nr, r, q, kappa)],
                        solver=details[0],
                        communication=bus.traffic.dictionary(),
                        ranks=ranks,
                    )
                )
                arrays.update(
                    calibration_robot_scores=calbatch.scores,
                    calibration_mission_scores=cal.mission_scores,
                    calibration_margin=r,
                    quantiles=q,
                    new_margin=nr,
                )
                r = nr
            np.savez_compressed(
                path / "data" / f"rep_{rep:02d}_episode_{j:02d}.npz", **arrays
            )
        print(path.name, rep + 1, exp.replicates, "margin", r, flush=True)
    report = dict(
        reg, evaluations=rows, updates=updates, completed=True, mc_per_bound=per_error
    )
    post = [row for row in rows if row["method"] == "maicp" and row["episode"] > 0]
    summary = dict(
        kappa=cert["kappa"],
        spectral_radius=cert["spectral_radius"],
        minimum_coverage=np.min([row["coverage"] for row in post], axis=0).tolist(),
        minimum_mc_lower=np.min([row["mc_lower"] for row in post], axis=0).tolist(),
        minimum_slack=min(min(u["slacks"]) for u in updates),
        maximum_quantile_gap=max(max(u["quantile_gap"]) for u in updates),
        terminal={},
    )
    for method in ["maicp", "nominal"]:
        selected = [
            row
            for row in rows
            if row["episode"] == exp.updates and row["method"] == method
        ]
        summary["terminal"][method] = {
            key: np.mean([row[key] for row in selected], axis=0).tolist()
            for key in [
                "margins",
                "coverage",
                "mean_rms_position",
                "mean_max_robot_position",
                "mean_rms_position_by_class",
                "mean_max_force_error",
                "mean_class2_share",
            ]
        }
    report["summary"] = summary
    write(path / "report.json", report)
    write(path / "summary.json", summary)
    return report


def sensitivity(path, cfg, missions=256, pairs=32, seed=870413):
    path = Path(path)
    cert = certificate(cfg)
    rng = np.random.default_rng(seed)
    allpairs = [
        (np.array(a, float), np.array(b, float))
        for a, b in [
            ((0, 0), (2, 0)),
            ((0, 0), (0, 2)),
            ((0, 0), (2, 2)),
            ((0, 2), (2, 0)),
            ((2, 2), (0, 2)),
            ((2, 2), (2, 0)),
        ]
    ]
    while len(allpairs) < pairs:
        allpairs.append((rng.uniform(0, 2, 2), rng.uniform(0, 2, 2)))
    write(
        path / "registration.json",
        dict(
            config=cfg.dictionary(),
            missions=missions,
            pairs=pairs,
            seed=seed,
            laws=["biased", "centered", "corners"],
            source_hash=hashes(),
        ),
    )
    T = physical_matrices(cfg)[2]
    Cu = np.array(cert["allocation_state_gain"])
    gu = np.array(cert["allocation_margin_gain"])
    A = np.array(cert["A_dt"])
    b = np.array(cert["b_dt"])
    records = []
    failures = []
    cl = np.array(cfg.classes)
    robot_ratio = np.zeros(6)
    worst_state_delta = 0.0
    trace = None
    raw = []
    for li, law in enumerate(["biased", "centered", "corners"]):
        noise = sample_missions(cfg, missions, (seed, li), law)
        for pi, (r, s) in enumerate(allpairs):
            d = np.linalg.norm(s - r)
            x = simulate(cfg, r, noise, True)
            y = simulate(cfg, s, noise, True)
            shift = np.abs(y.scores - x.scores)
            robot_ratio = np.maximum(robot_ratio, np.max(shift, axis=0) / d)
            if np.any(shift > np.array(cert["robot_kappa"]) * d + 1e-10):
                failures.append([law, pi, "robot score"])
            ms = np.abs(class_scores(y, cfg) - class_scores(x, cfg))
            bound = np.array(cert["kappa"]) * d
            if np.any(ms > bound + 1e-10):
                failures.append([law, pi, "mission score"])
            if cfg.dynamic:
                dx = np.linalg.norm(
                    np.einsum("nij,mknj->mkni", T, y.all_states - x.all_states), axis=-1
                )
                du = np.abs(y.all_commands - x.all_commands)
                worst_state_delta = max(
                    worst_state_delta, float(np.max(abs(y.all_states - x.all_states)))
                )
                if np.any(du > dx @ Cu.T + gu * d + 1e-10):
                    failures.append([law, pi, "allocation state/radius bound"])
                if np.any(dx[:, 1:] > dx[:, :-1] @ A.T + b * d + 1e-10):
                    failures.append([law, pi, "physical recursion"])
                if np.any(dx > np.array(cert["G_history"]) * d + 1e-10):
                    failures.append([law, pi, "physical accumulated gain"])
                if trace is None:
                    trace = dict(
                        r=r,
                        s=s,
                        observed_max_state=dx.max(0),
                        certified_state=np.array(cert["G_history"]) * d,
                    )
            records.append(
                dict(
                    law=law,
                    pair=pi,
                    r=r.tolist(),
                    s=s.tolist(),
                    distance=d,
                    shift_bound=bound.tolist(),
                    maximum_mission_shift=ms.max(0).tolist(),
                    maximum_robot_shift=shift.max(0).tolist(),
                )
            )
            raw.append(ms)
    ratio_class = [float(robot_ratio[cl == c].max()) for c in range(2)]
    result = dict(
        passed=not failures,
        failures=failures,
        kappa=cert["kappa"],
        robot_ratios=robot_ratio.tolist(),
        maximum_robot_ratio_by_class=ratio_class,
        maximum_mission_ratio_by_class=np.max(
            [np.array(v["maximum_mission_shift"]) / v["distance"] for v in records],
            axis=0,
        ).tolist(),
        maximum_observed_physical_state_difference=worst_state_delta,
        missions_per_law=missions,
        pairs=pairs,
        laws=3,
        mission_pair_comparisons=missions * pairs * 3,
        records=records,
        statement="Finite diagnostic checks are not proof, and are not used to select any sensitivity coefficient.",
    )
    write(path / "checks.json", result)
    np.savez_compressed(path / "score_shifts.npz", shifts=np.stack(raw))
    if trace is not None:
        np.savez_compressed(path / "state_comparison.npz", **trace)
    return result


def audit(path):
    p = Path(path)
    r = json.loads((p / "report.json").read_text())
    cfg = ModelConfig(**r["config"])
    k = np.array(certificate(cfg)["kappa"])
    assert np.array_equal(k, r["certificate"]["kappa"])
    for row in r["evaluations"]:
        with np.load(
            p / "data" / f"rep_{row['replicate']:02d}_episode_{row['episode']:02d}.npz"
        ) as d:
            method = row["method"]
            scores = d[method + "_robot_scores"]
            cl = np.array(cfg.classes)
            mis = np.column_stack([scores[:, cl == c].max(1) for c in range(2)])
            np.testing.assert_array_equal(mis, d[method + "_class_scores"])
            cover = mis <= d[method + "_margin"]
            cnt = [int(cover[:, c].sum()) for c in range(2)] + [int(cover.all(1).sum())]
            assert cnt == row["counts"]
            np.testing.assert_array_equal(
                [ci(v, row["missions"], r["mc_per_bound"]) for v in cnt],
                row["mc_lower"],
            )
    exp = r["experiment"]
    ranks = [
        binomial_calibration_rank(exp["calibration_missions"], alpha, delta)
        for alpha, delta in zip(exp["alpha"], exp["delta_cal"], strict=True)
    ]
    for u in r["updates"]:
        assert (
            min(
                decimal_slacks(
                    np.array(u["new"]), np.array(u["old"]), np.array(u["quantiles"]), k
                )
            )
            >= 0
        )
        assert u["ranks"] == ranks
        with np.load(
            p / "data" / f"rep_{u['replicate']:02d}_episode_{u['from_episode']:02d}.npz"
        ) as d:
            exact = np.sort(d["calibration_mission_scores"], axis=0)[np.asarray(ranks) - 1, np.arange(len(ranks))]
            assert np.all(d["quantiles"] >= exact)
    out = dict(
        passed=True,
        evaluation_groups=len(r["evaluations"]),
        updates=len(r["updates"]),
        source_hashes_unchanged=hashes() == r["source_hash"],
    )
    write(p / "audit.json", out)
    return out


def run(output):
    from maicp.visualization import paper_figures
    from .config import ExperimentConfig

    model = ModelConfig(dynamic=True)
    experiment(output, model, ExperimentConfig())
    audit(output)
    sensitivity(output / "sensitivity", model)
    paper_figures("payload", output)
