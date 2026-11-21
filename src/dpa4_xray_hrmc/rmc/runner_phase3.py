"""Executable Phase 3B alternating and Phase 3C inline pilot runners."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
import numpy as np
from ase.io import write

from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend
from dpa4_xray_hrmc.mlip.fire_relaxation import run_fixed_cell_fire
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.structure import load_structure
from .config import Phase2Config
from .coordination import CoordinationCutoffs
from .phase3 import Phase3State, inline_delta, metropolis_accept
from .runner import inclusive_grid


def _setup(root: Path, config: Phase2Config, cutoffs, weights, scale_mode="no_scale"):
    atoms = load_structure(root / config.structure).atoms
    exp = read_pdfgetx_gr(root / config.experimental_gr)
    mask = (exp.r_A >= config.fit_rmin_A) & (exp.r_A <= config.fit_rmax_A)
    q = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)
    state = Phase3State.build(atoms, q, exp.r_A[mask], exp.g_r_A2[mask],
                              config, cutoffs, weights, mode=scale_mode)
    return state, q, exp, mask


def _summary(state, energy, counters, wall, extra=None):
    common = state.objective.common_scores(state.score.A, state.score.B)
    snap = state.cn.current_snapshot()
    value = {
        "attempted": counters["attempted"], "accepted": counters["accepted"],
        "hard_sphere_rejected": counters["hard_sphere_rejected"],
        "metropolis_rejected": counters["metropolis_rejected"],
        "acceptance_rate": counters["accepted"] / max(counters["attempted"], 1),
        "energy_eV": energy, "energy_eV_atom": energy / len(state.atoms),
        "pdf_loss_sum": state.score.loss, "phase25_total": state.total,
        "Rw_no_scale": common["Rw_no_scale"],
        "Rw_fixed_initial": common["Rw_fixed_initial"],
        "Rw_bounded_dynamic": common["Rw_bounded_dynamic"],
        "Rw_shape_only": common["Rw_shape_only"],
        "Rw_active": float(np.sqrt(state.score.loss / state.objective.C)),
        "scale_strategy": state.mode.value,
        "scale_current": state.score.scale,
        "scale_raw": state.score.B / state.score.A,
        "scale_initial": state.objective.s0,
        "scale_bounds": [state.objective.lower_scale, state.objective.upper_scale],
        "P_fourfold": float(np.mean(snap.p_o == 4)),
        "Si_fourfold": float(np.mean(snap.si_total == 4)), "wall_s": wall,
    }
    if extra: value.update(extra)
    return value


def _write_outputs(out, state, summary, trace, examples=None):
    out.mkdir(parents=True, exist_ok=True)
    write(out / "final_structure.extxyz", state.atoms, format="extxyz")
    write(out / "final_structure.cif", state.atoms, format="cif")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "trace.json").write_text(json.dumps(trace, indent=2))
    if examples is not None:
        (out / "metropolis_examples.json").write_text(json.dumps(examples, indent=2))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_inline_checkpoint(path, state, counters, energy, initial_energy, step,
                            rngs, model_calls, trace, examples, wall_s, identity):
    metadata = {
        "step": step, "counters": counters, "energy_eV": energy,
        "initial_energy_eV": initial_energy, "model_calls": model_calls,
        "rng_states": [_jsonable(rng.bit_generator.state) for rng in rngs],
        "trace": trace, "examples": examples, "wall_s": wall_s,
        "identity": identity,
    }
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary, positions_A=state.atoms.positions,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    temporary.replace(path)
    (path.parent / "trace_progress.json").write_text(json.dumps(trace, indent=2))


def _restore_inline_checkpoint(path, state, q, exp, mask, config, cutoffs, rngs,
                               backend, expected_identity):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("identity") != expected_identity:
        raise ValueError(
            f"Checkpoint identity mismatch: {metadata.get('identity')} != {expected_identity}"
        )
    atoms = state.atoms.copy(); atoms.positions[:] = archive["positions_A"]
    state = state.rebuilt(atoms, q, exp.r_A[mask], exp.g_r_A2[mask], config, cutoffs)
    for rng, rng_state in zip(rngs, metadata["rng_states"], strict=True):
        rng.bit_generator.state = rng_state
    recomputed_energy = backend.get_energy(state.atoms)
    metadata["resume_energy_drift_eV"] = recomputed_energy - metadata["energy_eV"]
    metadata["energy_eV"] = recomputed_energy
    metadata["model_calls"] += 1
    return state, metadata


def run_alternating(root, config, cutoffs, weights, backend, out, seed=101,
                    target_attempts=10000, accepted_per_block=500, fire_steps=5,
                    watchdog_attempts=20000):
    out.mkdir(parents=True, exist_ok=True)
    state, q, exp, mask = _setup(root, config, cutoffs, weights)
    rngs = np.random.SeedSequence(seed).spawn(3)
    ra, rd, rx = (np.random.default_rng(x) for x in rngs)
    counters = dict(attempted=0, accepted=0, hard_sphere_rejected=0,
                    metropolis_rejected=0)
    energy = backend.get_energy(state.atoms)
    trace = [_summary(state, energy, counters, 0.0, {"stage": "initial", "block": 0})]
    start = time.perf_counter(); block = 0; accepted_in_block = 0; attempts_in_block = 0
    while counters["attempted"] < target_attempts:
        counters["attempted"] += 1; attempts_in_block += 1
        i = int(ra.integers(len(state.atoms)))
        new = state.atoms.positions[i] + rd.uniform(-config.move_max_A, config.move_max_A, 3)
        proposal, meta = state.propose(i, new)
        if proposal is None:
            counters["hard_sphere_rejected"] += 1
        else:
            accept, _ = metropolis_accept(meta["delta_phase25"], float(rx.random()))
            if accept:
                state.commit(proposal); counters["accepted"] += 1; accepted_in_block += 1
            else:
                state.reject(proposal); counters["metropolis_rejected"] += 1
        do_fire = accepted_in_block >= accepted_per_block
        stalled = attempts_in_block >= watchdog_attempts
        if do_fire:
            block += 1
            energy_before = backend.get_energy(state.atoms)
            before = state.atoms.copy()
            fire = run_fixed_cell_fire(before, backend, max_steps=fire_steps, fmax_eV_A=0.05,
                                       maxstep_A=0.05, dt_fs_like=0.05,
                                       logfile=str(out / f"fire_{block:04d}.log"))
            legal = all(state.hard_sphere.check_move(fire.atoms, j, fire.atoms.positions[j])[0]
                        for j in range(len(fire.atoms)))
            if legal and np.all(np.isfinite(fire.atoms.positions)):
                state = state.rebuilt(fire.atoms, q, exp.r_A[mask], exp.g_r_A2[mask], config, cutoffs)
                energy = fire.final_energy_eV; rollback = False
            else:
                state = state.rebuilt(before, q, exp.r_A[mask], exp.g_r_A2[mask], config, cutoffs)
                energy = energy_before; rollback = True
            trace.append(_summary(state, energy, counters, time.perf_counter()-start,
                {"stage": "after_fire", "block": block, "block_attempts": attempts_in_block,
                 "block_accepted": accepted_in_block, "fire_steps": fire.executed_steps,
                 "fire_delta_E_eV": fire.final_energy_eV-energy_before, "fire_rollback": rollback}))
            accepted_in_block = attempts_in_block = 0
        if stalled:
            trace.append({"stage":"watchdog", "block":block+1,
                          "block_attempts":attempts_in_block, "block_accepted":accepted_in_block})
            break
    energy = backend.get_energy(state.atoms)
    summary = _summary(state, energy, counters, time.perf_counter()-start,
                       {"blocks_completed": block, "target_attempts": target_attempts,
                        "accepted_per_block": accepted_per_block, "fire_steps": fire_steps})
    _write_outputs(out, state, summary, trace)
    return summary


def run_inline(root, config, cutoffs, weights, backend, out, seed=101,
               target_attempts=1000, temperature_K=300.0,
               checkpoint_interval=10000, resume=False,
               scale_mode="no_scale"):
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    out.mkdir(parents=True, exist_ok=True)
    state, q, exp, mask = _setup(root, config, cutoffs, weights, scale_mode)
    rngs = tuple(np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(3))
    ra, rd, rx = rngs
    counters = dict(attempted=0, accepted=0, hard_sphere_rejected=0,
                    metropolis_rejected=0)
    energy = backend.get_energy(state.atoms)
    initial_energy = energy; trace=[]; examples={"accepted":None,"rejected":None}
    start_step=0; wall_offset=0.0; model_calls=1
    checkpoint = out / "checkpoint_latest.npz"
    identity = {
        "seed": seed, "temperature_K": temperature_K,
        "move_max_A": config.move_max_A,
        "fit_rmin_A": config.fit_rmin_A,
        "fit_rmax_A": config.fit_rmax_A,
        "scale_mode": state.mode.value,
        "scale_initial": state.objective.s0,
        "scale_bounds": [state.objective.lower_scale, state.objective.upper_scale],
        "model_sha256": backend.manifest()["model_sha256"],
        "atom_count": len(state.atoms),
    }
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint}")
        state, restored = _restore_inline_checkpoint(
            checkpoint, state, q, exp, mask, config, cutoffs, rngs, backend,
            identity,
        )
        counters = restored["counters"]; energy = restored["energy_eV"]
        initial_energy = restored["initial_energy_eV"]
        trace = restored["trace"]; examples = restored["examples"]
        model_calls = restored["model_calls"]; start_step = restored["step"]
        wall_offset = restored["wall_s"]
        if start_step > target_attempts:
            raise ValueError("Checkpoint step exceeds requested target_attempts")
    start=time.perf_counter()
    for step in range(start_step+1, target_attempts+1):
        counters["attempted"] += 1
        i=int(ra.integers(len(state.atoms)))
        new=state.atoms.positions[i]+rd.uniform(-config.move_max_A,config.move_max_A,3)
        proposal,meta=state.propose(i,new)
        if proposal is None:
            counters["hard_sphere_rejected"] += 1; continue
        trial_atoms=state.atoms.copy(); trial_atoms.positions[i]=proposal[0].new_position_A
        trial_energy=backend.get_energy(trial_atoms); model_calls+=1
        dE=trial_energy-energy; dtotal=inline_delta(meta["delta_phase25"],dE,temperature_K)
        u=float(rx.random()); accept,probability=metropolis_accept(dtotal,u)
        record={"step":step,"atom":i,"element":str(state.atoms[i].symbol),
                "delta_phase25":meta["delta_phase25"],"delta_E_eV":dE,
                "delta_E_over_kBT":dE/(8.617333262145e-5*temperature_K),
                "delta_total":dtotal,"probability":probability,"random_u":u,"accepted":accept}
        if accept:
            state.commit(proposal); energy=trial_energy; counters["accepted"]+=1
            if examples["accepted"] is None: examples["accepted"]=record
        else:
            state.reject(proposal); counters["metropolis_rejected"]+=1
            if examples["rejected"] is None: examples["rejected"]=record
        if step % 100 == 0 or step == target_attempts:
            pdf_drift = state.pdf.validation_errors()
            cn_drift = state.cn.validation_errors()
            if (pdf_drift["max_abs_hist_count"] or pdf_drift["max_abs_G_A-2"] > 1e-8
                    or cn_drift["max_abs_integer_error"]
                    or cn_drift["max_abs_component_error"] > 1e-12):
                raise RuntimeError(f"incremental state drift at step {step}: {pdf_drift}, {cn_drift}")
            trace.append(_summary(state,energy,counters,
                                  wall_offset + time.perf_counter()-start,
                                  {"step":step,"temperature_K":temperature_K,
                                   "model_calls":model_calls,"pdf_drift":pdf_drift,
                                   "cn_drift":cn_drift}))
        if step % checkpoint_interval == 0 or step == target_attempts:
            _save_inline_checkpoint(
                checkpoint, state, counters, energy, initial_energy, step, rngs,
                model_calls, trace, examples,
                wall_offset + time.perf_counter() - start,
                identity,
            )
    check_energy=backend.get_energy(state.atoms); model_calls+=1
    drift=check_energy-energy; energy=check_energy
    wall_total = wall_offset + time.perf_counter()-start
    summary=_summary(state,energy,counters,wall_total,
                     {"target_attempts":target_attempts,"temperature_K":temperature_K,
                      "model_calls":model_calls,"energy_drift_eV":drift,
                      "delta_energy_from_MQ_eV":energy-initial_energy,
                      "resumed_from_step":start_step,
                      "checkpoint_interval":checkpoint_interval,
                      "fit_rmin_A":config.fit_rmin_A,
                      "fit_rmax_A":config.fit_rmax_A})
    _write_outputs(out,state,summary,trace,examples)
    _save_inline_checkpoint(
        checkpoint, state, counters, energy, initial_energy, target_attempts,
        rngs, model_calls, trace, examples, wall_total, identity,
    )
    return summary

