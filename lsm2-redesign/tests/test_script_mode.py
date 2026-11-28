"""Our code in the script's settings against (d)'s own Brian2 script on the same realised liquid, the same
stimulus and the same initial voltages: the spike trains must be identical (a deterministic version of 1e)."""
import os
import numpy as np
import pytest
import torch

from lsm2 import config as C, liquid as L, sim as S, brian2_ref

SCRIPT = os.environ.get("MAASS_SCRIPT", "/home/claude/ext/brian2ex/maass2002.py")


@pytest.mark.skipif(not os.path.exists(SCRIPT), reason="(d)'s script not available")
def test_script_mode_reproduces_d_exactly():
    mod = brian2_ref.script_module(SCRIPT)
    mod.prefs.codegen.target = "numpy"
    np.random.seed(5)
    import brian2; brian2.start_scope()
    neurons = mod.get_neurons()
    N_exc = int(0.8 * len(neurons))
    exc, inh = neurons[:N_exc], neurons[N_exc:]
    exc.tau_refrac = 3 * mod.ms; exc.tau_stimulus = 3 * mod.ms
    inh.tau_refrac = 2 * mod.ms; inh.tau_stimulus = 6 * mod.ms
    s, nA, ms = mod.second, mod.nA, mod.ms
    specs = [("ee_synapses", exc, exc, 0.3, 3, 30, 0.5, 1.1, 0.05, 1.5, 0), ("ei_synapses", exc, inh, 0.2, 3, 60, 0.05, 0.125, 1.2, 0.8, 1),
             ("ie_synapses", inh, exc, 0.4, 6, -19, 0.25, 0.7, 0.02, 0.8, 2), ("ii_synapses", inh, inh, 0.1, 6, -19, 0.32, 0.144, 0.06, 0.8, 3)]
    syns = [mod.get_synapses(nm, a, b, C=c, l=2, tau_I=ti * ms, A=A * nA, U=U, D=D * s, F=F * s, delay=dl * ms)
            for nm, a, b, c, ti, A, U, D, F, dl, _ in specs]
    stim = mod.SpikeGeneratorGroup(1, [], [] * ms, name="stimulus")
    se = mod.Synapses(stim, exc, "A : ampere (shared, constant)", on_pre="I_stimulus += A"); se.connect(p=1); se.A = 18 * nA
    si = mod.Synapses(stim, inh, "A : ampere (shared, constant)", on_pre="I_stimulus += A"); si.connect(p=1); si.A = 9 * nA
    me = mod.SpikeMonitor(exc, name="spike_monitor_exc"); mi = mod.SpikeMonitor(inh, name="spike_monitor_inh")
    mod.defaultclock.dt = mod.DT
    net = mod.Network([neurons, *syns, se, si, stim, me, mi])
    v0 = np.random.uniform(13.5, 15, len(neurons))
    neurons.v = v0 * mod.mV
    spikes = np.sort(np.random.uniform(0, 200, 6))
    spikes = np.round(spikes, 1)
    stim.set_spikes([0] * len(spikes), spikes * ms)
    T = 2000                                   # 200 ms at 0.1 ms
    net.run(T * 0.1 * ms)
    b_r = np.zeros((T, len(neurons)), bool)
    for mon, off in ((me, 0), (mi, N_exc)):
        k = np.round(np.asarray(mon.t / ms) / 0.1).astype(int)
        b_r[k, np.asarray(mon.i) + off] = True
    # our liquid with the same realised parameters
    liq = L.build_maass_script(1, n_sub=1, step_ms=0.1)
    pos = np.stack([np.asarray(neurons.x_pos), np.asarray(neurons.y_pos), np.asarray(neurons.z_pos)], 1)
    pre, post, typ, w, U, D, F = [], [], [], [], [], [], []
    for (nm, a, b, *_, ti_), syn in zip(specs, syns):
        oi = 0 if a is exc else N_exc
        oj = 0 if b is exc else N_exc
        pre.append(np.asarray(syn.i) + oi); post.append(np.asarray(syn.j) + oj); typ.append(np.full(len(syn), ti_))
        w.append(np.asarray(syn.A / nA) / 15.0); U.append(np.asarray(syn.U))
        D.append(np.asarray(syn.D / ms)); F.append(np.asarray(syn.F / ms))
    liq.pos = pos
    liq.pre, liq.post, liq.typ = (np.concatenate(v) for v in (pre, post, typ))
    liq.w, liq.U, liq.D_ms, liq.F_ms = (np.concatenate(v) for v in (w, U, D, F))
    x = np.zeros((1, T, 1), np.uint8)
    x[0, np.round(spikes / 0.1).astype(int), 0] = 1
    sm = S.Simulator(liq, dtype=torch.float64, g=1.0, in_scale=1.0)
    o = sm.run(x, (v0 / 15.0)[None], T, rec={"raster": True})
    r = o["raster"][:, 0, :].numpy()
    assert b_r.sum() > 50
    assert (r != b_r).sum() == 0, f"{(r != b_r).sum()} spike mismatches of {b_r.sum()}"
