"""Build the README results tables from ``docs/results/lessonNN.json``.

The README contains marker pairs  <!-- results:NN -->  ...  <!-- /results:NN -->;
``update_readme()`` replaces what is between them with tables generated from the
latest real run, so the numbers in the README can never drift from the code.
"""
from __future__ import annotations

import re

from .common import DOCS, ROOT, load_results, md_table


def _pct(x):
    return f"{100 * x:.1f}%"


def _sci(x):
    return f"{x:.1e}"


def table_01(r):
    return md_table(["quantity", "value"], [
        ["perceptron mistakes before convergence", r["perceptron_mistakes"]],
        ["Novikoff bound (R/γ)² with R, γ", f"{r['novikoff_bound']:.0f}  (R = {r['novikoff_R']:.2f}, γ = {r['novikoff_gamma']:.3f})"],
        ["hand gradient vs finite differences (rel. error)", _sci(r["grad_check_rel_err"])],
        ["GD vs L-BFGS optimum: max parameter difference", _sci(r["gd_param_err"])],
        ["logistic neuron test accuracy", _pct(r["logistic_test_acc"])],
        ["majority-class baseline", _pct(r["majority_baseline"])],
        ["single neuron on XOR (train accuracy)", _pct(r["xor_acc"])],
    ])


def table_02(r):
    g = r["gradcheck"]
    t1 = md_table(["network 2-8-8-3", "max rel. error, parameters", "max rel. error, inputs", "< 1e-6 ?"],
                  [[k, _sci(v["params"]), _sci(v["inputs"]), "PASS" if max(v.values()) < 1e-6 else "FAIL"]
                   for k, v in g.items()])
    t2 = md_table(["model (spirals, 3 classes)", "test accuracy"], [
        ["majority class", _pct(r["majority_test_acc"])],
        ["softmax regression (no hidden layer)", _pct(r["linear_test_acc"])],
        ["MLP 2-64-64-3, ReLU, Adam", _pct(r["mlp_test_acc"])],
        ["XOR with 4 hidden tanh units (train)", _pct(r["xor_acc_4_hidden"])],
    ])
    return t1 + "\n\n" + t2


def table_03(r):
    out = []
    out.append("**Activations** (6 hidden layers, Adam 3e-3, 150 epochs)\n\n" + md_table(
        ["activation", "final train loss", "train accuracy"],
        [[k, f"{v['final_loss']:.2e}", _pct(v["train_acc"])] for k, v in r["activations"].items()]))
    o = r["optimisers"]
    out.append("**Optimisers**\n\n" + md_table(
        ["run", "final value"],
        [[f"Rosenbrock, {k}", _sci(v)] if not k.startswith("mlp") else [f"spiral MLP loss, {k[4:]}", f"{v:.4f}"]
         for k, v in o.items()]))
    out.append("**LR schedules** (SGD + momentum, 100 epochs, mean of 3 seeds)\n\n" + md_table(
        ["schedule", "final train loss", "test accuracy"],
        [[k, f"{v['final_train_loss']:.4f}", _pct(v["test_acc"])] for k, v in r["schedules"].items()]))
    out.append("**Batch norm** (10 sigmoid hidden layers, SGD + momentum)\n\n" + md_table(
        ["network", "final train loss", "train accuracy"],
        [[k, f"{v['final_loss']:.3f}", _pct(v["train_acc"])] for k, v in r["batchnorm"].items()]))
    out.append("**Overfitting** (150 points, 20 % label noise, 2-256-256-3, 400 epochs)\n\n" + md_table(
        ["regularisation", "train acc", "val acc (final)", "val loss (final)", "early stopping: epoch / val acc"],
        [[k, _pct(v["train_acc"]), _pct(v["val_acc_final"]), f"{v['val_loss_final']:.3f}",
          f"{v['best_epoch']} / {_pct(v['val_acc_early_stop'])}"] for k, v in r["overfitting"].items()]))
    return "\n\n".join(out)


def table_04(r):
    e = r["equivalence"]
    return md_table(["check", "value"], [
        ["autograd vs hand derivative of x²y + sin(xy)", _sci(r["autograd_vs_hand_err"])],
        ["custom `MySoftplus` passes `torch.autograd.gradcheck`", r["custom_op_gradcheck"]],
        ["nested autograd: tanh' and tanh'' vs exact formulas", f"{_sci(r['nested_grad_err_d1'])}, {_sci(r['nested_grad_err_d2'])}"],
        ["NumPy vs PyTorch gradients at step 0 (float64)", _sci(e["lr0.1"]["grad_diff_step0"])],
        ["max loss difference over 300 GD steps, lr 0.1 (stable)", _sci(e["lr0.1"]["max_loss_diff"])],
        ["max loss difference over 300 GD steps, lr 0.5 (unstable, chaotic)", _sci(e["lr0.5"]["max_loss_diff"])],
        ["PyTorch MLP on spirals: test accuracy", _pct(r["torch_test_acc"])],
    ])


def table_05(r):
    rows = []
    for ds in ("MNIST", "FashionMNIST"):
        d = r[ds]
        rows.append([ds, "majority class", "-", _pct(d["majority"]), "-"])
        for m in ("logistic regression", "MLP 784-96-10", "CNN"):
            rows.append([ds, m, d[m]["params"], _pct(d[m]["test_acc"]), f"{d[m]['train_time_s']:.0f} s"])
    return md_table(["data set", "model", "parameters", "test accuracy", "CPU training time"], rows) + \
        f"\n\nConvolution implementations agree: loops vs `F.conv2d` {_sci(r['conv_loops_vs_torch'])}, " \
        f"im2col vs `F.conv2d` {_sci(r['conv_im2col_vs_torch'])}."


def table_06(r):
    rows = []
    for m, v in r["one_step"].items():
        ro = r["rollout"][m]
        rows.append([m, f"{v['rmse_vs_noisy']:.4f}", f"{v['rmse_vs_clean']:.4f}", f"{ro['rmse_h10']:.3f}",
                     f"{ro['rmse_h50']:.3f}", f"{ro['rmse_h150']:.3f}"])
    return md_table(["model", "1-step RMSE vs noisy obs.", "1-step RMSE vs true state",
                     "closed loop, 10 steps", "50 steps", "150 steps"], rows) + \
        "\n\nThe noise floor is 0.100: even a perfect model scores 0.100 against the noisy observations."


def table_07(r):
    c = r["compression"]
    t1 = md_table(["k", "PCA optimum (train)", "linear AE (train)", "gap", "max principal angle",
                   "PCA (test)", "MLP AE (test)", "conv AE (test)"],
                  [[k, f"{v['pca_train_optimum']:.5f}", f"{v['linear_ae_train']:.5f}",
                    f"{100 * (v['linear_ae_train'] / v['pca_train_optimum'] - 1):.1f}%",
                    f"{v['max_principal_angle_deg']:.1f}°", f"{v['pca_test']:.5f}", f"{v['mlp_ae_test']:.5f}",
                    f"{v['conv_ae_test']:.5f}"] for k, v in c.items()])
    d = r["denoising"]
    t2 = md_table(["denoiser (σ = 0.5)", "PSNR"], [
        ["noisy input", f"{d['psnr_noisy']:.2f} dB"],
        [f"PCA projection (best k = {d['pca_best_k']})", f"{d['psnr_pca_best']:.2f} dB"],
        ["convolutional denoising autoencoder", f"{d['psnr_conv_dae']:.2f} dB"]])
    t3 = md_table(["2-D code", "5-NN digit accuracy"], [["PCA", _pct(r["knn5_acc_pca2"])],
                                                       ["MLP autoencoder", _pct(r["knn5_acc_ae2"])]])
    return "\n\n".join([t1, t2, t3])


def table_08(r):
    b = r["baselines"]
    return md_table(["model", "validation cross-entropy (nats/char)"], [
        ["uniform over the 65 characters", f"{b['uniform']:.3f}"],
        ["unigram (character frequencies)", f"{b['unigram']:.3f}"],
        ["bigram (add-one smoothing)", f"{b['bigram']:.3f}"],
        [f"TinyGPT, {r['n_params'] / 1e6:.2f} M params, {r['steps']} steps ({r['train_time_s'] / 60:.1f} min CPU)",
         f"**{r['final_val']:.3f}** (train {r['final_train']:.3f})"],
    ]) + f"\n\nFrom-scratch attention vs PyTorch: {_sci(r['attn_numpy_vs_torch'])} (NumPy, float64), " \
         f"{_sci(r['mha_scratch_vs_torch'])} (multi-head module, float32)."


def table_09(r):
    t1 = md_table(["FD grid N", "rel. L2 error at t = 1 vs Cole-Hopf", "CPU time"],
                  [[n, _sci(e), f"{t:.2f} s"] for n, e, t in zip(r["fd_N"], r["fd_err"], r["fd_time"])])
    t1 += f"\n\nObserved order of accuracy: **{r['fd_order']:.2f}** (second-order scheme)."
    t2 = md_table(["quantity", "value"], [
        ["PINN rel. L2 error, whole space-time grid", f"**{_sci(r['pinn_rel_l2'])}**"],
        ["PINN max abs error", _sci(r["pinn_max_abs"])],
        ["PINN rel. L2 error at t = 0.25 / 0.5 / 0.75 / 1", " / ".join(_sci(e) for e in r["pinn_rel_l2_t"])],
        ["PINN parameters / training time (CPU)", f"{r['pinn_params']} / {r['pinn_time_s'] / 60:.1f} min"],
        ["Raissi et al. (2019), 9x20 net, 10 000 points (published)", "6.7e-04"],
    ])
    t3 = md_table(["model (test set, rel. L2 error)", "median", "95th percentile"], [
        ["mean solution (trivial baseline)", _sci(r["rel_err_mean_median"]), "-"],
        ["nearest training run", _sci(r["rel_err_nn_median"]), _sci(r["rel_err_nn_p95"])],
        ["piecewise-linear interpolation (Delaunay in parameter space)", _sci(r["rel_err_lin_median"]), _sci(r["rel_err_lin_p95"])],
        ["neural surrogate (MLP 2 -> 128 grid values)", _sci(r["rel_err_surrogate_median"]), _sci(r["rel_err_surrogate_p95"])],
    ]) + (f"\n\nSolver: {r['solver_ms']:.0f} ms per solution (vectorised over parameters); its own error at the "
          f"parameter-space corners is {_sci(r['solver_check'])}. Surrogate: {r['surrogate_ms'] * 1000:.1f} µs per "
          f"solution after {r['surrogate_train_s']:.0f} s of training on {r['n_train']} solver runs.")
    return "**Reference solver**\n\n" + t1 + "\n\n**PINN** (ν = 0.01/π)\n\n" + t2 + "\n\n**Surrogate**\n\n" + t3


def table_10(r):
    v = r["vae"]
    t1 = md_table(["model (binarised MNIST, test set)", "−ELBO (nats)", "reconstruction + KL", "IWAE NLL, K = 100 (nats)"], [
        ["independent Bernoulli pixels (exact NLL, trivial baseline)", "-", "-", f"{v['baseline_independent_pixels_nll']:.1f}"],
        ["VAE, latent 2", f"{v['k2']['neg_elbo']:.1f}", f"{v['k2']['rec']:.1f} + {v['k2']['kl']:.1f}", f"{v['k2']['iwae_nll']:.1f}"],
        ["VAE, latent 16", f"{v['k16']['neg_elbo']:.1f}", f"{v['k16']['rec']:.1f} + {v['k16']['kl']:.1f}", f"{v['k16']['iwae_nll']:.1f}"],
    ])
    d = r["diffusion"]
    t2 = md_table(["sampler (4000 samples)", "MMD² vs data", "within 3σ of a mode", "modes covered"],
                  [[k, _sci(x["mmd"]), _pct(x["frac_near_mode"]), f"{x['modes_hit']}/8"] for k, x in d["samples"].items()])
    e = d["eps_rel_err"]
    t3 = "Relative RMS error of the learnt noise predictor vs the exact $\\epsilon^*$: " + \
        ", ".join(f"t={k}: {float(x):.3f}" for k, x in e.items()) + "."
    return t1 + "\n\n" + t2 + "\n\n" + t3


def table_11(r):
    t1 = md_table(["15x15 SOM on MNIST (test digits)", "SOM", "k-means (225 centres)", "random data points as prototypes"], [
        ["quantisation error", f"{r['som_qe']:.3f}", f"{r['kmeans_qe']:.3f}", f"{r['random_prototypes_qe']:.3f}"],
        ["topographic error", f"{r['som_te']:.3f}", "n/a (no grid)", f"{r['random_prototypes_te']:.3f}"],
        ["digit accuracy, unit majority label", _pct(r["som_label_acc"]), _pct(r["kmeans_label_acc"]),
         f"majority class: {_pct(r['majority_acc'])}"],
    ])
    t2 = md_table(["Hopfield rule (8 letters, 20 % flipped pixels)", "correct pixels after recall, per letter"],
                  [[k, " ".join(f"{x:.2f}" for x in v)] for k, v in r["glyph_recall"].items()])
    t3 = md_table(["P/N (N = 200 random patterns)"] + [f"{p / 200:.2f}" for p in r["capacity_loads"]],
                  [["classic (Hebbian) recalled"] + [f"{x:.2f}" for x in r["capacity_classic"]],
                   ["modern (attention) recalled"] + [f"{x:.2f}" for x in r["capacity_modern"]]])
    return "\n\n".join([t1, t2, t3])


def table_12(r):
    rows = [[name] + [f"{e:.3f}" for e in v] for name, v in r["ber"].items()]
    t = md_table(["bit error rate at length L"] + [f"L={L}" + (" (train)" if L <= 10 else "") for L in r["lengths"]], rows)
    return t + (f"\n\nParameters: DNC {r['dnc_params']:,} (LSTM controller 64 + 32 × 16 memory), LSTM baseline "
                f"{r['lstm_params']:,} (2 × 128). Training: 6000 steps; DNC {r['dnc_time_s'] / 60:.1f} min, "
                f"LSTM {r['lstm_time_s'] / 60:.1f} min on CPU. Chance level is 0.5.")


def table_14(r):
    t = md_table(["operator (test set, 200 new u0)", "mean relative L2 error"], [
        ["identity  u(T) = u0", f"{r['identity']:.3f}"],
        ["best linear operator (least squares)", f"{r['best linear operator']:.3f}"],
        ["DeepONet", f"{r['DeepONet']:.4f}"],
        ["FNO", f"**{r['FNO']:.4f}**"],
        ["FNO evaluated on the 512-point grid (trained on 128)", f"{r['FNO on 512 grid (trained on 128)']:.4f}"],
        ["DeepONet queried at 512 points", f"{r['DeepONet queried on 512 grid']:.4f}"],
    ])
    return t + (f"\n\nCost per sample: spectral solver {r['solver_ms_per_sample']:.0f} ms (batched), FNO "
                f"{r['fno_ms_per_sample']:.2f} ms. Training on CPU: DeepONet {r['train_s']['DeepONet']:.0f} s, "
                f"FNO {r['train_s']['FNO']:.0f} s. The solver's own resolution check (2× finer grid): "
                f"{_sci(r['solver_resolution_check'])}.")


def table_15(r):
    h = r["hnn"]
    rows = [[k, " / ".join(f"{d:.3f}" for d in v["energy_drift_rel"]),
             " / ".join(f"{e:.3f}" for e in v["traj_err"]) if "traj_err" in v else "0 (reference)"] for k, v in h.items()]
    t1 = md_table(["pendulum, 100 time units, 3 unseen initial states", "relative change of the true energy",
                   "mean state error vs exact"], rows)
    n = r["neural_ode"]
    t2 = md_table(["Lotka-Volterra model", "log-RMSE on [0, 30] (data window)", "log-RMSE on (30, 60] (extrapolation)",
                   "max change of the LV invariant"],
                  [[k, f"{v['log_rmse_train_window']:.3f}", f"{v['log_rmse_extrapolation']:.3f}",
                    f"{v['max_invariant_change']:.3f}"] for k, v in n.items() if isinstance(v, dict)])
    return t1 + "\n\n" + t2


def table_solvers(r):
    f, sp, fem, lbm = r["fd_burgers"], r["spectral_burgers_nu0.02"], r["p1_fem_poisson"], r["lbm_poiseuille"]
    return md_table(["solver", "resolutions", "errors", "observed order"], [
        ["FD Burgers, rel. L2 vs Cole-Hopf (t = 1, ν = 0.01/π)", " / ".join(map(str, f["N"])),
         " / ".join(_sci(e) for e in f["rel_l2"]), f"{f['order']:.2f}"],
        ["spectral Burgers, max error vs Cole-Hopf (ν = 0.02)", " / ".join(map(str, sp["N"])),
         " / ".join(_sci(e) for e in sp["max_err"]), "exponential"],
        ["P1 FEM Poisson, rel. L2 (manufactured)", " / ".join(f"{n}²" for n in fem["n"]),
         " / ".join(_sci(e) for e in fem["rel_l2"]), f"{fem['order']:.2f}"],
        ["LBM Poiseuille, max rel. error", " / ".join(map(str, lbm["ny"])),
         " / ".join(_sci(e) for e in lbm["max_rel_err"]), f"{lbm['order']:.2f}"],
    ])


TABLES = {1: table_01, 2: table_02, 3: table_03, 4: table_04, 5: table_05, 6: table_06,
          7: table_07, 8: table_08, 9: table_09, 10: table_10, 11: table_11, 12: table_12, 14: table_14, 15: table_15}


def _load_json(name):
    import json

    return json.loads((DOCS / "results" / f"{name}.json").read_text())


def update_readme(path=ROOT / "README.md") -> None:
    text = path.read_text()
    items = list(TABLES.items()) + [("solvers", table_solvers)]
    for n, fn in items:
        try:
            table = fn(load_results(n) if n != "solvers" else _load_json("solvers"))
        except (FileNotFoundError, KeyError):  # lesson not (re-)run with the current code yet
            continue
        tag = f"{n:02d}" if isinstance(n, int) else n
        pat = re.compile(rf"(<!-- results:{tag} -->)(.*?)(<!-- /results:{tag} -->)", re.S)
        text = pat.sub(lambda m: f"{m.group(1)}\n{table}\n{m.group(3)}", text)
    path.write_text(text)
    (DOCS / "results").mkdir(exist_ok=True)
