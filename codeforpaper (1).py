"""
Driver/Sensor/Dual/Ordinary node roles, maximum matching tie-breaking robustness,
and Lin's structural controllability tests (input-connection + augmented matching)
on the IEEE 118-bus power grid across a 24-hour dispatch cycle.

Methodological notes:
- Node indices match the 0-based pandapower bus indices.
- Randomization performs vertex-ordering tie-breaking (permutation of bipartite vertex
  labels with explicit random seeds), not uniform sampling over all maximum matchings.
- Three matching procedures:
    REF = identity permutation (deterministic baseline, independent of PYTHONHASHSEED)
    A   = one random permutation per run, kept FIXED across all 24 hours
    B   = independent random permutations resampled at EVERY hour
- GRAPH_MODE:
    "full"  = transmission lines + in-line transformers; parallel circuits collapsed onto
              net active power flow per bus pair; zero flow defaults to from_bus -> to_bus.
- TIEBREAK:
    "left"  = permutes left-hand bipartite vertex orderings only.
    "both"  = permutes both left-hand and right-hand bipartite vertex labels and sorts edges.
- Output files are saved in a dedicated folder for each (GRAPH_MODE, TIEBREAK) configuration.
"""

import os
import re
import random
import warnings
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import pandapower as pp
from networkx.algorithms import bipartite

# ================================================================
# 0. PARAMETERS
# ================================================================
SAMPLES_DIR = "data/samples"  # Set the directory containing the hourly JSON snapshots
DAY = (2024, 1, 1)            # Target date; N_HOURS is inferred from available snapshots
N_RUNS = 200                  # Number of Monte Carlo runs per variant
SEED_BASE = 1000              # Seed for run k is SEED_BASE + k
GRAPH_MODE = "full"           # "full" (lines + transformers) or "lines" (lines only)
TIEBREAK = "both"             # "both" (two-sided label permutation) or "left"
assert GRAPH_MODE in ("lines", "full") and TIEBREAK in ("left", "both")

OUTPUT_DIR = os.path.join(SAMPLES_DIR, f"monte_carlo_results_{GRAPH_MODE}_{TIEBREAK}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INCLUDE_SGEN = False          # Whether to include static generators as physical inputs
WATCH_NODE = 103              # Representative dynamic bus inspected hour-by-hour
TOL = 1e-9                    # Threshold below which active power flow is treated as zero

ROLES = ["Driver", "Sensor", "Dual", "Ordinary"]
DRV, SNS, DUA, ORD = 0, 1, 2, 3


# ================================================================
# 1. DATA LOADING (CHRONOLOGICAL ORDER)
# ================================================================
def file_time(name):
    m = re.match(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.json$", name)
    return datetime(*map(int, m.groups())) if m else None


if not os.path.isdir(SAMPLES_DIR):
    raise FileNotFoundError(
        f"Directory '{SAMPLES_DIR}' not found. Please place the hourly JSON files there "
        "or update SAMPLES_DIR."
    )

timed = sorted((file_time(f), f) for f in os.listdir(SAMPLES_DIR) if file_time(f))
day_files = [(t, f) for t, f in timed if (t.year, t.month, t.day) == DAY]

if len(day_files) < 2:
    raise ValueError(f"Insufficient snapshots found for {DAY}: {len(day_files)}")

json_files = [f for _, f in day_files]
HOUR_LABELS = [t.hour for t, _ in day_files]
N_HOURS = len(json_files)
missing_hours = sorted(set(range(24)) - set(HOUR_LABELS))
print(f"Date {DAY}: {N_HOURS} snapshots | missing hours: "
      f"{missing_hours if missing_hours else 'none'}")
print("Hours loaded:", HOUR_LABELS)


def injection_buses(net):
    """Return in-service physical generation/slack injection buses."""
    tables = ["gen", "ext_grid"] + (["sgen"] if INCLUDE_SGEN else [])
    buses = set()
    for t in tables:
        df = net[t]
        if len(df) == 0:
            continue
        if "in_service" in df.columns:
            df = df[df["in_service"]]
        buses.update(int(b) for b in df["bus"])
    return buses


def build_edges(net, mode=None):
    """Build directed edges oriented according to active power flow."""
    mode = mode or GRAPH_MODE
    if mode == "lines":
        edges = {}
        for idx, r in net.line.iterrows():
            if "in_service" in net.line.columns and not r["in_service"]:
                continue
            f, t = int(r["from_bus"]), int(r["to_bus"])
            e = (f, t) if float(net.res_line.loc[idx, "p_from_mw"]) >= 0 else (t, f)
            if e[0] != e[1]:
                edges[e] = None
        return list(edges)

    pairs = {}  # (min_bus, max_bus) -> [default_from, default_to, net_active_flow]

    def add(f, t, p):
        if f == t:
            return
        key = (min(f, t), max(f, t))
        if key not in pairs:
            pairs[key] = [f, t, p]
        elif (f, t) == (pairs[key][0], pairs[key][1]):
            pairs[key][2] += p
        else:
            pairs[key][2] -= p

    for idx, r in net.line.iterrows():
        if "in_service" in net.line.columns and not r["in_service"]:
            continue
        add(int(r["from_bus"]), int(r["to_bus"]),
            float(net.res_line.loc[idx, "p_from_mw"]))

    if len(net.trafo) > 0:
        for idx, r in net.trafo.iterrows():
            if "in_service" in net.trafo.columns and not r["in_service"]:
                continue
            add(int(r["hv_bus"]), int(r["lv_bus"]),
                float(net.res_trafo.loc[idx, "p_hv_mw"]))

    edges = []
    for f, t, p in pairs.values():
        edges.append((t, f) if p < -TOL else (f, t))
    return edges


hours = []
for h, fname in enumerate(json_files):
    net = pp.from_json(os.path.join(SAMPLES_DIR, fname))
    nodes = sorted(int(b) for b in net.bus.index)
    idx = {b: i for i, b in enumerate(nodes)}
    edges_bus = build_edges(net)
    edges_by_mode = {m: build_edges(net, m) for m in ("lines", "full")}
    if h == 0:
        oos = int((~net.line["in_service"]).sum()) if "in_service" in net.line.columns else 0
        print(f"Hour 0: {len(net.line)} lines ({oos} out of service), {len(net.trafo)} transformers, "
              f"{len(edges_bus)} directed edges | graph mode: {GRAPH_MODE} | tie-breaking: {TIEBREAK}")
        print("Edge count at hour 0 by mode:", {m: len(e) for m, e in edges_by_mode.items()})
    src = np.array([idx[a] for a, _ in edges_bus])
    dst = np.array([idx[b] for _, b in edges_bus])

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges_bus)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hub, auth = nx.hits(G, max_iter=1000, normalized=True)

    gen_b = injection_buses(net)
    hours.append({
        "hour": h, "nodes": nodes, "idx": idx,
        "edges_bus": edges_bus, "edges_by_mode": edges_by_mode, "src": src, "dst": dst,
        "in_deg": np.array([G.in_degree(b) for b in nodes]),
        "out_deg": np.array([G.out_degree(b) for b in nodes]),
        "hub": np.array([hub[b] for b in nodes]),
        "auth": np.array([auth[b] for b in nodes]),
        "gen_buses": sorted(gen_b),
        "is_gen": np.array([b in gen_b for b in nodes]),
        "G": G,
    })

NODES = hours[0]["nodes"]
N = len(NODES)
for d in hours:
    if d["nodes"] != NODES:
        raise ValueError(f"Bus set mismatch at hour {d['hour']}.")

print(f"Snapshots: {N_HOURS} | buses: {N} | runs per variant: {N_RUNS}")
print("Directed edges per hour (min/max):",
      min(len(d["edges_bus"]) for d in hours),
      max(len(d["edges_bus"]) for d in hours),
      "| physical injection buses (hour 0):", len(hours[0]["gen_buses"]))
_e0 = set(hours[0]["edges_bus"])
print("Bidirectional bus pairs (hour 0):",
      sum(1 for (a, b) in _e0 if (b, a) in _e0) // 2)
if WATCH_NODE not in NODES:
    raise ValueError("WATCH_NODE not found among network buses.")
WATCH_I = NODES.index(WATCH_NODE)


# ================================================================
# 2. MAXIMUM BIPARTITE MATCHING & NODE ROLE CLASSIFICATION
# ================================================================
def matching_roles(d, perm_L, perm_R):
    """
    Compute maximum bipartite matching via Hopcroft-Karp under vertex permutations
    perm_L and perm_R, and classify each bus into Driver, Sensor, Dual, or Ordinary.
    """
    edges = [(int(perm_L[s]), N + int(perm_R[t]))
             for s, t in zip(d["src"], d["dst"])]
    if TIEBREAK == "both":
        edges.sort()
    H = nx.Graph()
    H.add_nodes_from(range(N), bipartite=0)
    H.add_nodes_from(range(N, 2 * N), bipartite=1)
    H.add_edges_from(edges)
    m = bipartite.hopcroft_karp_matching(H, top_nodes=range(N))

    matched_L = np.array([perm_L[i] in m for i in range(N)])
    matched_R = np.array([(N + perm_R[i]) in m for i in range(N)])
    size = int(matched_L.sum())

    # Consistency assertions
    assert size == int(matched_R.sum()), "Left/Right matching cardinality mismatch"
    assert all(m[m[u]] == u for u in m), "Asymmetric matching dictionary"

    driver, sensor = ~matched_R, ~matched_L
    roles = np.full(N, ORD)
    roles[driver] = DRV
    roles[sensor] = SNS
    roles[driver & sensor] = DUA

    # Identity check: n - |M| = #Driver + #Dual = #Sensor + #Dual
    assert (roles == DRV).sum() + (roles == DUA).sum() == N - size
    assert (roles == SNS).sum() + (roles == DUA).sum() == N - size
    return roles, size


IDENT = list(range(N))


def draw_perms(rng):
    pL = rng.sample(IDENT, N)
    pR = rng.sample(IDENT, N) if TIEBREAK == "both" else IDENT
    return pL, pR


def one_run(variant, seed=None):
    """Execute one 24-hour run for variant 'REF', 'A', or 'B'."""
    roles = np.zeros((N_HOURS, N), dtype=int)
    sizes = np.zeros(N_HOURS, dtype=int)
    rng = random.Random(seed) if seed is not None else None
    if variant == "A":
        pL, pR = draw_perms(rng)
    for d in hours:
        if variant == "REF":
            pL, pR = IDENT, IDENT
        elif variant == "B":
            pL, pR = draw_perms(rng)
        roles[d["hour"]], sizes[d["hour"]] = matching_roles(d, pL, pR)
    return roles, sizes


# ================================================================
# 3. RUN-LEVEL SUMMARY METRICS
# ================================================================
def summarize(roles):
    row = {}
    # Network-level role persistence across all 24 hours
    persistent = (roles == roles[0]).all(axis=0)
    row["persistent_nodes"] = int(persistent.sum())
    row["persistence_pct"] = 100 * persistent.mean()

    # Heuristic generator coverage (share of Driver/Dual node-hours on generator buses)
    dd = (roles == DRV) | (roles == DUA)
    gen = np.array([d["is_gen"] for d in hours])
    row["generator_coverage_pct"] = 100 * (dd & gen).sum() / dd.sum()

    in_d = np.array([d["in_deg"] for d in hours])
    out_d = np.array([d["out_deg"] for d in hours])
    hub = np.array([d["hub"] for d in hours])
    auth = np.array([d["auth"] for d in hours])
    tot_d = in_d + out_d

    for r, name in enumerate(ROLES):
        mk = roles == r
        n_obs = int(mk.sum())
        row[f"{name}_obs"] = n_obs
        row[f"{name}_share_pct"] = 100 * n_obs / roles.size
        if n_obs == 0:
            continue
        for lab, arr in (("in", in_d), ("out", out_d), ("tot", tot_d)):
            v = arr[mk].astype(float)
            mu = v.mean()
            sd = v.std(ddof=1) if n_obs > 1 else 0.0
            row[f"{name}_{lab}_mean"] = mu
            row[f"{name}_{lab}_std"] = sd
            row[f"{name}_{lab}_cv"] = sd / mu if mu > 0 else np.nan
        row[f"{name}_hub_mean"] = hub[mk].mean()
        row[f"{name}_auth_mean"] = auth[mk].mean()
        row[f"{name}_in_gt_out_pct"] = 100 * (in_d[mk] > out_d[mk]).mean()
        row[f"{name}_out_gt_in_pct"] = 100 * (out_d[mk] > in_d[mk]).mean()
        row[f"{name}_eq_pct"] = 100 * (in_d[mk] == out_d[mk]).mean()
        if n_obs > 2 and in_d[mk].std() > 0 and out_d[mk].std() > 0:
            row[f"{name}_corr_in_out"] = np.corrcoef(in_d[mk], out_d[mk])[0, 1]
        else:
            row[f"{name}_corr_in_out"] = np.nan

    # Role-to-role directed edge adjacency matrix (%)
    E = np.zeros((4, 4))
    for d in hours:
        np.add.at(E, (roles[d["hour"]][d["src"]], roles[d["hour"]][d["dst"]]), 1)
    E = 100 * E / E.sum()
    for a in range(4):
        for b in range(4):
            row[f"edge_{ROLES[a]}_to_{ROLES[b]}_pct"] = E[a, b]
    return row


# ================================================================
# 4. EXECUTION: DETERMINISTIC REFERENCE (REF) AND MONTE CARLO (A, B)
# ================================================================
print("Running deterministic reference matching...")
ref_roles, ref_sizes = one_run("REF")
ref_row = summarize(ref_roles)
print(f"REF persistence: {ref_row['persistence_pct']:.2f}% | "
      f"generator coverage: {ref_row['generator_coverage_pct']:.2f}%")

results = {}
for variant in ("A", "B"):
    print(f"
Running Monte Carlo Variant {variant} ({N_RUNS} runs)...")
    rows, role_counts = [], np.zeros((N, 4), dtype=int)
    watch_hour = np.zeros((N_HOURS, 4), dtype=int)
    for k in range(N_RUNS):
        seed = SEED_BASE + k
        roles, sizes = one_run(variant, seed)
        if not np.array_equal(sizes, ref_sizes):
            raise AssertionError(
                f"Matching cardinality differs from reference (variant {variant}, seed {seed})."
            )
        r = summarize(roles)
        r.update({"variant": variant, "run": k, "seed": seed})
        rows.append(r)
        for rr in range(4):
            role_counts[:, rr] += (roles == rr).sum(axis=0)
            watch_hour[:, rr] += (roles[:, WATCH_I] == rr)
        if (k + 1) % 50 == 0:
            print(f"  completed {k + 1}/{N_RUNS}")
    results[variant] = {
        "runs": pd.DataFrame(rows),
        "counts": role_counts,
        "watch": watch_hour,
    }


# ================================================================
# 5. REFERENCE vs MONTE CARLO COMPARISON AND CORE NODES
# ================================================================
total_assign = N_RUNS * N_HOURS
summary_rows = []
for variant in ("A", "B"):
    df = results[variant]["runs"]
    for metric in ("persistence_pct", "persistent_nodes", "generator_coverage_pct"):
        pct_rank = 100 * (df[metric] < ref_row[metric]).mean()
        summary_rows.append({
            "variant": variant, "metric": metric,
            "reference": ref_row[metric],
            "MC_mean": df[metric].mean(), "MC_std": df[metric].std(ddof=1),
            "MC_min": df[metric].min(), "MC_max": df[metric].max(),
            "reference_percentile_in_MC": pct_rank,
        })
summary_df = pd.DataFrame(summary_rows)
print("
===== REFERENCE vs MONTE CARLO =====")
print(summary_df.round(3).to_string(index=False))

core = {}
for variant in ("A", "B"):
    cnt = results[variant]["counts"]
    is_core = cnt.max(axis=1) == total_assign
    core[variant] = {
        r: [NODES[i] for i in range(N) if is_core[i] and cnt[i].argmax() == k]
        for k, r in enumerate(ROLES)
    }
    print(f"
Core nodes in Variant {variant}: {int(is_core.sum())}")
    for r in ROLES:
        print(f"  {r}: {len(core[variant][r])} {core[variant][r]}")

# Consistency check against deterministic reference
for variant in ("A", "B"):
    cnt = results[variant]["counts"]
    bad = []
    for i in range(N):
        if cnt[i].max() == total_assign:
            core_role = int(cnt[i].argmax())
            if not (ref_roles[:, i] == core_role).all():
                bad.append(NODES[i])
    print(f"Core vs reference inconsistencies (Variant {variant}):", bad if bad else "none")


# ================================================================
# 6. LIN TESTS FOR MULTI-INPUT LINEAR SYSTEMS
# ================================================================
def lin_tests(mode):
    """
    Evaluate Lin's structural controllability conditions (input-connection + rank condition
    via augmented bipartite matching) for graph mode ('lines' or 'full').
    """
    rows = []
    for d in hours:
        edges, idx, gens = d["edges_by_mode"][mode], d["idx"], d["gen_buses"]
        src = [idx[a] for a, _ in edges]
        dst = [idx[b] for _, b in edges]
        G = nx.DiGraph()
        G.add_nodes_from(NODES)
        G.add_edges_from(edges)

        # (i) Input-connection condition
        reached = set(gens)
        for g in gens:
            reached |= nx.descendants(G, g)
        unreached = sorted(set(NODES) - reached)

        # State bipartite graph matching size |M_t|
        H0 = nx.Graph()
        H0.add_nodes_from(range(N), bipartite=0)
        H0.add_nodes_from(range(N, 2 * N), bipartite=1)
        H0.add_edges_from((s, N + t) for s, t in zip(src, dst))
        M = len(bipartite.hopcroft_karp_matching(H0, top_nodes=range(N))) // 2

        # (ii) Rank condition via augmented bipartite graph V(Sigma_t)
        n_in = len(gens)
        R0 = N + n_in
        H = nx.Graph()
        H.add_nodes_from(range(R0), bipartite=0)
        H.add_nodes_from(range(R0, R0 + N), bipartite=1)
        for s_, t_ in zip(src, dst):
            H.add_edge(s_, R0 + t_)
        for k, g in enumerate(gens):
            H.add_edge(N + k, R0 + idx[g])
        m = bipartite.hopcroft_karp_matching(H, top_nodes=range(R0))
        covered = sum(1 for i in range(N) if (R0 + i) in m)

        rows.append({
            "graph": mode, "hour": d["hour"], "clock_hour": HOUR_LABELS[d["hour"]],
            "n_edges": len(edges), "n_inputs": n_in,
            "reached": len(reached), "reach_ok": len(reached) == N,
            "unreached_nodes": unreached,
            "aug_matching": covered, "match_ok": covered == N,
            "rank_defect": N - covered,
            "base_matching": M,
            "frac_deficit_covered": (covered - M) / (N - M),
            "controllable_Lin": (len(reached) == N) and (covered == N),
        })
    return pd.DataFrame(rows)


lin_dfs = {m: lin_tests(m) for m in ("lines", "full")}
for m, ldf in lin_dfs.items():
    print(f"
===== LIN TESTS - graph '{m}' =====")
    print(ldf.drop(columns=["unreached_nodes", "graph"]).round(3).to_string(index=False))
    print("Unreachable nodes (hour 0):", ldf.loc[0, "unreached_nodes"])
    print(f"Structurally controllable hours: {int(ldf['controllable_Lin'].sum())}/{N_HOURS}")


# ================================================================
# 7. CSV EXPORT
# ================================================================
ref_df = pd.DataFrame([ref_row])
ref_df.insert(0, "variant", "REF")

node_rows = []
for variant in ("A", "B"):
    cnt = results[variant]["counts"]
    for i in range(N):
        ref_set = set(ref_roles[:, i])
        row = {
            "variant": variant,
            "node": NODES[i],
            "ref_role": ROLES[ref_roles[0, i]] if len(ref_set) == 1 else "dynamic",
        }
        for k, r in enumerate(ROLES):
            row[f"{r}_pct"] = 100 * cnt[i, k] / total_assign
        row["core"] = cnt[i].max() == total_assign
        node_rows.append(row)

watch_rows = []
for variant in ("A", "B"):
    for h in range(N_HOURS):
        row = {
            "variant": variant,
            "hour": h,
            "clock_hour": HOUR_LABELS[h],
            "ref_role": ROLES[ref_roles[h, WATCH_I]],
            "in_degree": int(hours[h]["in_deg"][WATCH_I]),
            "out_degree": int(hours[h]["out_deg"][WATCH_I]),
        }
        for k, r in enumerate(ROLES):
            row[f"{r}_pct"] = 100 * results[variant]["watch"][h, k] / N_RUNS
        watch_rows.append(row)

tables = {
    "summary_ref_vs_MC.csv": summary_df,
    "reference_stats.csv": ref_df,
    "MC_runs_A.csv": results["A"]["runs"],
    "MC_runs_B.csv": results["B"]["runs"],
    "node_role_frequencies.csv": pd.DataFrame(node_rows),
    f"node{WATCH_NODE}_by_hour.csv": pd.DataFrame(watch_rows),
    "lin_tests_lines.csv": lin_dfs["lines"],
    "lin_tests_full.csv": lin_dfs["full"],
}
for fname, tab in tables.items():
    tab.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

print("Exported files to:", OUTPUT_DIR)
for fname in tables:
    print(" -", fname)


# ================================================================
# 8. FORMATTED SUMMARY FOR MANUSCRIPT TABLES & FIGURES
# ================================================================
print("TABLE 1: TIME-INVARIANT NODES (DETERMINISTIC REFERENCE)")
print("=" * 75)
perm_nodes_by_role = {r: [] for r in ROLES}
dynamic_nodes = []

for i in range(N):
    b_id = NODES[i]
    roles_24h = ref_roles[:, i]
    if len(set(roles_24h)) == 1:
        perm_nodes_by_role[ROLES[roles_24h[0]]].append(b_id)
    else:
        dynamic_nodes.append((i, b_id))

tot_perm = sum(len(v) for v in perm_nodes_by_role.values())
print(f"Total time-invariant nodes in Reference: {tot_perm} / {N} ({100 * tot_perm / N:.2f}%)")
for r in ROLES:
    print(f"\\textbf{{{r}}} & {', '.join(str(x) for x in perm_nodes_by_role[r])} \\\\")

print("
" + "=" * 75)
print("TABLE 2: DYNAMIC NODES (DETERMINISTIC REFERENCE - hourly share %)")
print("=" * 75)
dyn_rows = []
for i, b_id in dynamic_nodes:
    roles_24h = ref_roles[:, i]
    counts = Counter(roles_24h)
    dyn_rows.append({
        "Node": b_id,
        "Driver": f"{100 * counts[DRV] / N_HOURS:.1f}" if counts[DRV] > 0 else "--",
        "Sensor": f"{100 * counts[SNS] / N_HOURS:.1f}" if counts[SNS] > 0 else "--",
        "Dual": f"{100 * counts[DUA] / N_HOURS:.1f}" if counts[DUA] > 0 else "--",
        "Ordinary": f"{100 * counts[ORD] / N_HOURS:.1f}" if counts[ORD] > 0 else "--",
    })
df_dyn_print = pd.DataFrame(dyn_rows).sort_values("Node")
for _, row in df_dyn_print.iterrows():
    print(f"{row['Node']:3d} & {row['Driver']:5s} & {row['Sensor']:5s} & {row['Dual']:5s} & {row['Ordinary']:5s} \\\\")

print("
" + "=" * 75)
print(f"FIGURE 1: CASE STUDY NODE {WATCH_NODE}")
print("=" * 75)
watch_coords_in = []
watch_coords_out = []
watch_role_labels = []

for h in range(N_HOURS):
    r_name = ROLES[ref_roles[h, WATCH_I]]
    in_d = int(hours[h]["in_deg"][WATCH_I])
    out_d = int(hours[h]["out_deg"][WATCH_I])
    clk_h = HOUR_LABELS[h]
    watch_coords_in.append(f"({clk_h},{in_d})")
    watch_coords_out.append(f"({clk_h},{out_d})")
    watch_role_labels.append(f"\\node at (axis cs:{clk_h}, 2.5) {{\\tiny {r_name[0]}}};")
    print(f"Hour {clk_h:2d} -> Role={r_name:8s} | In={in_d} | Out={out_d}")

print("
--- TikZ \\addplot coordinates (in-degree):")
print(" ".join(watch_coords_in))
print("
--- TikZ \\addplot coordinates (out-degree):")
print(" ".join(watch_coords_out))
print("
--- TikZ role labels:")
print(" ".join(watch_role_labels))

print("
" + "=" * 75)
print("TABLE 3: CONDITIONAL DEGREE FREQUENCY BY ROLE (REFERENCE)")
print("=" * 75)
for r in ROLES:
    sub_in_gt = ref_row[f"{r}_in_gt_out_pct"]
    sub_out_gt = ref_row[f"{r}_out_gt_in_pct"]
    sub_eq = ref_row[f"{r}_eq_pct"]
    print(f"{r:8s} & {sub_in_gt:5.1f} & {sub_out_gt:5.1f} & {sub_eq:5.1f} \\\\")

print("
" + "=" * 75)
print("TABLES 4, 5, 6: MEAN DEGREE, STANDARD DEVIATION, AND CV (REFERENCE)")
print("=" * 75)
for r in ROLES:
    t_m, t_s, t_cv = ref_row[f"{r}_tot_mean"], ref_row[f"{r}_tot_std"], ref_row[f"{r}_tot_cv"]
    i_m, i_s, i_cv = ref_row[f"{r}_in_mean"], ref_row[f"{r}_in_std"], ref_row[f"{r}_in_cv"]
    o_m, o_s, o_cv = ref_row[f"{r}_out_mean"], ref_row[f"{r}_out_std"], ref_row[f"{r}_out_cv"]
    print(f"{r:8s} -> Tot: {t_m:.2f} (std {t_s:.2f}, CV {t_cv:.2f}) | In: {i_m:.2f} (std {i_s:.2f}, CV {i_cv:.2f}) | Out: {o_m:.2f} (std {o_s:.2f}, CV {o_cv:.2f})")

print("
" + "=" * 75)
print("FIGURE 2: HITS CENTRALITIES (REFERENCE)")
print("=" * 75)
for r in ROLES:
    h_val = ref_row[f"{r}_hub_mean"]
    a_val = ref_row[f"{r}_auth_mean"]
    print(f"({r}, {h_val:.6f})  % Hub Score")
    print(f"({r}, {a_val:.6f})  % Authority Score")

print("
Analysis completed successfully.")
