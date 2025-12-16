# gradio2.py
# -*- coding: utf-8 -*-

import os
import time
import json
import csv
import math
import random
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional


# ---------- FIX(0): avoid localhost check broken by proxy ----------
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# ---------- FIX(1): gradio_client JSON schema bug (schema may be bool/None) ----------
try:
    import gradio_client.utils as _gcu
    _old_get_type = getattr(_gcu, "get_type", None)
    if callable(_old_get_type):
        def _get_type_patched(schema):
            if not isinstance(schema, dict):
                return "any"
            return _old_get_type(schema)
        _gcu.get_type = _get_type_patched
except Exception:
    pass

import gradio as gr
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Matplotlib 中文字体兜底（消除 CJK missing warnings）
# =========================================================
def setup_chinese_font():
    try:
        from matplotlib import font_manager
        candidates = [
            "Noto Sans CJK SC", "Noto Sans CJK", "Source Han Sans SC",
            "SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei"
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return name
    except Exception:
        pass
    return None

setup_chinese_font()


# =========================================================
# 工具函数
# =========================================================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def stable_rng(seed_val: int, key: str) -> random.Random:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed_val + (h % 100000))

def safe_file_path(file_obj: Any) -> Optional[str]:
    """
    兼容 gr.File 返回：str / dict / FileData / tempfile 等。
    """
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        # gradio 有时返回 {"path":..., "name":...}
        return file_obj.get("path") or file_obj.get("name")
    # FileData / tempfile
    for attr in ["name", "path"]:
        if hasattr(file_obj, attr):
            v = getattr(file_obj, attr)
            if isinstance(v, str):
                return v
    return None


# =========================================================
# Mock：节点分类风险（第3章）
# =========================================================
def mock_node_cls_risk_prob(rng: random.Random) -> float:
    return rng.uniform(0.85, 0.99)

def risk_level_from_prob(p: float, thr_high: float = 0.90, thr_mid: float = 0.88):
    score = int(round(p * 100))
    if p >= thr_high:
        return score, "高风险", "#e53935"
    elif p >= thr_mid:
        return score, "中风险", "#fb8c00"
    else:
        return score, "低风险", "#43a047"


# =========================================================
# 数据接入：从 CSV/JSON 构图（可选）
# =========================================================
def parse_edges_from_csv(fp: str) -> List[Dict[str, Any]]:
    """
    CSV 约定（尽量宽松）：
    必需：src, dst
    可选：amt, ts, etype
    """
    edges = []
    with open(fp, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = (row.get("src") or row.get("source") or row.get("u") or "").strip()
            dst = (row.get("dst") or row.get("target") or row.get("v") or "").strip()
            if not src or not dst:
                continue
            amt_raw = row.get("amt") or row.get("amount") or row.get("money") or ""
            try:
                amt = float(amt_raw) if amt_raw != "" else 0.0
            except Exception:
                amt = 0.0
            etype = (row.get("etype") or row.get("type") or "tx").strip() or "tx"
            ts = (row.get("ts") or row.get("time") or row.get("timestamp") or "").strip()
            edges.append({"src": src, "dst": dst, "amt": amt, "etype": etype, "ts": ts})
    return edges

def parse_edges_from_json(fp: str) -> List[Dict[str, Any]]:
    """
    JSON 支持两类：
    1) list[{"src":..., "dst":..., "amt":..., "etype":..., "ts":...}]
    2) {"edges":[...]}
    """
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "edges" in obj:
        obj = obj["edges"]
    edges = []
    if isinstance(obj, list):
        for e in obj:
            if not isinstance(e, dict):
                continue
            src = str(e.get("src") or e.get("source") or e.get("u") or "").strip()
            dst = str(e.get("dst") or e.get("target") or e.get("v") or "").strip()
            if not src or not dst:
                continue
            amt = e.get("amt", e.get("amount", 0.0))
            try:
                amt = float(amt)
            except Exception:
                amt = 0.0
            etype = str(e.get("etype", e.get("type", "tx"))).strip() or "tx"
            ts = str(e.get("ts", e.get("time", e.get("timestamp", "")))).strip()
            edges.append({"src": src, "dst": dst, "amt": amt, "etype": etype, "ts": ts})
    return edges

def build_graph_from_edges(edges: List[Dict[str, Any]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for e in edges:
        u, v = e["src"], e["dst"]
        # 节点类型：简单规则（也可由数据字段扩展）
        def ntype(x: str):
            if x.startswith("D_") or x.lower().startswith("dev") or "device" in x.lower():
                return "device"
            return "account"
        if u not in G:
            G.add_node(u, ntype=ntype(u), label=u)
        if v not in G:
            G.add_node(v, ntype=ntype(v), label=v)
        G.add_edge(u, v,
                   amt=float(e.get("amt", 0.0)),
                   etype=str(e.get("etype", "tx")),
                   ts=str(e.get("ts", "")))
    return G

def sample_k_hop_subgraph(G: nx.DiGraph, center: str, hops: int) -> nx.DiGraph:
    """
    从全图按 hops 采样中心邻域（有向图：同时考虑入/出邻居）。
    """
    if center not in G:
        return nx.DiGraph()
    visited = {center}
    frontier = {center}
    for _ in range(max(1, int(hops))):
        nxt = set()
        for x in frontier:
            nxt.update(G.successors(x))
            nxt.update(G.predecessors(x))
        nxt -= visited
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return G.subgraph(visited).copy()


# =========================================================
# Mock：无外部数据时的子图构建（保留您原逻辑）
# =========================================================
def build_mock_subgraph(
    center_account: str,
    hops: int = 2,
    pattern: str = "随机",
    base_nodes: int = 12,
    anomaly_cnt: int = 3,
    seed: int = None,
):
    if seed is not None:
        random.seed(int(seed))

    n_total = max(8, min(60, base_nodes + (hops - 1) * 8))
    G = nx.DiGraph()

    center = center_account.strip() if center_account and center_account.strip() else "U_1234"
    G.add_node(center, ntype="account", label=center)

    acc_nodes = []
    for _ in range(n_total - 1):
        uid = f"U_{random.randint(1000, 9999)}"
        acc_nodes.append(uid)
        G.add_node(uid, ntype="account", label=uid)

    black_dev = "D_5678"
    G.add_node(black_dev, ntype="device", label=black_dev)

    if pattern == "随机":
        pattern = random.choice(["中心辐射型", "环状"])

    highlight_edges = set()

    if pattern == "中心辐射型":
        for u in random.sample(acc_nodes, k=min(len(acc_nodes), n_total // 2)):
            G.add_edge(center, u, amt=round(random.uniform(100, 5000), 2), etype="tx", ts="")
            if random.random() < 0.35:
                G.add_edge(u, center, amt=round(random.uniform(50, 3000), 2), etype="tx", ts="")
        G.add_edge(center, black_dev, amt=0.0, etype="bind", ts="")
        highlight_edges.add((center, black_dev))
    else:
        ring = random.sample(acc_nodes, k=min(6, len(acc_nodes)))
        if len(ring) < 4 and len(acc_nodes) >= 4:
            ring = ring + random.sample(acc_nodes, k=(4 - len(ring)))
        ring = [center] + ring[:5]
        for i in range(len(ring)):
            a = ring[i]
            b = ring[(i + 1) % len(ring)]
            G.add_edge(a, b, amt=round(random.uniform(200, 8000), 2), etype="tx", ts="")
            highlight_edges.add((a, b))

        G.add_edge(center, black_dev, amt=0.0, etype="bind", ts="")
        highlight_edges.add((center, black_dev))

        for u in random.sample(acc_nodes, k=min(len(acc_nodes), max(4, hops * 3))):
            if u != center:
                G.add_edge(u, center, amt=round(random.uniform(50, 6000), 2), etype="tx", ts="")

    anomalies = {black_dev}
    rest = [x for x in acc_nodes if x != center]
    for x in random.sample(rest, k=min(int(anomaly_cnt), len(rest))):
        anomalies.add(x)

    if pattern == "环状":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=int(seed) if seed is not None else 7, k=0.8)

    return G, pos, anomalies, highlight_edges, pattern, black_dev


# =========================================================
# 绘图：更“学术化”的图展示（图例、标题、边宽与高亮）
# =========================================================
def draw_graph(G, pos, center, anomalies, highlight_edges, show_labels=True, show_edge_amt=True):
    fig = plt.figure(figsize=(8.2, 5.8), dpi=120)
    ax = plt.gca()
    ax.set_title("交易网络子图 / K-hop 邻域（Prototype）", fontsize=12)
    ax.axis("off")

    node_colors = []
    for n in G.nodes():
        ntype = G.nodes[n].get("ntype", "account")
        if n == center:
            node_colors.append("#1e88e5")  # center
        elif n in anomalies:
            node_colors.append("#e53935")  # anomaly
        elif ntype == "device":
            node_colors.append("#8e24aa")  # device
        else:
            node_colors.append("#90a4ae")  # normal

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=560, ax=ax)

    tx_edges = [(u, v) for (u, v) in G.edges() if G.edges[(u, v)].get("etype") != "bind"]
    bind_edges = [(u, v) for (u, v) in G.edges() if G.edges[(u, v)].get("etype") == "bind"]

    # 边宽与金额粗略关联（便于展示“强联系”）
    def edge_width(u, v):
        amt = G.edges[(u, v)].get("amt", 0.0)
        try:
            amt = float(amt)
        except Exception:
            amt = 0.0
        return 1.2 + min(2.5, math.log1p(max(0.0, amt)) / 3.0)

    nx.draw_networkx_edges(
        G, pos,
        edgelist=tx_edges,
        edge_color=["#fb8c00" if (u, v) in highlight_edges else "#b0bec5" for (u, v) in tx_edges],
        width=[edge_width(u, v) for (u, v) in tx_edges],
        arrows=True, arrowstyle="-|>", arrowsize=12, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=bind_edges,
        edge_color=["#fb8c00" if (u, v) in highlight_edges else "#b0bec5" for (u, v) in bind_edges],
        width=1.2,
        style="dashed",
        arrows=True, arrowstyle="-|>", arrowsize=12, ax=ax
    )

    if show_labels:
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    if show_edge_amt:
        edge_labels = {}
        for (u, v) in tx_edges:
            amt = G.edges[(u, v)].get("amt", None)
            if amt is not None:
                edge_labels[(u, v)] = f"{amt:.0f}" if isinstance(amt, (int, float)) else str(amt)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    # 图例（学术展示常见）
    ax.scatter([], [], c="#1e88e5", s=80, label="Center")
    ax.scatter([], [], c="#e53935", s=80, label="Anomaly")
    ax.scatter([], [], c="#90a4ae", s=80, label="Normal")
    ax.scatter([], [], c="#8e24aa", s=80, label="Device")
    ax.legend(loc="lower left", frameon=True, fontsize=8)

    plt.tight_layout()
    return fig


# =========================================================
# UI：仪表盘 / KPI 卡片
# =========================================================
def render_gauge(score: int, color: str):
    score = max(0, min(100, int(score)))
    return f"""
    <div style="padding:12px 14px;border:1px solid #e0e0e0;border-radius:14px;">
      <div style="font-size:13px;color:#555;margin-bottom:6px;">风险评分（0-100）</div>
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="flex:1;height:12px;background:#f5f5f5;border-radius:999px;overflow:hidden;">
          <div style="width:{score}%;height:100%;background:{color};"></div>
        </div>
        <div style="min-width:44px;text-align:right;font-weight:800;">{score}</div>
      </div>
      <div style="font-size:12px;color:#888;margin-top:8px;">score = round(p×100)</div>
    </div>
    """

def render_badge(level: str, color: str):
    return f"""
    <div style="padding:12px 14px;border:1px solid #e0e0e0;border-radius:14px;">
      <div style="font-size:13px;color:#555;margin-bottom:6px;">风险等级</div>
      <span style="
        display:inline-block;
        padding:6px 12px;
        border-radius:999px;
        background:{color};
        color:white;
        font-weight:800;
        font-size:13px;
      ">{level}</span>
    </div>
    """

def render_kpis(kpi: Dict[str, Any]):
    """
    额外 KPI：节点/边规模、异常点数、可疑环数、延迟等。
    """
    def it(k, default="-"):
        v = kpi.get(k, default)
        return v
    return f"""
    <div style="padding:12px 14px;border:1px solid #e0e0e0;border-radius:14px;">
      <div style="font-size:13px;color:#555;margin-bottom:8px;">分析摘要（KPI）</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px;color:#444;">
        <div>Nodes：<b>{it("nodes")}</b></div>
        <div>Edges：<b>{it("edges")}</b></div>
        <div>Anomalies：<b>{it("anomaly_nodes")}</b></div>
        <div>Cycles：<b>{it("cycle_cnt")}</b></div>
        <div>Tx(24h)：<b>{it("tx_cnt_24h")}</b></div>
        <div>Latency(ms)：<b>{it("latency_ms")}</b></div>
      </div>
      <div style="font-size:12px;color:#888;margin-top:8px;">用于论文展示：规模/结构/效率维度</div>
    </div>
    """


# =========================================================
# 学术化分析：基础数据、特征贡献、规则命中、可疑路径
# =========================================================
def graph_tx_stats(G: nx.DiGraph) -> Tuple[int, float, float]:
    tx_amts = []
    for _, _, d in G.edges(data=True):
        if d.get("etype") == "tx" and isinstance(d.get("amt"), (int, float)):
            tx_amts.append(float(d["amt"]))
    tx_cnt = len(tx_amts)
    tx_sum = round(sum(tx_amts), 2) if tx_amts else 0.0
    tx_avg = round(tx_sum / max(1, tx_cnt), 2) if tx_cnt else 0.0
    return tx_cnt, tx_sum, tx_avg

def detect_cycles_around_center(G: nx.DiGraph, center: str, max_len: int = 6, max_cycles: int = 20) -> List[List[str]]:
    """
    小图可用：提取包含 center 的简单环（限制数量，避免卡顿）。
    """
    cycles = []
    if center not in G:
        return cycles
    try:
        for cyc in nx.simple_cycles(G):
            if len(cyc) <= max_len and center in cyc:
                # 规范化显示：从 center 开始旋转
                while cyc and cyc[0] != center:
                    cyc = cyc[1:] + [cyc[0]]
                cycles.append(cyc)
                if len(cycles) >= max_cycles:
                    break
    except Exception:
        return cycles
    return cycles

def extract_top_paths(G: nx.DiGraph, center: str, targets: List[str], k: int = 5, cutoff: int = 6) -> List[Dict[str, Any]]:
    """
    以 center->target 的简单路径作为“证据路径”示例（学术展示：Top-K 证据）。
    """
    out = []
    if center not in G:
        return out

    for t in targets[:]:
        if t not in G or t == center:
            continue
        # 尝试找若干条简单路径
        cnt = 0
        try:
            for path in nx.all_simple_paths(G, source=center, target=t, cutoff=cutoff):
                cnt += 1
                # 计算路径金额和（tx 边）
                s = 0.0
                for i in range(len(path) - 1):
                    d = G.edges.get((path[i], path[i+1]), {})
                    if d.get("etype") == "tx":
                        amt = d.get("amt", 0.0)
                        if isinstance(amt, (int, float)):
                            s += float(amt)
                out.append({
                    "target": t,
                    "path": " -> ".join(path),
                    "hops": len(path) - 1,
                    "tx_sum_on_path": round(s, 2),
                    "reason": "center 到异常节点/设备的连接路径（示例证据）"
                })
                if cnt >= k:
                    break
        except Exception:
            continue

    # 排序：金额更大/跳数更少优先
    out.sort(key=lambda x: (-x["tx_sum_on_path"], x["hops"]))
    return out[:k]

def mock_feature_contrib(rng: random.Random) -> List[Dict[str, Any]]:
    """
    Mock 的“特征贡献”表（论文展示：近似 SHAP/线性贡献形式）。
    """
    feats = [
        ("tx_cnt_24h", rng.randint(5, 300), rng.uniform(0.01, 0.05)),
        ("distinct_counterparties_7d", rng.randint(5, 200), rng.uniform(0.01, 0.04)),
        ("avg_tx_amt_24h", round(rng.uniform(50, 6000), 2), rng.uniform(0.01, 0.05)),
        ("max_tx_amt_24h", round(rng.uniform(500, 20000), 2), rng.uniform(0.01, 0.04)),
        ("recent_devices_30d", rng.randint(1, 5), rng.uniform(0.05, 0.20)),
        ("chargeback_flag", rng.choice([0, 0, 0, 1]), rng.uniform(0.10, 0.35)),
    ]
    rows = []
    for name, val, w in feats:
        contrib = float(val) * float(w) if isinstance(val, (int, float)) else float(w) * 10.0
        rows.append({
            "feature": name,
            "value": val,
            "weight(mock)": round(w, 4),
            "contribution": round(contrib, 4)
        })
    rows.sort(key=lambda x: -abs(x["contribution"]))
    return rows

def mock_rule_hits(G: nx.DiGraph, center: str, black_dev: str, cycles: List[List[str]], used_pattern: str) -> List[Dict[str, Any]]:
    """
    规则命中（学术展示：可解释规则证据）。
    """
    hits = []
    # Rule 1: 黑名单设备绑定
    if black_dev in G and G.has_edge(center, black_dev):
        hits.append({"rule": "R1_black_device_bind", "hit": True, "evidence": f"{center} -> {black_dev} (bind)"})
    else:
        hits.append({"rule": "R1_black_device_bind", "hit": False, "evidence": "-"})

    # Rule 2: 环流/Layering
    if len(cycles) > 0 or used_pattern == "环状":
        hits.append({"rule": "R2_cycle_layering", "hit": True, "evidence": f"cycles_cnt={len(cycles)}, pattern={used_pattern}"})
    else:
        hits.append({"rule": "R2_cycle_layering", "hit": False, "evidence": f"pattern={used_pattern}"})

    # Rule 3: 高出度/高入度（简单阈值示例）
    out_deg = G.out_degree(center) if center in G else 0
    in_deg = G.in_degree(center) if center in G else 0
    if out_deg >= 8 or in_deg >= 8:
        hits.append({"rule": "R3_high_degree_activity", "hit": True, "evidence": f"out={out_deg}, in={in_deg}"})
    else:
        hits.append({"rule": "R3_high_degree_activity", "hit": False, "evidence": f"out={out_deg}, in={in_deg}"})
    return hits

def compute_graph_anomaly_score(G: nx.DiGraph, center: str, black_dev: str, cycles: List[List[str]]) -> float:
    """
    一个简单的“图结构异常分数”示例，用于与节点分类分数融合（学术展示：融合策略）。
    """
    if center not in G:
        return 0.0
    out_deg = G.out_degree(center)
    in_deg = G.in_degree(center)
    bind_hit = 1.0 if (black_dev in G and G.has_edge(center, black_dev)) else 0.0
    cyc = min(1.0, len(cycles) / 5.0)
    deg = min(1.0, (out_deg + in_deg) / 20.0)
    # 组合（可在 UI 用权重调）
    return 0.45 * bind_hit + 0.35 * cyc + 0.20 * deg

def mock_base_data(acc: str, rng: random.Random, G: nx.DiGraph, anomalies: set, black_dev: str,
                   prob: float, score: int, level: str, used_pattern: str,
                   graph_score: float, fused_score: int, data_source: str,
                   hops: int, seed_val: int):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    out_deg = G.out_degree(acc) if acc in G else 0
    in_deg = G.in_degree(acc) if acc in G else 0
    tx_cnt, tx_sum, tx_avg = graph_tx_stats(G)

    base = {
        "time": now_str(),
        "account_id": acc,
        "data_source": data_source,
        "params": {"hops": hops, "seed": seed_val},
        "risk": {
            "node_cls_prob": round(float(prob), 4),
            "node_cls_score": int(score),
            "graph_anomaly_score": round(float(graph_score), 4),
            "fused_score(0-100)": int(fused_score),
            "risk_level": level,
            "pattern": used_pattern,
        },
        "key_device": black_dev,
        "graph_stats": {
            "nodes": int(n_nodes),
            "edges": int(n_edges),
            "center_out_degree": int(out_deg),
            "center_in_degree": int(in_deg),
            "anomaly_nodes": int(len(anomalies)),
            "graph_tx_cnt": int(tx_cnt),
            "graph_tx_sum": float(tx_sum),
            "graph_tx_avg": float(tx_avg),
        },
        "profile_features_mock": {
            "account_age_days": rng.randint(1, 2500),
            "kyc_level": rng.choice(["L1", "L2", "L3"]),
            "recent_login_ips_7d": rng.randint(1, 8),
            "recent_devices_30d": rng.randint(1, 5),
            "chargeback_flag": rng.choice([0, 0, 0, 1]),
        },
        "transaction_features_mock": {
            "tx_cnt_24h": rng.randint(5, 300),
            "tx_cnt_7d": rng.randint(50, 2000),
            "distinct_counterparties_7d": rng.randint(5, 200),
            "avg_tx_amt_24h": round(rng.uniform(50, 6000), 2),
            "max_tx_amt_24h": round(rng.uniform(500, 20000), 2),
        }
    }
    return base


# =========================================================
# 报告：流式推理输出（messages 格式）
# =========================================================
def stream_reasoning_report(acc: str, black_dev: str, used_pattern: str,
                            rules: List[Dict[str, Any]], paths: List[Dict[str, Any]],
                            fused_score: int, level: str):
    """
    更像“论文可用”的报告结构：结论-证据-路径-处置建议-复现实验信息。
    """
    def md_list(items: List[str]) -> str:
        return "\n".join([f"- {x}" for x in items]) + "\n"

    rule_lines = []
    for r in rules:
        flag = "✓" if r.get("hit") else "×"
        rule_lines.append(f"[{flag}] {r.get('rule')} ：{r.get('evidence')}")
    path_lines = []
    for p in paths:
        path_lines.append(f"{p.get('path')}  (tx_sum={p.get('tx_sum_on_path')}, hops={p.get('hops')})")

    text = (
        f"### 图推理分析报告（Prototype / Mock）\n\n"
        f"**结论**：账户 `{acc}` 评估为 **{level}**（fused_score={fused_score}/100）。\n\n"
        f"#### 1. 风险判定依据（可解释证据）\n"
        f"{md_list(rule_lines)}\n"
        f"#### 2. 关键路径证据（Top-K）\n"
        f"{md_list(path_lines if path_lines else ['未检索到有效路径（可能为弱连通/方向受限）'])}\n"
        f"#### 3. 异常模式归纳\n"
        f"- 异常模式（pattern）：`{used_pattern}`\n"
        f"- 关键设备（key device）：`{black_dev}`\n\n"
        f"#### 4. 建议处置（业务可落地）\n"
        f"- 触发二次核验（KYC/设备指纹/实名一致性）\n"
        f"- 提升交易风控策略（限额、延迟结算、增强审核）\n"
        f"- 进入人工复核队列并记录反馈用于策略迭代\n"
    )

    buf = ""
    for line in text.split("\n"):
        buf += line + "\n"
        time.sleep(0.08)
        yield buf


# =========================================================
# 可视化：简易“交易金额分布/时间线”（学术展示辅助）
# =========================================================
def draw_tx_amount_hist(G: nx.DiGraph):
    amts = []
    for _, _, d in G.edges(data=True):
        if d.get("etype") == "tx" and isinstance(d.get("amt"), (int, float)):
            amts.append(float(d["amt"]))
    fig = plt.figure(figsize=(6.8, 3.2), dpi=120)
    ax = plt.gca()
    ax.set_title("子图交易金额分布（Histogram）", fontsize=11)
    if amts:
        ax.hist(amts, bins=12)
        ax.set_xlabel("amount")
        ax.set_ylabel("count")
    else:
        ax.text(0.5, 0.5, "No tx amount found", ha="center", va="center")
        ax.axis("off")
    plt.tight_layout()
    return fig


# =========================================================
# 日志：风险清单、人工反馈（闭环）
# =========================================================
def append_log(log_list, record: Dict[str, Any], keep: int = 50):
    log_list = log_list or []
    log_list.append(record)
    return log_list[-keep:]

def log_to_table(log_list):
    headers = ["time", "account", "fused_score", "level", "pattern", "data_source"]
    rows = []
    for r in (log_list or []):
        rows.append([r.get(h, "") for h in headers])
    return rows

def append_feedback(feedback_list, item: Dict[str, Any], keep: int = 200):
    feedback_list = feedback_list or []
    feedback_list.append(item)
    return feedback_list[-keep:]

def feedback_to_table(feedback_list):
    headers = ["time", "account", "label", "note"]
    return [[r.get(h, "") for h in headers] for r in (feedback_list or [])]

def export_json(obj: Any, prefix: str):
    ensure_dir("exports")
    fp = os.path.join("exports", f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return fp

def export_report_md(report_md: str, base_data: Dict[str, Any], prefix: str = "report"):
    ensure_dir("exports")
    fp = os.path.join("exports", f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(report_md.strip() + "\n\n")
        f.write("----\n")
        f.write("### Base Data (JSON)\n")
        f.write("```json\n")
        f.write(json.dumps(base_data or {}, ensure_ascii=False, indent=2))
        f.write("\n```\n")
    return fp

def export_graph_graphml(G: nx.DiGraph, prefix: str = "graph"):
    ensure_dir("exports")
    fp = os.path.join("exports", f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.graphml")
    nx.write_graphml(G, fp)
    return fp


# =========================================================
# 核心回调：开始分析（更完整输出 + 流式报告）
# =========================================================
def start_analysis(
    account_id: str,
    data_mode: str,
    upload_file: Any,
    hops: int,
    pattern_choice: str,
    base_nodes: int,
    anomaly_cnt: int,
    show_labels: bool,
    show_edge_amt: bool,
    seed: int,
    w_node: float,
    w_graph: float,
    thr_high: float,
    thr_mid: float,
    chat_history: Any,
    graph_state: Any,
    log_state: Any,
):
    t0 = time.time()

    acc = account_id.strip() if account_id and account_id.strip() else "U_1234"
    hops = int(hops) if hops is not None else 2
    base_nodes = int(base_nodes) if base_nodes is not None else 12
    anomaly_cnt = int(anomaly_cnt) if anomaly_cnt is not None else 3
    seed_val = int(seed) if seed is not None else 7

    rng = stable_rng(seed_val, acc)

    # -------- 1) 构图：上传数据优先，否则 Mock --------
    data_source = "mock"
    full_G = None
    sub_G = None
    used_pattern = pattern_choice

    if data_mode == "上传数据构图":
        fp = safe_file_path(upload_file)
        if fp and os.path.exists(fp):
            try:
                if fp.lower().endswith(".csv"):
                    edges = parse_edges_from_csv(fp)
                elif fp.lower().endswith(".json"):
                    edges = parse_edges_from_json(fp)
                else:
                    edges = []
                if edges:
                    full_G = build_graph_from_edges(edges)
                    sub_G = sample_k_hop_subgraph(full_G, acc, hops=hops)
                    data_source = f"upload:{os.path.basename(fp)}"
            except Exception:
                full_G, sub_G = None, None

    if sub_G is None or sub_G.number_of_nodes() == 0:
        # fallback to mock
        G, pos, anomalies, highlight_edges, used_pattern, black_dev = build_mock_subgraph(
            center_account=acc,
            hops=hops,
            pattern=pattern_choice,
            base_nodes=base_nodes,
            anomaly_cnt=anomaly_cnt,
            seed=seed_val,
        )
        data_source = "mock"
    else:
        # 为上传图补齐 pos/anomaly/highlight/black_dev
        G = sub_G
        pos = nx.spring_layout(G, seed=seed_val, k=0.8)
        # 黑名单设备：优先寻找 device 节点，否则给一个默认
        device_nodes = [n for n in G.nodes() if G.nodes[n].get("ntype") == "device" or str(n).startswith("D_")]
        black_dev = device_nodes[0] if device_nodes else "D_5678"
        if black_dev not in G:
            G.add_node(black_dev, ntype="device", label=black_dev)
            if acc in G:
                G.add_edge(acc, black_dev, amt=0.0, etype="bind", ts="")
        # 异常节点：简单规则（device + 高度节点）
        anomalies = set()
        anomalies.add(black_dev)
        degs = sorted([(n, G.degree(n)) for n in G.nodes()], key=lambda x: -x[1])
        for n, _d in degs[:max(1, anomaly_cnt)]:
            if n != acc:
                anomalies.add(n)
        highlight_edges = set()
        if G.has_edge(acc, black_dev):
            highlight_edges.add((acc, black_dev))
        # pattern：上传图不强行指定，显示“上传图子图”
        used_pattern = "上传图子图"

    # -------- 2) 风险评分：节点分类 + 图异常（融合）--------
    prob = mock_node_cls_risk_prob(rng)
    node_score, _, _ = risk_level_from_prob(prob, thr_high=thr_high, thr_mid=thr_mid)

    cycles = detect_cycles_around_center(G, acc, max_len=6, max_cycles=20)
    graph_score = compute_graph_anomaly_score(G, acc, black_dev, cycles)

    # 融合：归一化到 0-100（示例）
    w_node = float(w_node)
    w_graph = float(w_graph)
    denom = max(1e-6, (w_node + w_graph))
    fused = (w_node * (node_score / 100.0) + w_graph * graph_score) / denom
    fused_score = int(round(max(0.0, min(1.0, fused)) * 100))

    # 用 fused_score 映射等级（便于论文解释：融合输出）
    fused_prob_proxy = fused_score / 100.0
    _, level, color = risk_level_from_prob(fused_prob_proxy, thr_high=thr_high, thr_mid=thr_mid)

    gauge_html = render_gauge(fused_score, color)
    badge_html = render_badge(level, color)

    # -------- 3) 绘图 + 统计 + 证据 --------
    fig_graph = draw_graph(G, pos, acc, set(anomalies), highlight_edges,
                           show_labels=bool(show_labels),
                           show_edge_amt=bool(show_edge_amt))

    fig_hist = draw_tx_amount_hist(G)

    targets = list(anomalies)[:]
    paths = extract_top_paths(G, acc, targets=targets, k=5, cutoff=6)

    rules = mock_rule_hits(G, acc, black_dev, cycles, used_pattern)
    contrib = mock_feature_contrib(rng)

    # 基础数据（面板）
    base_data = mock_base_data(
        acc, rng, G, set(anomalies), black_dev,
        prob=prob, score=node_score, level=level, used_pattern=used_pattern,
        graph_score=graph_score, fused_score=fused_score, data_source=data_source,
        hops=hops, seed_val=seed_val
    )

    evidence_md = (
        f"### 推理证据摘要（Prototype）\n\n"
        f"- **数据源**：`{data_source}`\n"
        f"- **融合评分**：`{fused_score}`（node={node_score}, graph={graph_score:.3f}）\n"
        f"- **关键设备**：`{black_dev}`\n"
        f"- **异常节点数**：`{len(anomalies)}`\n"
        f"- **环流证据**：`cycles_cnt={len(cycles)}`\n"
        f"- **模式**：`{used_pattern}`\n"
    )

    # 表格：节点/边（数据浏览）
    node_rows = []
    for n in G.nodes():
        node_rows.append([
            n,
            G.nodes[n].get("ntype", ""),
            1 if n == acc else 0,
            1 if n in anomalies else 0,
            int(G.in_degree(n)),
            int(G.out_degree(n)),
        ])
    edge_rows = []
    for u, v, d in G.edges(data=True):
        edge_rows.append([
            u, v,
            d.get("etype", ""),
            d.get("amt", 0.0),
            d.get("ts", ""),
            1 if (u, v) in highlight_edges else 0
        ])

    # KPI
    latency_ms = int(round((time.time() - t0) * 1000))
    kpi = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "anomaly_nodes": len(anomalies),
        "cycle_cnt": len(cycles),
        "tx_cnt_24h": base_data["transaction_features_mock"]["tx_cnt_24h"],
        "latency_ms": latency_ms
    }
    kpi_html = render_kpis(kpi)

    # Chatbot messages 初始化
    chat_history = [
        {"role": "user", "content": f"开始分析：{acc}"},
        {"role": "assistant", "content": "正在运行：数据构图 → 节点分类评分 → 图结构证据 → 报告生成（流式）..."}
    ]

    # 日志记录（用于风险清单）
    log_record = {
        "time": base_data["time"],
        "account": acc,
        "fused_score": fused_score,
        "level": level,
        "pattern": used_pattern,
        "data_source": data_source,
    }
    log_state = append_log(log_state, log_record, keep=50)
    log_table = log_to_table(log_state)

    # graph_state 保存（用于复盘/导出）
    graph_state = {
        "account": acc,
        "data_source": data_source,
        "base_data": base_data,
        "params": {
            "hops": hops, "pattern_choice": used_pattern, "base_nodes": base_nodes,
            "anomaly_cnt": anomaly_cnt, "show_labels": bool(show_labels),
            "show_edge_amt": bool(show_edge_amt),
            "seed": seed_val,
            "w_node": w_node, "w_graph": w_graph,
            "thr_high": thr_high, "thr_mid": thr_mid
        },
        # 注意：Graph/pos 直接放 state 里会变大；这里用边表复原即可
        "edges": edge_rows,
        "nodes": node_rows,
        "black_dev": black_dev,
        "anomalies": list(anomalies),
        "used_pattern": used_pattern
    }

    node_choices = list(G.nodes())
    node_dd_update = gr.update(choices=node_choices, value=acc)

    # 先返回“非流式部分”，让界面立即饱满
    yield (
        fig_graph, fig_hist,
        gauge_html, badge_html, kpi_html,
        chat_history,
        evidence_md, base_data,
        contrib, rules, paths,
        node_rows, edge_rows,
        node_dd_update,
        log_table,
        graph_state, log_state
    )

    # 再流式报告：把最后一条 assistant 更新为增量报告
    report_buf = ""
    for partial in stream_reasoning_report(acc, black_dev, used_pattern, rules, paths, fused_score, level):
        report_buf = partial
        chat_history[-1] = {
            "role": "assistant",
            "content": "正在运行：数据构图 → 节点分类评分 → 图结构证据 → 报告生成（流式）...\n\n" + report_buf
        }
        yield (
            fig_graph, fig_hist,
            gauge_html, badge_html, kpi_html,
            chat_history,
            evidence_md, base_data,
            contrib, rules, paths,
            node_rows, edge_rows,
            node_dd_update,
            log_table,
            graph_state, log_state
        )


# =========================================================
# 交互：中心切换（同时刷新图、基础数据、表格）
# =========================================================
def recenter_graph(new_center: str, graph_state: Dict[str, Any]):
    if not graph_state:
        return None, None, "请先点击“开始分析”。", {}, [], []

    acc = str(new_center) if new_center else graph_state.get("account", "U_1234")
    params = graph_state.get("params", {})
    seed_val = int(params.get("seed", 7))
    hops = int(params.get("hops", 2))

    # 从 edges 重建图
    edges = graph_state.get("edges", [])
    G = nx.DiGraph()
    for u, v, etype, amt, ts, hl in edges:
        if u not in G:
            G.add_node(u, ntype=("device" if str(u).startswith("D_") else "account"), label=u)
        if v not in G:
            G.add_node(v, ntype=("device" if str(v).startswith("D_") else "account"), label=v)
        try:
            amt = float(amt)
        except Exception:
            amt = 0.0
        G.add_edge(u, v, etype=etype, amt=amt, ts=ts)

    # 若切换中心不在当前图，直接提示
    if acc not in G:
        tip = f"中心节点 `{acc}` 不在当前子图中（请在左侧重新分析或上传更完整数据）。"
        return None, None, tip, graph_state.get("base_data", {}), graph_state.get("nodes", []), graph_state.get("edges", [])

    # 做一次 hop 采样（在当前子图上再采样，保证展示一致）
    sub = sample_k_hop_subgraph(G, acc, hops=hops)
    pos = nx.spring_layout(sub, seed=seed_val, k=0.8)

    anomalies = set(graph_state.get("anomalies", []))
    black_dev = graph_state.get("black_dev", "D_5678")
    highlight_edges = set()
    if sub.has_edge(acc, black_dev):
        highlight_edges.add((acc, black_dev))

    fig_graph = draw_graph(sub, pos, acc, anomalies, highlight_edges,
                           show_labels=bool(params.get("show_labels", True)),
                           show_edge_amt=bool(params.get("show_edge_amt", True)))
    fig_hist = draw_tx_amount_hist(sub)

    # 刷新基础数据（保持风险概率不变，但图统计变）
    prob = float(graph_state.get("base_data", {}).get("risk", {}).get("node_cls_prob", 0.0))
    node_score = int(graph_state.get("base_data", {}).get("risk", {}).get("node_cls_score", 0))
    used_pattern = graph_state.get("used_pattern", "子图")
    cycles = detect_cycles_around_center(sub, acc, max_len=6, max_cycles=20)
    graph_score = compute_graph_anomaly_score(sub, acc, black_dev, cycles)

    w_node = float(params.get("w_node", 1.0))
    w_graph = float(params.get("w_graph", 1.0))
    denom = max(1e-6, w_node + w_graph)
    fused = (w_node * (node_score / 100.0) + w_graph * graph_score) / denom
    fused_score = int(round(max(0.0, min(1.0, fused)) * 100))

    thr_high = float(params.get("thr_high", 0.90))
    thr_mid = float(params.get("thr_mid", 0.88))
    _, level, _ = risk_level_from_prob(fused_score / 100.0, thr_high=thr_high, thr_mid=thr_mid)

    rng = stable_rng(seed_val, acc)
    base_data = mock_base_data(
        acc, rng, sub, anomalies, black_dev,
        prob=prob, score=node_score, level=level, used_pattern=used_pattern,
        graph_score=graph_score, fused_score=fused_score,
        data_source=graph_state.get("data_source", "mock"),
        hops=hops, seed_val=seed_val
    )

    # 更新 node/edge 表
    node_rows = []
    for n in sub.nodes():
        node_rows.append([n, sub.nodes[n].get("ntype", ""), 1 if n == acc else 0, 1 if n in anomalies else 0,
                          int(sub.in_degree(n)), int(sub.out_degree(n))])
    edge_rows = []
    for u, v, d in sub.edges(data=True):
        edge_rows.append([u, v, d.get("etype", ""), d.get("amt", 0.0), d.get("ts", ""), 1 if (u, v) in highlight_edges else 0])

    tip = f"已切换中心节点：`{acc}`（K-hop={hops}），并刷新基础数据与统计。"
    return fig_graph, fig_hist, tip, base_data, node_rows, edge_rows


# =========================================================
# 复盘：回放最近一次（界面一致性）
# =========================================================
def replay_last(graph_state: Dict[str, Any]):
    if not graph_state:
        return None, None, render_gauge(0, "#9e9e9e"), render_badge("未分析", "#9e9e9e"), render_kpis({}), "暂无复盘数据。", {}, [], [], []

    acc = graph_state.get("account", "U_1234")
    base_data = graph_state.get("base_data", {})
    params = graph_state.get("params", {})
    seed_val = int(params.get("seed", 7))

    # 用 edges 重建图
    edges = graph_state.get("edges", [])
    G = nx.DiGraph()
    for u, v, etype, amt, ts, hl in edges:
        if u not in G:
            G.add_node(u, ntype=("device" if str(u).startswith("D_") else "account"), label=u)
        if v not in G:
            G.add_node(v, ntype=("device" if str(v).startswith("D_") else "account"), label=v)
        try:
            amt = float(amt)
        except Exception:
            amt = 0.0
        G.add_edge(u, v, etype=etype, amt=amt, ts=ts)

    hops = int(params.get("hops", 2))
    sub = sample_k_hop_subgraph(G, acc, hops=hops)
    pos = nx.spring_layout(sub, seed=seed_val, k=0.8)

    anomalies = set(graph_state.get("anomalies", []))
    black_dev = graph_state.get("black_dev", "D_5678")
    highlight_edges = set()
    if sub.has_edge(acc, black_dev):
        highlight_edges.add((acc, black_dev))

    fig_graph = draw_graph(sub, pos, acc, anomalies, highlight_edges,
                           show_labels=bool(params.get("show_labels", True)),
                           show_edge_amt=bool(params.get("show_edge_amt", True)))
    fig_hist = draw_tx_amount_hist(sub)

    fused_score = int(base_data.get("risk", {}).get("fused_score(0-100)", 0))
    level = str(base_data.get("risk", {}).get("risk_level", "未分析"))
    # 颜色按 fused_score 复用风险阈值
    thr_high = float(params.get("thr_high", 0.90))
    thr_mid = float(params.get("thr_mid", 0.88))
    _, _, color = risk_level_from_prob(fused_score / 100.0, thr_high=thr_high, thr_mid=thr_mid)

    gauge_html = render_gauge(fused_score, color)
    badge_html = render_badge(level, color)

    # KPI
    kpi = {
        "nodes": sub.number_of_nodes(),
        "edges": sub.number_of_edges(),
        "anomaly_nodes": len([x for x in sub.nodes() if x in anomalies]),
        "cycle_cnt": len(detect_cycles_around_center(sub, acc, max_len=6, max_cycles=20)),
        "tx_cnt_24h": base_data.get("transaction_features_mock", {}).get("tx_cnt_24h", "-"),
        "latency_ms": "-"
    }
    kpi_html = render_kpis(kpi)

    evidence_md = (
        f"### 离线复盘（Prototype）\n\n"
        f"- 账户：`{acc}`\n"
        f"- 数据源：`{graph_state.get('data_source', 'mock')}`\n"
        f"- 融合评分：`{fused_score}`，等级：**{level}**\n"
        f"- 关键设备：`{black_dev}`\n"
        f"- 模式：`{graph_state.get('used_pattern', '-')}`\n"
    )

    node_rows = graph_state.get("nodes", [])
    edge_rows = graph_state.get("edges", [])
    node_choices = list(sub.nodes())
    node_dd_update = gr.update(choices=node_choices, value=acc)

    return fig_graph, fig_hist, gauge_html, badge_html, kpi_html, evidence_md, base_data, node_rows, edge_rows, node_dd_update


# =========================================================
# 导出：报告 / 图 / 日志
# =========================================================
def on_export_report(chat_history: List[Dict[str, str]], base_data: Dict[str, Any]):
    # 从 chatbot 取最后一条 assistant 内容作为报告主体
    report_md = ""
    if isinstance(chat_history, list):
        for msg in chat_history[::-1]:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                report_md = msg.get("content", "")
                break
    if not report_md:
        report_md = "### Report\n\n(No report content)"
    fp = export_report_md(report_md, base_data or {}, prefix="analysis_report")
    return fp

def on_export_graph(graph_state: Dict[str, Any]):
    if not graph_state:
        return None
    # 重建图
    edges = graph_state.get("edges", [])
    G = nx.DiGraph()
    for u, v, etype, amt, ts, hl in edges:
        if u not in G:
            G.add_node(u, ntype=("device" if str(u).startswith("D_") else "account"), label=u)
        if v not in G:
            G.add_node(v, ntype=("device" if str(v).startswith("D_") else "account"), label=v)
        try:
            amt = float(amt)
        except Exception:
            amt = 0.0
        G.add_edge(u, v, etype=etype, amt=amt, ts=ts)
    fp = export_graph_graphml(G, prefix="subgraph")
    return fp

def on_export_risk_list(log_state: List[Dict[str, Any]]):
    fp = export_json(log_state or [], prefix="risk_list")
    return fp

def on_export_feedback(feedback_state: List[Dict[str, Any]]):
    fp = export_json(feedback_state or [], prefix="feedback")
    return fp


# =========================================================
# 人工审核反馈：闭环入口
# =========================================================
def submit_feedback(account_id: str, label: str, note: str, feedback_state: Any, graph_state: Any):
    acc = account_id.strip() if account_id else (graph_state.get("account") if graph_state else "U_1234")
    item = {"time": now_str(), "account": acc, "label": label, "note": note.strip()}
    feedback_state = append_feedback(feedback_state, item, keep=200)
    table = feedback_to_table(feedback_state)
    return feedback_state, table, ""


# =========================================================
# 批量分析：输入多账户，输出风险清单（用于论文展示“批处理”）
# =========================================================
def batch_analyze(accounts_text: str, seed: int, thr_high: float, thr_mid: float):
    seed_val = int(seed) if seed is not None else 7
    lines = [x.strip() for x in (accounts_text or "").splitlines() if x.strip()]
    rows = []
    for acc in lines[:200]:
        rng = stable_rng(seed_val, acc)
        prob = mock_node_cls_risk_prob(rng)
        score, level, _ = risk_level_from_prob(prob, thr_high=thr_high, thr_mid=thr_mid)
        rows.append([acc, round(prob, 4), score, level])
    return rows


# =========================================================
# UI：更饱满布局 + 多 Tab（在线 / 批量 / 反馈闭环）
# =========================================================


def build_ui():
    CSS = """
    .gradio-container { max-width: 100% !important; }
    #root, body { width: 100% !important; }
    """
    with gr.Blocks(title="面向节点分类和图推理的图结构基座大模型的金融反欺诈系统（Prototype）", css=CSS) as demo:
        gr.Markdown(
            "# 面向节点分类和图推理的图结构基座大模型的金融反欺诈系统（Prototype）\n"
            "**展示目标**：覆盖“多源数据接入 → 欧拉图构建与采样 → 通用图基座模型进行节点分类评分 → 强化学习微调的图基座模型进行图推理证据 → 可解释报告 → 导出与复盘 → 人工反馈闭环”。\n\n"
            "- **节点分类（第3章）**：高通量风险打分\n"
            "- **图推理（第4章）**：规则/路径/环流证据 + 报告生成\n"
            "- **学术展示增强**：风险分解、规则命中、证据路径、批量评估、导出与闭环反馈\n"
        )

        graph_state = gr.State(None)
        log_state = gr.State([])
        feedback_state = gr.State([])

        with gr.Tabs():
            # =========================
            # Tab 1：在线分析
            # =========================
            with gr.TabItem("在线分析（更完整展示）"):
                with gr.Row():
                    # 左侧：数据与配置 + 图
                    with gr.Column(scale=5, min_width=460):
                        with gr.Accordion("1) 数据接入 / 配置", open=True):
                            with gr.Row():
                                account_id = gr.Textbox(label="目标账户ID", value="U_1234")
                                data_mode = gr.Radio(label="数据模式", choices=["Mock 生成子图", "上传数据构图"], value="Mock 生成子图")
                            upload_file = gr.File(label="上传交易边表（CSV/JSON，可选）", file_types=[".csv", ".json"])
                            gr.Markdown(
                                "CSV 最简列：`src,dst,amt,etype,ts`（仅 src/dst 必需）。JSON 支持 `[{src,dst,...}]` 或 `{edges:[...]}`。"
                            )

                            with gr.Row():
                                hops = gr.Slider(label="K-hop 采样阶数", minimum=1, maximum=4, step=1, value=2)
                                seed = gr.Number(label="随机种子（可复现）", value=7, precision=0)

                            with gr.Accordion("Mock 子图形状（仅 Mock 模式有效）", open=False):
                                pattern_choice = gr.Radio(label="子图形状", choices=["随机", "中心辐射型", "环状"], value="随机")
                                base_nodes = gr.Slider(label="子图基础节点数", minimum=8, maximum=40, step=1, value=12)
                                anomaly_cnt = gr.Slider(label="异常节点数量（标红）", minimum=1, maximum=10, step=1, value=3)

                            with gr.Accordion("评分融合与阈值（论文可解释）", open=False):
                                with gr.Row():
                                    w_node = gr.Slider(label="融合权重：节点分类 w_node", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                                    w_graph = gr.Slider(label="融合权重：图异常 w_graph", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                                with gr.Row():
                                    thr_high = gr.Slider(label="高风险阈值（prob）", minimum=0.50, maximum=0.99, step=0.01, value=0.90)
                                    thr_mid = gr.Slider(label="中风险阈值（prob）", minimum=0.50, maximum=0.99, step=0.01, value=0.88)

                            with gr.Row():
                                show_labels = gr.Checkbox(label="显示节点标签", value=True)
                                show_edge_amt = gr.Checkbox(label="显示边金额", value=True)

                            with gr.Row():
                                btn_analyze = gr.Button("开始分析", variant="primary")
                                btn_replay = gr.Button("离线复盘：回放最近一次", variant="secondary")

                        with gr.Accordion("2) 图谱展示 / 多跳探索", open=True):
                            plot_graph = gr.Plot(label="交易网络子图（Graph）")
                            plot_hist = gr.Plot(label="交易金额分布（Histogram）")
                            node_dd = gr.Dropdown(label="切换中心节点（刷新子图与统计）", choices=[], value=None)
                            tip = gr.Markdown("")

                    # 右侧：KPI + 报告 + 面板
                    with gr.Column(scale=7, min_width=600):
                        with gr.Row():
                            gauge_html = gr.HTML(render_gauge(0, "#9e9e9e"))
                            badge_html = gr.HTML(render_badge("未分析", "#9e9e9e"))
                            kpi_html = gr.HTML(render_kpis({}))

                        with gr.Row():
                            with gr.Column(scale=7):
                                chatbot = gr.Chatbot(label="推理报告（messages，流式输出）", height=600)
                                with gr.Row():
                                    btn_export_report = gr.Button("导出报告（Markdown）")
                                    btn_export_graph = gr.Button("导出子图（GraphML）")
                                    export_file = gr.File(label="导出文件")
                            with gr.Column(scale=5):
                                evidence = gr.Markdown("### 证据摘要\n\n（点击开始分析后生成）")
                                base_data_panel = gr.JSON(label="基础数据（结构化）", value={})

                        with gr.Accordion("3) 可解释性：特征贡献 / 规则命中 / 证据路径", open=True):
                            with gr.Row():
                                feat_df = gr.Dataframe(
                                    headers=["feature", "value", "weight(mock)", "contribution"],
                                    value=[],
                                    interactive=False,
                                    row_count=8,
                                    col_count=(4, "fixed"),
                                    label="特征贡献（Mock，近似 SHAP 展示）"
                                )
                                rule_df = gr.Dataframe(
                                    headers=["rule", "hit", "evidence"],
                                    value=[],
                                    interactive=False,
                                    row_count=6,
                                    col_count=(3, "fixed"),
                                    label="规则命中（Mock）"
                                )
                            path_df = gr.Dataframe(
                                headers=["target", "path", "hops", "tx_sum_on_path", "reason"],
                                value=[],
                                interactive=False,
                                row_count=6,
                                col_count=(5, "fixed"),
                                label="证据路径 Top-K（Mock）"
                            )

                        with gr.Accordion("4) 数据浏览：节点表 / 边表（用于论文展示可复核）", open=False):
                            node_df = gr.Dataframe(
                                headers=["node", "ntype", "is_center", "is_anomaly", "in_deg", "out_deg"],
                                value=[],
                                interactive=False,
                                row_count=10,
                                col_count=(6, "fixed"),
                                label="节点表"
                            )
                            edge_df = gr.Dataframe(
                                headers=["src", "dst", "etype", "amt", "ts", "is_highlight"],
                                value=[],
                                interactive=False,
                                row_count=10,
                                col_count=(6, "fixed"),
                                label="边表"
                            )

                        with gr.Accordion("5) 风险清单（最近 50 条）", open=False):
                            log_table = gr.Dataframe(
                                headers=["time", "account", "fused_score", "level", "pattern", "data_source"],
                                value=[],
                                interactive=False,
                                row_count=10,
                                col_count=(6, "fixed"),
                                label="风险清单"
                            )
                            with gr.Row():
                                btn_export_risk = gr.Button("导出风险清单（JSON）")
                                export_risk_file = gr.File(label="导出文件")

                # ---- Events ----
                btn_analyze.click(
                    fn=start_analysis,
                    inputs=[
                        account_id, data_mode, upload_file,
                        hops, pattern_choice, base_nodes, anomaly_cnt,
                        show_labels, show_edge_amt, seed,
                        w_node, w_graph, thr_high, thr_mid,
                        chatbot, graph_state, log_state
                    ],
                    outputs=[
                        plot_graph, plot_hist,
                        gauge_html, badge_html, kpi_html,
                        chatbot,
                        evidence, base_data_panel,
                        feat_df, rule_df, path_df,
                        node_df, edge_df,
                        node_dd,
                        log_table,
                        graph_state, log_state
                    ],
                )

                node_dd.change(
                    fn=recenter_graph,
                    inputs=[node_dd, graph_state],
                    outputs=[plot_graph, plot_hist, tip, base_data_panel, node_df, edge_df],
                )

                btn_replay.click(
                    fn=replay_last,
                    inputs=[graph_state],
                    outputs=[plot_graph, plot_hist, gauge_html, badge_html, kpi_html, evidence, base_data_panel, node_df, edge_df, node_dd],
                )

                btn_export_report.click(
                    fn=on_export_report,
                    inputs=[chatbot, base_data_panel],
                    outputs=[export_file],
                )
                btn_export_graph.click(
                    fn=on_export_graph,
                    inputs=[graph_state],
                    outputs=[export_file],
                )
                btn_export_risk.click(
                    fn=on_export_risk_list,
                    inputs=[log_state],
                    outputs=[export_risk_file],
                )

            # =========================
            # Tab 2：批量分析
            # =========================
            with gr.TabItem("批量分析（论文评测展示）"):
                gr.Markdown(
                    "输入多账户（每行一个），一键生成批量风险清单。该页用于论文展示“批处理能力/吞吐入口”。"
                )
                with gr.Row():
                    accounts_text = gr.Textbox(label="账户列表（每行一个）", lines=10, value="U_1234\nU_2345\nU_3456")
                    with gr.Column():
                        seed_b = gr.Number(label="随机种子", value=7, precision=0)
                        thr_high_b = gr.Slider(label="高风险阈值（prob）", minimum=0.50, maximum=0.99, step=0.01, value=0.90)
                        thr_mid_b = gr.Slider(label="中风险阈值（prob）", minimum=0.50, maximum=0.99, step=0.01, value=0.88)
                        btn_batch = gr.Button("运行批量分析", variant="primary")

                batch_df = gr.Dataframe(
                    headers=["account", "prob", "score", "level"],
                    value=[],
                    interactive=False,
                    row_count=12,
                    col_count=(4, "fixed"),
                    label="批量风险输出"
                )

                btn_batch.click(
                    fn=batch_analyze,
                    inputs=[accounts_text, seed_b, thr_high_b, thr_mid_b],
                    outputs=[batch_df]
                )

            # =========================
            # Tab 3：人工反馈闭环
            # =========================
            with gr.TabItem("人工审核反馈（闭环入口）"):
                gr.Markdown(
                    "用于展示“模型输出 → 人工审核 → 反馈回流 → 策略迭代”的闭环能力。"
                )
                with gr.Row():
                    fb_account = gr.Textbox(label="账户ID（可为空，默认使用最近一次分析账户）", value="")
                    fb_label = gr.Radio(label="人工标签", choices=["Fraud", "Legit", "Unknown"], value="Fraud")
                    fb_note = gr.Textbox(label="备注（可选）", lines=4, placeholder="例如：命中黑名单设备，疑似团伙洗钱；建议冻结并复核。")
                btn_submit_fb = gr.Button("提交反馈", variant="primary")
                with gr.Row():
                    fb_table = gr.Dataframe(
                        headers=["time", "account", "label", "note"],
                        value=[],
                        interactive=False,
                        row_count=12,
                        col_count=(4, "fixed"),
                        label="反馈记录"
                    )
                    with gr.Column():
                        btn_export_fb = gr.Button("导出反馈（JSON）")
                        export_fb_file = gr.File(label="导出文件")

                btn_submit_fb.click(
                    fn=submit_feedback,
                    inputs=[fb_account, fb_label, fb_note, feedback_state, graph_state],
                    outputs=[feedback_state, fb_table, fb_note]
                )
                btn_export_fb.click(
                    fn=on_export_feedback,
                    inputs=[feedback_state],
                    outputs=[export_fb_file]
                )

        gr.Markdown(
            "---\n"
            "**运行方式**：`pip install gradio==6.1.0 networkx matplotlib` 后执行 `python gradio2.py`。\n"
            "如需上传 CSV/JSON 构图，请确保文件包含最少 `src,dst` 两列/字段。\n"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    import inspect
    kw = dict(server_name="127.0.0.1", server_port=7860, share=False, allowed_paths=["exports"])
    sig = inspect.signature(demo.launch)
    if "show_api" in sig.parameters:
        kw["show_api"] = False
    demo.launch(**kw)
