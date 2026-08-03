#!/usr/bin/env python3
"""把训练好的权重导出成 JSON，给网页版用。

导出的是一串"层描述"而不是固定的字段名，网页那边照着顺序走一遍就行，
以后改了网络结构（层数、宽度、要不要归一化）也不用动前端代码。

用法：
    python export_weights.py                      # models/agent.pt -> docs/model.json
    python export_weights.py --model x.pt --out y.json
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

import torch
import torch.nn as nn

from paodekuai.features import FEATURE_DIM, FEATURE_NAMES
from paodekuai.policy import load_agent

#: 权重保留几位小数。6 位对 float32 足够，JSON 体积能小三成。
PRECISION = 6


def dump_layers(net: nn.Sequential) -> List[dict]:
    """把网络拆成 [{type, ...}, ...]，前端顺序执行即可。"""
    layers: List[dict] = []
    for module in net:
        if isinstance(module, nn.Linear):
            layers.append({
                "type": "linear",
                "in": module.in_features,
                "out": module.out_features,
                # 按输出通道分行存，前端做点积时不用再转置
                "w": [[round(v, PRECISION) for v in row] for row in module.weight.tolist()],
                "b": [round(v, PRECISION) for v in module.bias.tolist()],
            })
        elif isinstance(module, nn.LayerNorm):
            layers.append({
                "type": "layernorm",
                "eps": module.eps,
                "w": [round(v, PRECISION) for v in module.weight.tolist()],
                "b": [round(v, PRECISION) for v in module.bias.tolist()],
            })
        elif isinstance(module, nn.ReLU):
            layers.append({"type": "relu"})
        else:
            raise ValueError(f"不认识的层 {type(module).__name__}，导出脚本要跟着更新")
    return layers


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/agent.pt", help="权重路径")
    parser.add_argument("--out", default="docs/model.json", help="导出到哪里")
    args = parser.parse_args(argv)

    agent = load_agent(args.model)
    scorer = agent.scorer

    blob = {
        "feature_dim": FEATURE_DIM,
        # 前端算完特征后会核对这个列表，顺序错了立刻能发现
        "feature_names": FEATURE_NAMES,
        "hidden": scorer.hidden,
        "layers": scorer.layers,
        "norm": scorer.norm,
        "n_params": scorer.n_params,
        "net": dump_layers(scorer.net),
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, separators=(",", ":"))

    import os

    size = os.path.getsize(args.out) / 1024
    print(f"{args.model} -> {args.out}")
    print(f"  {scorer.layers} 层 x {scorer.hidden} 宽，归一化 {scorer.norm}，{scorer.n_params:,} 个参数")
    print(f"  特征 {FEATURE_DIM} 维，文件 {size:.0f} KB")

    # 自检：JSON 里的权重跑一遍前向，必须和 PyTorch 一致
    check = torch.randn(7, FEATURE_DIM)
    with torch.no_grad():
        expected = scorer(check)
    got = torch.tensor([forward(blob["net"], row.tolist()) for row in check])
    gap = float((expected - got).abs().max())
    print(f"  自检：与 PyTorch 的最大偏差 {gap:.2e}")
    if gap > 1e-4:
        raise SystemExit("导出的权重和原模型对不上")
    return 0


def forward(layers: List[dict], x: List[float]) -> float:
    """纯 Python 走一遍导出的层，用来自检（和网页里的实现是同一套逻辑）。"""
    for layer in layers:
        if layer["type"] == "linear":
            x = [sum(w * v for w, v in zip(row, x)) + b for row, b in zip(layer["w"], layer["b"])]
        elif layer["type"] == "layernorm":
            mean = sum(x) / len(x)
            var = sum((v - mean) ** 2 for v in x) / len(x)
            scale = (var + layer["eps"]) ** 0.5
            x = [(v - mean) / scale * w + b for v, w, b in zip(x, layer["w"], layer["b"])]
        elif layer["type"] == "relu":
            x = [v if v > 0 else 0.0 for v in x]
    return x[0]


if __name__ == "__main__":
    raise SystemExit(main())
