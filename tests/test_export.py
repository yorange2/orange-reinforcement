import json
import os
import shutil
import subprocess
import tempfile
import unittest

import torch

import export_weights
from paodekuai.features import FEATURE_DIM, FEATURE_NAMES
from paodekuai.policy import MoveScorer, ValueNet, save_agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestExport(unittest.TestCase):
    """导出的 JSON 要能原样复现 PyTorch 的打分，网页版才可信。"""

    def export(self, **kwargs):
        scorer = MoveScorer(**kwargs)
        for param in scorer.parameters():
            torch.nn.init.normal_(param, std=0.3)
        return scorer, export_weights.dump_layers(scorer.net)

    def test_layers_are_described_in_order(self):
        _, layers = self.export(hidden=16, layers=2, norm="layer")
        self.assertEqual([l["type"] for l in layers],
                         ["linear", "layernorm", "relu",
                          "linear", "layernorm", "relu", "linear"])

    def test_without_norm(self):
        _, layers = self.export(hidden=16, layers=2, norm="none")
        self.assertEqual([l["type"] for l in layers],
                         ["linear", "relu", "linear", "relu", "linear"])

    def test_linear_weights_are_stored_row_per_output(self):
        scorer, layers = self.export(hidden=8, layers=1, norm="none")
        first = layers[0]
        self.assertEqual((len(first["w"]), len(first["w"][0])), (8, FEATURE_DIM))
        self.assertEqual(len(first["b"]), 8)

    def test_forward_matches_pytorch(self):
        scorer, layers = self.export(hidden=32, layers=2, norm="layer")
        x = torch.randn(6, FEATURE_DIM)
        with torch.no_grad():
            expected = scorer(x)
        for i, row in enumerate(x):
            got = export_weights.forward(layers, row.tolist())
            self.assertAlmostEqual(got, float(expected[i]), places=3)

    def test_unknown_layer_is_rejected(self):
        net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh())
        with self.assertRaises(ValueError):
            export_weights.dump_layers(net)

    def test_end_to_end_writes_a_usable_blob(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "agent.pt")
            out = os.path.join(tmp, "model.json")
            save_agent(weights, MoveScorer(hidden=16), ValueNet(hidden=8))
            export_weights.main(["--model", weights, "--out", out])

            blob = json.loads(open(out, encoding="utf-8").read())
            self.assertEqual(blob["feature_dim"], FEATURE_DIM)
            self.assertEqual(blob["feature_names"], FEATURE_NAMES)
            self.assertTrue(blob["net"])


@unittest.skipUnless(shutil.which("node"), "没装 node，跳过 JS 移植的一致性核对")
class TestJavaScriptParity(unittest.TestCase):
    """网页版把引擎重写了一遍 JS，必须和 Python 算出一样的东西。"""

    def test_shipped_weights_and_engine_agree(self):
        cases = os.path.join(ROOT, "tools", "parity_cases.json")
        if not os.path.exists(cases):
            subprocess.run(
                ["python", "tools/parity_export.py", "--games", "8"],
                cwd=ROOT, check=True, capture_output=True,
                env={**os.environ, "PYTHONPATH": ROOT},
            )
        result = subprocess.run(
            ["node", "tools/parity_check.mjs"], cwd=ROOT, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("全部一致", result.stdout)


if __name__ == "__main__":
    unittest.main()
