"""Unit tests for the Tazkiya pipeline: python3 -m unittest training/test_tazkiya.py"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tazkiya as T

class Redaction(unittest.TestCase):
    def test_personal_data_is_replaced(self):
        s = T.redact("Mail me at ali.r@example.org or +62 812-3456-7890, card 4111 1111 1111 1111, key sk-abcdefghijklmnop1234, my name is Siti Aminah.")
        for tag in ("[EMAIL]", "[PHONE]", "[CARD]", "[SECRET]", "[NAME]"):
            self.assertIn(tag, s)
        for leak in ("ali.r@", "4111", "sk-abc", "Siti"):
            self.assertNotIn(leak, s)
    def test_maths_survives(self):
        s = T.redact("365 - 220 = 145 minutes; 1234567890123 is not a valid card; 3.14159")
        self.assertIn("145", s); self.assertIn("1234567890123", s); self.assertIn("3.14159", s)

class Rationale(unittest.TestCase):
    def test_bounded_and_no_fragment(self):
        r = T.rationale(" ".join(f"Step {i} adds {i} to the total." for i in range(200)) + " Let me check again. From 3:")
        self.assertLessEqual(T.approx_tokens(r), 150)
        self.assertTrue(r.endswith("."))

class Cap(unittest.TestCase):
    def test_no_domain_above_cap(self):
        items = [{"domain": "math"}] * 70 + [{"domain": "code"}] * 20 + [{"domain": "open"}] * 15 + [{"domain": "structured"}] * 15
        out, info = T.cap_domains(items, 0.30, 1)
        self.assertTrue(info["feasible"], info)
        self.assertTrue(all(v <= 0.30 + 1e-9 for v in info["shares"].values()))
    def test_infeasible_reported(self):
        _, info = T.cap_domains([{"domain": "math"}] * 5, 0.30, 1)
        self.assertFalse(info["feasible"])

class Gate(unittest.TestCase):
    def items(self, correct):
        return [{"task_id": f"t{i}", "system": "fast_only", "domain": "math" if i % 2 else "code", "correct": c} for i, c in enumerate(correct)]
    def test_clear_gain_passes(self):
        base = self.items([i % 2 == 0 for i in range(400)])
        cand = self.items([i % 2 == 0 or i % 3 == 0 for i in range(400)])
        d = T.gate_decision(base, cand, "fast_only")
        self.assertTrue(d["pass"], d["reasons"]); self.assertGreater(d["ci95"][0], 0)
    def test_regression_in_a_domain_blocks(self):
        base = self.items([True] * 400)
        cand = self.items([i % 2 == 0 for i in range(400)])  # loses every "math" item
        d = T.gate_decision(base, cand, "fast_only")
        self.assertFalse(d["pass"])
    def test_no_gain_blocks(self):
        same = self.items([i % 2 == 0 for i in range(400)])
        self.assertFalse(T.gate_decision(same, same, "fast_only")["pass"])

class Registry(unittest.TestCase):
    def test_promote_and_rollback_round_trip(self):
        import tempfile, json
        d = tempfile.mkdtemp(); T.REGISTRY = os.path.join(d, "reg.json")
        reg = T.load_registry(); first = reg["active"]["system1"]["gguf"]
        reg["history"].append({"at": "now", "role": "system1", "action": "promote", "from": reg["active"]["system1"], "to": {"gguf": "new.gguf"}, "gguf": "new.gguf"})
        reg["active"]["system1"] = {"gguf": "new.gguf"}; T.save_registry(reg)
        class A: role = "system1"; sign_off = "tester"
        self.assertEqual(T.rollback(A()), 0)
        self.assertEqual(json.load(open(T.REGISTRY))["active"]["system1"]["gguf"], first)
        self.assertEqual(T.rollback(A()), 1, "a second rollback has nothing to undo")

if __name__ == "__main__":
    unittest.main()
