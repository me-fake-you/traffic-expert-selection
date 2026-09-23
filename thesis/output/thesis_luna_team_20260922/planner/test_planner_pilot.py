import json
import unittest
import pandas as pd
from run_planner_pilot import build_cards, prompt, validate


class PlannerContractTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({"case_id": [f"q{i:03d}" for i in range(24)],
            "sample_hash": [str(i) for i in range(24)], "group": ["train_a"]*12+["train_b"]*12,
            "y": [0,1]*12, "p_temporal": [.3,.7]*12, "p_stats": [.9,.1]*12})

    def test_query_labels_and_uncalled_output_never_reach_prompt(self):
        cards = build_cards(self.rows)
        modified = self.rows.copy()
        modified["y"], modified["p_stats"], modified["group"] = 1, .987654321, "forbidden-group"
        for rag in (False, True):
            self.assertEqual(prompt(self.rows, cards, rag), prompt(modified, cards, rag))
            body = json.loads(prompt(self.rows, cards, rag))
            self.assertTrue(all(set(c) == {"case_id", "first_probability", "first_margin", "extra_expert_available"} for c in body["case_data"]))

    def test_response_has_exact_unique_budget(self):
        ids = self.rows.case_id.tolist()
        self.assertEqual(validate(json.dumps({"selected_case_ids": ids[:6]}), ids), ids[:6])
        for bad in ([ids[0]]*6, ids[:5], ids[:7], ids[:5]+["foreign"], [0]*6):
            with self.assertRaises(ValueError):
                validate(json.dumps({"selected_case_ids": bad}), ids)
        with self.assertRaises(ValueError):
            validate(json.dumps({"selected_case_ids": ids[:6], "final_verdict": "invented"}), ids)

    def test_cards_retain_harm_and_missing_support(self):
        cards = build_cards(self.rows)
        self.assertEqual(cards["3"]["new_errors"], 12)
        self.assertEqual(cards["3"]["mean_error_reduction"], -1)
        self.assertIsNone(cards["0"]["mean_error_reduction"])
        self.assertEqual(cards["3"]["independent_groups_observed"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
