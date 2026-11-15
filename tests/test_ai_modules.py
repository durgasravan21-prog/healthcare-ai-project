import unittest

from ai.search_algorithms import GRAPH, bfs, dfs, astar, greedy_best_first, hill_climbing
from ai.knowledge import naive_bayes_category, apply_rules
from ai.agents import simple_reflex_agent, multi_agent_demo


class SearchTests(unittest.TestCase):
    def test_search_algorithms_reach_goal(self):
        for algorithm in (bfs, dfs, astar, greedy_best_first, hill_climbing):
            result = algorithm(GRAPH, "Start", "End")
            self.assertIsNotNone(result["path"], algorithm.__name__)
            self.assertEqual(result["path"][0], "Start")
            self.assertEqual(result["path"][-1], "End")

    def test_bfs_finds_shortest_edge_count(self):
        result = bfs(GRAPH, "Start", "End")
        self.assertEqual(result["steps"], 5)


class KnowledgeTests(unittest.TestCase):
    def test_bayes_probabilities_sum_to_one(self):
        records = [
            {"reported_symptoms": "cough fever", "category": "Respiratory"},
            {"reported_symptoms": "rash itching", "category": "Skin"},
            {"reported_symptoms": "cough sore throat", "category": "Respiratory"},
        ]
        probabilities = naive_bayes_category(records, "cough")
        self.assertTrue(probabilities)
        self.assertAlmostEqual(sum(p for _, p in probabilities), 1.0, places=3)

    def test_production_rule_matches(self):
        record = {"age_group": "Senior", "reported_severity": "Severe", "visit_type": "Follow-up"}
        self.assertEqual(len(apply_rules(record)), 3)


class AgentTests(unittest.TestCase):
    def test_reflex_agent_uses_current_severity(self):
        result = simple_reflex_agent({"reported_severity": "Severe"})
        self.assertEqual(result.agent_name, "Simple reflex")
        self.assertIn("human review", result.action)

    def test_multi_agent_has_three_roles(self):
        result = multi_agent_demo({"category": "Respiratory"})
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
