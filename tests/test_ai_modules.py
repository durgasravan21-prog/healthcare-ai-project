import unittest
from ai.search_algorithms import GRAPH,bfs,dfs,best_first,astar,hill_climbing
from ai.knowledge import naive_bayes,apply_rules
from ai.agents import agent_examples,multi_agent

class ProjectTests(unittest.TestCase):
    def test_search_algorithms(self):
        for method in (bfs,dfs,best_first,astar,hill_climbing):
            path=method(GRAPH,"Start","End")["path"]
            self.assertEqual(path[0],"Start"); self.assertEqual(path[-1],"End")

    def test_bfs_shortest_path(self):
        self.assertEqual(bfs(GRAPH,"Start","End")["steps"],5)

    def test_bayes_probabilities(self):
        rows=[{"reported_symptoms":"cough fever","category":"Respiratory"},
              {"reported_symptoms":"rash itching","category":"Skin"}]
        self.assertAlmostEqual(sum(p for _,p in naive_bayes(rows,"cough")),1.0,places=3)

    def test_rules_and_agents(self):
        row={"age_group":"Senior","reported_severity":"Severe","visit_type":"Follow-up"}
        self.assertEqual(len(apply_rules(row)),3)
        self.assertEqual(len(agent_examples(row)),5)
        self.assertEqual(len(multi_agent(row)),3)

if __name__=="__main__": unittest.main()
