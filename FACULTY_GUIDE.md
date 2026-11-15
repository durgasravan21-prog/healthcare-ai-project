# CSE276 Faculty Demonstration and Viva Guide

## One-minute project explanation
"Our project uses 500 synthetic healthcare-style records to demonstrate concepts from all six units of CSE276 Artificial Intelligence Foundations. It includes classical search algorithms, knowledge representation and reasoning, intelligent-agent simulations, introductory supervised and unsupervised learning, prompt engineering and local retrieval, responsible-AI checks, and an overview of the AI development ecosystem. A Streamlit dashboard provides one interface for demonstrating these concepts. This is an educational project, not a medical diagnosis or treatment system."

## Suggested 10-minute demonstration
1. **Unit I:** Explain the dataset, AI lifecycle, state-space graph, initial state, actions, goal and path cost.
2. **Unit II:** Run BFS, DFS, Greedy Best-First, Hill Climbing and A*. Compare path, explored nodes, cost and time. Show semantic network, frames, rules, logic and Bayes example.
3. **Unit III:** Explain PEAS and compare reflex, model-based, goal-based, utility-based and learning agents. Show the cooperative agents.
4. **Unit IV:** Run Decision Tree and K-Means. Explain supervised versus unsupervised learning and discuss deep learning and reinforcement learning conceptually.
5. **Unit V:** Compare prompt designs, retrieve local notes with RAG, show a record explanation and group distribution, then discuss privacy, fairness and human oversight.
6. **Unit VI:** Explain Python, scikit-learn, TensorFlow, PyTorch, Hugging Face, Ollama, LangChain and vector databases.

## Algorithm explanations
- **BFS:** FIFO queue; expands shallowest nodes first. Complete for finite branching and optimal when all step costs are equal.
- **DFS:** LIFO stack; explores deeply then backtracks. Not generally optimal.
- **Greedy Best-First:** selects the frontier node with smallest h(n). It may be fast but is not generally optimal.
- **Hill Climbing:** moves to a better neighbour according to a heuristic. It may stop at a local optimum or plateau.
- **A*:** f(n) = g(n) + h(n). With suitable admissible/consistent heuristics and nonnegative edge costs, it finds an optimal path.
- **Naive Bayes:** applies Bayes theorem with a conditional-independence assumption. Here it estimates synthetic category labels from symptom words only for learning.
- **Decision Tree:** learns if/else-like splits from labelled examples.
- **K-Means:** groups records by distance to cluster centroids; clusters are not diagnoses.

## Viva questions and short answers
**1. Why is this an AI project and not only visualization?**  
It demonstrates search, knowledge representation, reasoning, agents, probabilistic inference and machine learning in addition to charts.

**2. What is a state?**  
A representation of the current situation in a problem. In our graph, each workflow location is a state.

**3. What is a heuristic?**  
An estimate of the remaining cost from a state to the goal.

**4. Difference between BFS and DFS?**  
BFS uses a FIFO queue and explores level by level; DFS uses a stack and explores deeply before backtracking.

**5. What does A* calculate?**  
It combines cost already paid g(n) with estimated remaining cost h(n): f(n)=g(n)+h(n).

**6. What is knowledge representation?**  
A way to store facts and relationships so a program can use them for reasoning.

**7. What is a production rule?**  
An IF–THEN rule mapping a condition to an observation or action.

**8. What is Bayes theorem?**  
P(A|B)=P(B|A)P(A)/P(B); it updates a probability when evidence is observed.

**9. What is PEAS?**  
Performance measure, Environment, Actuators and Sensors; it describes an agent's task environment.

**10. Reflex versus model-based agent?**  
A reflex agent uses the current percept; a model-based agent also uses internal state or memory.

**11. Supervised versus unsupervised learning?**  
Supervised learning uses labelled examples; unsupervised learning finds structure in unlabelled data.

**12. What is overfitting?**  
When a model fits training examples too closely and performs poorly on unseen data.

**13. What is RAG?**  
Retrieval-Augmented Generation retrieves relevant information and supplies it as context to a generative model.

**14. What is hallucination?**  
A generated response that sounds plausible but is unsupported or incorrect. Grounding, constrained prompts and human verification can reduce risk.

**15. Why is this not a medical decision system?**  
The data and labels are synthetic, models are classroom demonstrations and no clinical validation has been performed.

**16. What is explainable AI?**  
Methods that help people understand which inputs or rules influenced an output.

**17. What is a vector database?**  
A database designed to store and search vector embeddings, often by semantic similarity.

**18. What is AGI?**  
A hypothetical AI with broad general capabilities across many domains; this project does not implement AGI.

## Important presentation wording
Say "synthetic category estimate", "illustrative workflow" and "educational model". Do not say "AI diagnosis", "recommended treatment", "real patient history" or "clinically accurate prediction".
