# Multi-Layer Prompt Injection Defense System

This repository contains an innovative multi-layer defense system designed to protect large language models (LLMs) from prompt injection attacks. Built with a focus on real-world B2B applications, the system features a dynamic architecture that combines a train-free pre-filter, a transformer-based analysis layer, and an adaptive reverse defense post-layer. Key highlights include:

- **Pre-Filter Layer**: Utilizes regex patterns with a reinforcement learning (RL) agent to dynamically generate new detection rules, reducing manual updates and enhancing adaptability to evolving threats.
- **Transformer Layer**: Employs a fine-tuned DistilBERT model with dynamic input augmentation for deep analysis of suspicious prompts, improving accuracy and resilience.
- **Post-Layer**: Implements an LLM judge and adaptive reverse guard to verify outputs and mitigate attacks, with logging for continuous learning and improvement.
- **Context Engineering**: Integrates delimiters to isolate safe and harmful contexts, minimizing false positives and negatives.
- **On-Premise Ready**: Optimized for low latency (<2s) and scalability, suitable for deployment in secure environments.

## Purpose

The system aims to detect and neutralize over 90% of prompt injection attacks (e.g., jailbreaking, indirect injections) with a false positive rate below 10%, addressing a critical challenge in LLM security as of 2025.

## Getting Started

- Clone the repo: `git clone https://github.com/d12o6aa/prompt-injection-defense.git`
- Install dependencies: `pip install -r requirements.txt`
- Run the pipeline: `python main.py`

## Contributions

Contributions are welcome! Please open an issue or submit a pull request to enhance the RL agent, add new attack patterns, or optimize performance.

## License

MIT License - See the [LICENSE](LICENSE) file for details.
