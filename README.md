# AFCFL — Adaptive Fairness Compensation for Federated Learning

#### Video Demonstration: https://youtu.be/kH8LLVZwAFE

## Methodology
We implemented an adaptive, fairness-aware federated learning pipeline in three steps:

1) As starting from scratch is time-consuming, we built on an existing federated learning repository: [Federated-Learning-PyTorch](https://github.com/AshwinRJ/Federated-Learning-PyTorch) by Ashwin Ramesh J.

2) **Base work reproduction: [FCFL](https://link.springer.com/chapter/10.1007/978-3-031-70352-2_23).**  
   Implemented unfairness queues with fixed **α** and **r**; higher-queue clients are prioritized to mitigate participation unfairness.

3) **Our contribution: AFCFL.**  
   - **Adaptive α:** queue growth adapts online to observed unfairness.  
   - **Adaptive r:** deterministic share of long-waiting clients adapts online.   
   - Benchmarked on IID vs Non-IID; tracked accuracy, variance, and unfairness metrics.

> **Results & analysis** are in `submissions/AML_Project_Report.pdf`.

---

## How to run
Check `command.bat` file for example runs.

