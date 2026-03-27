# Cellular Network Coverage Optimization Simulator

## Overview

This project models a simplified cellular network and applies cost-aware optimization techniques to improve signal coverage. It simulates how telecom networks handle weak signal regions while balancing performance and infrastructure cost.

## Problem Statement

Cellular networks often contain areas with poor signal strength, which affects user experience. Directly adding infrastructure is expensive, so operators must choose efficient strategies.

This project focuses on:

* Identifying weak signal regions
* Evaluating multiple optimization strategies
* Selecting the most cost-effective solution

## Why This Project Stands Out

This project goes beyond basic simulation by incorporating:
- Cost-aware decision making
- Multiple optimization strategies
- Realistic constraints (power limits, environment effects)

It reflects how telecom networks are optimized in real-world scenarios.

## Approach

The system models a 2D city grid with different environments (urban, suburban, rural). Signal propagation is calculated based on distance, path loss, and noise.

Weak coverage areas are detected using clustering, and the following strategies are evaluated:

* Transmission power adjustment
* Small cell deployment
* Macro tower placement
* Sectorization

Each strategy is scored using an improvement-to-cost ratio, and the most efficient option is applied iteratively.

## Results

The model typically achieves a 15–50% reduction in weak coverage areas depending on the scenario. It also demonstrates how low-cost strategies are preferred initially, followed by infrastructure-based solutions when necessary.

## Key Metrics (Sample Run)

Weak Zone Reduction: ~30–50%  
Average Signal Improvement: ~2x increase  
Cost Efficiency: ~2–3 cells per unit cost  

## Visualization

The project includes:

Initial Coverage: Shows weak signal regions before optimization

Optimized Coverage: Shows improved signal distribution after applying strategies

Improvement Map: Highlights areas where signal strength increased
<img width="1559" height="949" alt="image" src="https://github.com/user-attachments/assets/a582f452-d87d-48d3-b84c-d61945340457" />

## Real-World Relevance

This project reflects how telecom operators optimize network performance by balancing cost and coverage. It demonstrates decision-making under constraints, similar to real-world network planning.

## How to Run

1. Clone the repository:
   git clone https://github.com/tejaswinijudah/cellular-network-optimization

2. Navigate to the folder:
   cd cellular-network-optimization

3. Install dependencies:
   pip install -r requirements.txt

4. Run the simulation:
   python network_optimizer.py
   
## Tech Stack

* Python
* NumPy
* Matplotlib
* SciPy

## Limitations

* Simplified signal propagation model
* Limited interference modeling
* Greedy optimization approach (local optimum possible)

## Future Work

* Incorporate interference-aware optimization
* Introduce advanced optimization algorithms
* Use real-world datasets for validation

## Author

Tejaswini
