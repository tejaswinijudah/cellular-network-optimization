# Cellular Network Coverage Optimization Simulator

## Overview

This project models a simplified cellular network and applies cost-aware optimization techniques to improve signal coverage. It simulates how telecom networks handle weak signal regions while balancing performance and infrastructure cost.

## Problem Statement

Cellular networks often contain areas with poor signal strength, which affects user experience. Directly adding infrastructure is expensive, so operators must choose efficient strategies.

This project focuses on:

* Identifying weak signal regions
* Evaluating multiple optimization strategies
* Selecting the most cost-effective solution

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

## Visualization

The project includes:

* Initial network coverage
* Optimized network coverage
* Coverage improvement map
<img width="1559" height="949" alt="image" src="https://github.com/user-attachments/assets/a582f452-d87d-48d3-b84c-d61945340457" />

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
