# Electritect README

## Introduction
Electricity is essential for residential, commercial, and industrial applications, driving quality of life and economic development globally. As economies and technologies advance, electricity demand, especially among large corporations, continues to increase. These corporations, which are major energy consumers, face rising operational costs and regulatory pressures to improve energy efficiency and reduce greenhouse gas emissions.

## Our Solution: Electritect
**Electritect** is an innovative Software as a Service (SaaS) designed to optimize energy consumption for medium and large corporations. By identifying inefficiencies and suggesting targeted improvements, Electritect helps businesses:
- **Reduce Operating Costs:** Optimize energy use to lower costs and improve profitability.
- **Enhance Environmental Sustainability:** Reduce carbon footprints and contribute to sustainability.
- **Ensure Compliance with Regulations:** Meet stringent energy consumption regulations to avoid penalties and improve public perception.

## Proof of Concept
To validate our solution, we developed the technical aspects of Electritect using a dataset from a three-site industry in the European Union. The dataset includes detailed operational data on energy consumption and other environmental measurements collected by IoT sensors.

## Optimization Algorithm
The core of our solution is an optimization algorithm designed using Pyomo, a Python-based open-source optimization modeling language. Key components of the optimization model include:
- **Objective Function:** Minimize energy costs by optimizing energy consumption.
- **Decision Variables & Parameters:** Control variables like temperature set points and active equipment numbers, while accounting for fixed parameters like energy prices and production schedules.
- **Constraint Equations:** Ensure solutions are practical and adhere to real-world limitations.
- **Bounds:** Define limits for variables to simulate real-world scenarios.
- **Forecasting:** Utilize ARIMA models to predict future energy needs based on historical data.

## Results
Our optimization algorithm demonstrated significant cost savings. For instance, in June 2023, the algorithm reduced energy costs by approximately 20.5%, translating to savings of 6,375.40 Euros.

## Future Development
Future improvements for Electritect include exploring advanced optimization techniques, standardizing implementation processes, and expanding to optimize other energy sources like natural gas and water systems. Additionally, partnerships with established IoT suppliers will enhance data collection and scalability.

## References
For a detailed list of references, please refer to the project's documentation.

---

This summary provides an overview of the Electritect project, highlighting its purpose, methodology, and results. For more detailed information, please refer to the full documentation.
