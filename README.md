## **Project Overview**

This repository contains the computational framework and high-fidelity Smoothed Particle Hydrodynamics (SPH) datasets developed for my MSc Dissertation (Independent Research Project) at Imperial College London. This work focuses on the tidal disruption of asteroids during close encounters with Mars as a potential origin mechanism for **Phobos and Deimos**.

*   **Department:** Earth Science and Engineering / I-X
*   **Course:** MSc Applied Computational Science and Engineering (ACSE)
*   **Institution:** Imperial College London
*   **Supervision:** Dr. Jacob Kegerreis (NASA Ames Research Center)

---

### **Project Title**
**Parameter Sensitivity Analysis and Machine Learning Prediction of Tidal Disruption Outcomes in SPH Simulations of Martian Moon Formation**

### **Research Questions**
1. How do orbital and physical parameters (mass, velocity, spin, etc.) influence tidal disruption outcomes?
2. Which parameters most strongly control fragment formation and the mass of bound debris?
3. Can a machine learning model reliably predict disruption outcomes across this multidimensional parameter space?

### **Methodology & Infrastructure**
*   **Simulation Data:** High-fidelity SPH datasets modeling planetary-scale tidal forces and asteroid fragmentation.
*   **HPC Integration:** All simulations and data processing pipelines are executed on Imperial’s **High Performance Computing (HPC) cluster, CX3**.
*   **Framework:** A custom Python-based pipeline for particle-level data analysis, fragment identification, and sensitivity analysis, integrated with machine learning models to approximate simulation outcomes.

---

### **Abstract**
Tidal disruption is a leading hypothesis for generating the debris necessary to form the Martian satellite system. This project investigates how an asteroid's physical and orbital conditions, such as periapsis distance and encounter velocity, dictate the resulting debris structure. By leveraging SPH datasets and machine learning, this framework quantifies these relationships to provide a predictive tool for planetary science, bypassing the need for computationally expensive full simulations in every parameter regime.

The ultimate goal is to evaluate if tidal disruption debris disks could realistically coalesce into the current Martian moons and to provide a robust computational pipeline for future planetary disruption research.