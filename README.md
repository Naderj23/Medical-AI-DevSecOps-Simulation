# Medical-AI-DevSecOps-Simulation
Official Monte Carlo simulation code and reproducibility artifacts for Adaptive DevSecOps Framework in Medical Imaging AI.
# An Evidence-Driven, Adaptive DevSecOps Framework for Medical Imaging AI

This repository contains the official Monte Carlo simulation code, parameter configurations, and statistical verification scripts for the manuscript:
> **"An Evidence-Driven, Adaptive DevSecOps Framework for Medical Imaging AI: Heuristic Residual Risk Minimization Based on Provenance Graphs"**

## Overview
This simulation evaluates the proposed State-Aware W3C PROV Trust Graph coupled with a constrained 0/1 Knapsack greedy risk-minimization engine against static (B1), random (B2), and provenance-agnostic (B3) baselines across $K=10$ independent replications ($N=1,000$ commits per run).

## Requirements
* Python 3.8+
* `numpy`
* `scipy`

Install dependencies via pip:
```bash
pip install numpy scipy
