"""Robustness of trained GNN-MAPPO policies to communication degradation.

Self-contained on purpose: nothing in sar_env/, models/, training/ or evaluation/
is modified. The study only subclasses the environment and reads the checkpoints
already stored in results/ at the repository root.
"""
