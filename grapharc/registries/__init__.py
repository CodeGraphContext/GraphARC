"""Operator-authored registries larger than the stdlib phases.

`grapharc.stdlib` ships general-purpose phases; the modules here ship whole
jobs — a registry, its state contract, its policy default and its loop,
travelling together under the `RegistryBundle` contract that
`grapharc plan --registry` reads. Each module is usable as
`grapharc.registries.<job>:build_registry` and owns its goal check, so a job
is judged complete by its own deterministic rule rather than by another
module's.
"""
