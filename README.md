 Controllability and Role Persistence of the IEEE 118-Bus Grid over 24 Hours


This repository provides the complete pipeline used to:
- Reconstruct hourly directed active power flow graphs of the IEEE 118-bus power grid.
- Classify network buses into Driver, Sensor, Dual, and Ordinary roles via the Hopcroft–Karp maximum bipartite matching algorithm.
- Evaluate node role persistence and generator coverage under deterministic reference ordering and Monte Carlo two-sided vertex permutations.
- Test Lin's structural controllability conditions (input-connection and rank condition via augmented bipartite matching).
- Compute Kleinberg's Hub and Authority (HITS) centralities across node categories.


Hourly OPF snapshots of the IEEE 118-bus system are openly available on Zenodo:
- Tsydenov, E.; Prokhorov, A. *Power flow data of IEEE-118 bus system*. Zenodo (2022), Release v1.0.0. DOI: [10.5281/zenodo.7306932](https://doi.org/10.5281/zenodo.7306932).
Download the 24 hourly files for January 1, 2024 (`2024_01_01_00_00_00.json` to `2024_01_01_23_00_00.json`) and place them in `data/samples/`.
