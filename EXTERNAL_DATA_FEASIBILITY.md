# External sensor-failure dataset feasibility audit

Audit date: 2026-07-14

## Required evidence

A qualifying external recovery dataset must jointly provide:

1. a sensor channel that remains available while naturally corrupted or failed;
2. an independent downstream target that remains meaningful after suppressing that sensor;
3. event-, run- or device-level identifiers for leakage-resistant splitting; and
4. a public licence and enough metadata to reproduce the task.

## Candidates checked

| Candidate | Sensor-failure provenance | Independent downstream recovery target | Device/run replication | Decision |
|---|---|---|---|---|
| NASA, *Modeling, Detection, and Disambiguation of Sensor Faults for Aerospace Applications* | Bias, drift, scaling and dropout are simulated and injected into experimental electromechanical-actuator data | The public landing page exposes the paper rather than a directly usable labelled data package | One experimental setting is described | Not a qualifying natural-failure dataset; useful only as context |
| BASiC UAV sensor-failure dataset, DOI 10.5281/zenodo.8195068 | Six sensor failures are generated through ArduPilot software-in-the-loop parameter changes | Flight-state variables exist, but the published target is failure status and any recovery target would require a new, carefully justified task | 70 simulated flights | Potential controlled cross-domain study, not natural hardware-failure evidence; 13.9 GB archive and low evidential gain for the present paper |
| ALFA UAV dataset, DOI 10.1177/0278364920966642 | Real autonomous flights with engine and control-surface failures | Flight telemetry and event times are available | 47 processed sequences | Real platform evidence, but faults are engine/actuator failures rather than sensor-channel failures; does not answer the manuscript's estimand |
| Operational AHU dataset already used in the manuscript, DOI 10.6084/m9.figshare.27147678.v3 | Expert-annotated operational sensor-fault labels | The native label is the sensor fault itself; no independent component, comfort, energy or control outcome is supplied | Three buildings | Retain as a negative task-boundary experiment, not downstream recovery validation |
| UCI hydraulic test rig already used in the manuscript, DOI 10.24432/C5CW21 | Component labels are native, but pressure/vibration sensor corruptions are controlled | Four independent component-condition targets | One rig with repeated cycles | Retain as controlled external-platform validation; not natural sensor-failure evidence |

## Audit conclusion

No checked public candidate jointly satisfies natural sensor-failure provenance,
an independent downstream recovery target and multi-device replication. Adding a
large simulated or task-mismatched dataset would increase experiment count without
resolving the reviewer's central validity concern. The upgraded manuscript should
therefore retain an explicit controlled-intervention scope and state that the
highest evidence tier requires prospective multi-device acquisition with adjudicated
sensor-failure events and an independent downstream outcome.

## Source links

- NASA: https://data.nasa.gov/dataset/modeling-detection-and-disambiguation-of-sensor-faults-for-aerospace-applications
- BASiC: https://doi.org/10.5281/zenodo.8195068
- BASiC data article: https://doi.org/10.1016/j.dib.2024.110069
- ALFA: https://theairlab.org/alfa-dataset/
- ALFA article: https://doi.org/10.1177/0278364920966642
