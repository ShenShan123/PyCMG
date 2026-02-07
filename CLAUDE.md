# CLAUDE.md - BSIM-CMG Python Model Interface & Verification

## Project Overview
Develop a standalone Python interface for the BSIM-CMG Verilog-A model using OpenVAF/OSDI. The system must parse SPICE artifacts, execute model evaluations via a C++ OSDI host, and verify accuracy against NGSPICE Ground Truth using the exact same compiled binary.

## Environment & Tools
* **OpenVAF Compiler:** `/usr/local/bin/openvaf`
* **NGSPICE Simulator:** `/usr/local/ngspice-45.2/bin/ngspice`
* **Build System:** CMake / Make
* **Python Bindings:** PyBind11

## Directory Structure
* `bsim-cmg-va/`: Verilog-A source files (.va) and include files (.include).
* `pycmg/`: Python package (PyBind11 extension + helpers).
* `cpp/`: C++ OSDI host and bindings.
* `bsim-cmg-va/benchmark_test/`: SPICE netlists and model cards for verification.
* `build/`: Generated compilation artifacts (.osdi, binaries) when present.
* `build-deep-verify/`: Dedicated build outputs for verification tooling.
* `circuit_examples/`: Test circuits and verification outputs.

## Implementation Workflow

### 1. Model Compilation (OpenVAF)
* Compile `bsimcmg_main.va` directly to the OSDI shared library format.
* **Constraint:** Ensure the output is a standard `.osdi` file compatible with both the custom C++ host and NGSPICE.

### 2. C++ OSDI Host (PyBind11 Wrapper)
* **Dynamic Loading:** Load the compiled `.osdi` library using `dlopen`.
* **Symbol Binding:** Map OSDI standard functions (`create`, `set_param`, `evaluate`).
* **Derivative Extraction:**
    * Directly access the Jacobian matrix returned by the OSDI `evaluate` function.
    * **Strict Rule:** Do not calculate derivatives numerically (finite difference) in Python. Map specific Jacobian indices to model outputs (Gm, Gds, Gmb, etc.).
* **Memory Management:** Handle instance creation and destruction cleanly.

### 3. Python Interface Layer
* **A) Model Card Parser:**
    * Read `.lib`, `.l`, etc., files.
    * Extract global model parameters (e.g., `EOT`, `CIGC`).
    * Handle unit conversion (e.g., `15n` -> `1.5e-8`).
    * Apply default values for parameters undefined in the card (relying on OSDI defaults).
* **B) Netlist Parameter Extraction:**
    * Parse instance lines from SPICE netlists (e.g., `X1 ...` in `.cir` or `.sp`).
    * Extract instance-specific geometric parameters (e.g., `L`, `TFIN`, `NFIN`).
* **C) Simulation Conditions:**
    * Parse `.dc` or `.tran` commands to generate input voltage vectors ($V_d, V_g, V_s, V_e$) and temperature settings.
* **Execution:** Pass combined Model Params + Instance Params + Voltage Vectors to the C++ core.

### 4. Verification (NGSPICE Ground Truth)
* **Configuration:**
    * NGSPICE must load the **exact same** `.osdi` file generated in Step 1 using the `.osdi`. Do NOT use `.hdl` command.
    * Do not allow NGSPICE to re-compile the Verilog-A source; it must use the pre-compiled binary to ensure binary-level consistency.
* **Procedure:**
    1.  Run NGSPICE on test netlists to generate `.raw` output.
    2.  Run the Python Model Interface using identical parameters and voltage vectors.
    3.  Compare currents ($I_d, I_g$) and Derivatives ($g_m, g_{ds}$) numerically.
    4.  Assert accuracy within accepted tolerance (e.g., `1e-9`).
* **Temperature sweeps:** Always apply `.temp` in NGSPICE and pass `temperature` (K) to `pycmg.Instance`.
* **Stress tests:** Random OP points must compare **pycmg vs NGSPICE** (not osdi_eval). Use NGSPICE `.op` per point and compare I/Q/gm/gds/gmb within tolerances.
* **Robustness tests:** Use `tests/verify_utils.py` helpers via `pytest tests/test_robustness_helpers.py -v` for pulse stability utilities, param sensitivity, and thread safety checks.
* Comprehensive verification and ASAP7 reproduction now live under `tests/`:
    * `pytest tests/test_comprehensive.py -v`
    * `pytest tests/test_reproduce_asap7.py -v`
* `main.py` is the entrypoint for running pytest suites and data collection workflows.

## Development Rules
1.  **No Circuit Solvers:** The Python code must not contain KCL/KVL solvers or circuit simulation logic. It is strictly a Model Evaluator ($V \to I, Q, Jacobian$).
2.  **Source of Truth:** The OSDI binary is the single source of truth for physics calculations.
3.  **Data Flow:**
    * *Input:* Text (Netlists/Model Cards) -> Python Parsers -> Float Values.
    * *Compute:* Float Values -> C++ Wrapper -> OSDI Binary.
    * *Output:* OSDI Results (Values + Derivatives) -> Numpy Arrays -> Verification.

## Other Tips in This Project
* **Start every complex task in plan mode:** 
    * Pour your energy into the plan for 1-shot the implementation.
    * The moment something goes sideways, just switch back to plan mode and re-plan. Don't keep pushing.
    * Enter plan mode for verification steps, not just for the build.
* **Update CLAUDE.md:**
    * After every correction, update your CLAUDE.md so you don't make that mistake again.
* **Never be lazy:** 
    * Never be lazy in writing the code and running tests.
    * Do NOT use any simplifed equations or self-defined CMG models as reference, ALWAYS use simulation results as ground truth for comparison.
* Use subagents. 
    * Use a second agent to review the plan as a staff engineer.
    * If you want to try multiple solutions, use multiple subagents, git commit to different branches. Roll back and to the main branch and create new branch when the subagent find it's a dead end.
* Enable the "Explanatory" or "Learning" output style in /config to explain the *why* behind its changes.

## Lessons from Bugs (Keep Coming)
- Modelcard parsing must handle spaced `PARAM = VALUE` and exponent `1e+22`; otherwise key params (NBODY/NSD/NSEG/GEOMOD) silently default and mismatch ngspice.
- OSDI init out-of-bounds errors should be treated as warnings (matching ngspice behavior), not fatal.
- Some OSDI params are integer-typed; read/write using `PARA_TY_INT` to avoid garbage values.
- Internal-node DC solve must use residuals/Jacobian with cleared buffers; once params are correct, residuals match ngspice currents.
- Do not pass `prev_solve` to OSDI unless it is explicitly initialized; uninitialized `prev_solve` breaks DC/AC comparisons.
- Stress tests must align NGSPICE sign conventions: compare `i(vx)` directly to pycmg currents (no sign flip) for OP.

## Gap Checklist (Inventory vs Workflow)
- OSDI build pipeline: CMake builds `.osdi` via OpenVAF.
- C++ OSDI host: implemented in `cpp/osdi_host.cpp`.
- PyBind11 layer: `_pycmg` module exposes `Model`, `Instance`, and `eval_dc`.
- Transient evaluation: `_pycmg` exposes `eval_tran`; playback verification runs via `tests/verify_utils.py` and is exercised by ASAP7 verification tests.
- ASAP7 verification: `tests/test_asap7_full_verify.py` runs DC/AC/TRAN across ASAP7 modelcards; temperature/voltage sweeps live in `tests/test_comprehensive.py`.
- ASAP7 modelcard override: set `ASAP7_MODELCARD` to a file or directory to redirect the ASAP7 verification inputs.
