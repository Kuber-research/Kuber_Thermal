"""Assemble a single-region cold-plate coolant-channel case (buoyantSimpleFoam).

The domain IS the coolant (the channel interior) - a rectangular duct. Coolant
enters the inlet, leaves the outlet at a reference pressure; the base wall carries
a chip heat flux (externalWallHeatFluxTemperature, mode flux), the other walls are
adiabatic. Reuses the stabilized buoyantSimpleFoam recipe from build_bsf
(absolute p_rgh ~1e5, GAMG/PBiCGStab, p_rgh 0.3 / U 0.2 / h 0.5 relaxation,
fvOptions temperature/velocity limiting). blockMesh only -> robust + fast, no snappy.

Domain (metres):  x streamwise [0,L], y height [0,H] (g=-y), z width [0,W].
Patches: inlet(x-) outlet(x+) base(y-, heated) top(y+) sides(z-,z+).

Usage:  python build_coldplate.py --case <dir> --params <params.json> --iters 800
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

# reuse the exact stabilized physics dicts from the heatsink pipeline
from build_bsf import (thermo, turb, g_dict, controlDict, fvSchemes,
                       hdr, _field, FLUIDS)


# tighter residual gate than the heatsink recipe: with the loose h 1e-3 the SIMPLE
# loop stops ~iter 500 with a ~10% global energy imbalance; h/p_rgh 1e-5 converges
# (~1600 iters) to an EXACT enthalpy balance (validated: mixing-cup dT == Q/(mCp)).
def fvSolution_cp():
    return hdr("dictionary", "fvSolution") + """
solvers
{
    p_rgh { solver GAMG; tolerance 1e-8; relTol 0.01; smoother DICGaussSeidel; }
    "(U|h|k|epsilon|omega)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-8; relTol 0.01; }
}
SIMPLE
{
    momentumPredictor no; nNonOrthogonalCorrectors 1; pRefCell 0; pRefValue 0;
    residualControl { p_rgh 1e-5; h 1e-5; "(k|epsilon|omega)" 1e-4; }
}
relaxationFactors
{
    rho 1.0; p_rgh 0.3; U 0.2; h 0.5; "(k|epsilon|omega)" 0.3;
}
"""


# ---------------- mesh (blockMesh only) ----------------
def blockMeshDict(d):
    L, H, W = d["L"], d["H"], d["W"]
    # cell counts: ~1.5 mm streamwise/spanwise, 32 across the (thin) height, graded
    # finer toward the heated base for the thermal/velocity boundary layer.
    nx = min(200, max(60, round(L / 0.0015)))
    nz = min(60, max(16, round(W / 0.0015)))
    ny = 32
    gy = 6                                            # last(top)/first(base) cell size -> fine at base
    v = [(0,0,0),(L,0,0),(L,H,0),(0,H,0),(0,0,W),(L,0,W),(L,H,W),(0,H,W)]
    vs = "\n".join(f"    ({x} {y} {z})" for x, y, z in v)
    return hdr("dictionary", "blockMeshDict") + f"""
scale 1;
vertices
(
{vs}
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 {gy} 1) );
edges ();
boundary
(
    inlet  {{ type patch; faces ( (0 4 7 3) ); }}
    outlet {{ type patch; faces ( (1 2 6 5) ); }}
    base   {{ type wall;  faces ( (0 1 5 4) ); }}
    top    {{ type wall;  faces ( (3 7 6 2) ); }}
    sides  {{ type wall;  faces ( (0 3 2 1) (4 5 6 7) ); }}
);
mergePatchPairs ();
"""


# ---------------- fvOptions (bound the transient) ----------------
def fvOptions_cp(p):
    tlo, thi = p["T_in"] - 2, p["T_in"] + 500      # heat flux can raise T a lot; cap catches runaway
    return hdr("dictionary", "fvOptions") + f"""
limitT
{{ type limitTemperature; active true; selectionMode all; min {tlo}; max {thi}; }}
limitU
{{ type limitVelocity; active true; selectionMode all; max 50; }}
"""


# ---------------- 0/ fields ----------------
def fields(p):
    U, Tin, q = p["u_in"], p["T_in"], p["q"]
    kind = p.get("turb", "kOmegaSST")
    Dh = 2 * p["W"] * p["H"] / (p["W"] + p["H"])
    I, Cmu = 0.05, 0.09
    k = max(1.5 * (I * max(U, 1e-3)) ** 2, 1e-6)
    l = max(0.07 * Dh, 1e-4)                          # turbulence length scale ~ hydraulic diameter
    omega = math.sqrt(k) / (Cmu ** 0.25 * l)
    epsilon = Cmu ** 0.75 * k ** 1.5 / l
    out = {}
    out["U"] = _field("volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)",
        f"    inlet {{ type fixedValue; value uniform ({U} 0 0); }}\n"
        f"    outlet {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}\n"
        f"    base {{ type noSlip; }}\n"
        f"    top {{ type noSlip; }}\n"
        f"    sides {{ type noSlip; }}\n")
    # base carries the chip heat flux; top/sides adiabatic; inlet fixed coolant temp
    out["T"] = _field("volScalarField", "T", "[0 0 0 1 0 0 0]", f"uniform {Tin}",
        f"    inlet {{ type fixedValue; value uniform {Tin}; }}\n"
        f"    outlet {{ type inletOutlet; inletValue uniform {Tin}; value uniform {Tin}; }}\n"
        f"    base {{ type externalWallHeatFluxTemperature; mode flux; q uniform {q}; "
        f"kappaMethod fluidThermo; value uniform {Tin}; }}\n"
        f"    top {{ type zeroGradient; }}\n"
        f"    sides {{ type zeroGradient; }}\n")
    out["p_rgh"] = _field("volScalarField", "p_rgh", "[1 -1 -2 0 0 0 0]", "uniform 1e5",
        f"    inlet {{ type fixedFluxPressure; gradient uniform 0; value uniform 1e5; }}\n"
        f"    outlet {{ type fixedValue; value uniform 1e5; }}\n"
        f"    base {{ type fixedFluxPressure; gradient uniform 0; value uniform 1e5; }}\n"
        f"    top {{ type fixedFluxPressure; gradient uniform 0; value uniform 1e5; }}\n"
        f"    sides {{ type fixedFluxPressure; gradient uniform 0; value uniform 1e5; }}\n")
    out["p"] = _field("volScalarField", "p", "[1 -1 -2 0 0 0 0]", "uniform 1e5",
        "    inlet { type calculated; value uniform 1e5; }\n"
        "    outlet { type calculated; value uniform 1e5; }\n"
        "    base { type calculated; value uniform 1e5; }\n"
        "    top { type calculated; value uniform 1e5; }\n"
        "    sides { type calculated; value uniform 1e5; }\n")
    if kind == "laminar":
        return out
    out["k"] = _field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"uniform {k:.6g}",
        f"    inlet {{ type fixedValue; value uniform {k:.6g}; }}\n"
        f"    outlet {{ type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g}; }}\n"
        f"    base {{ type kqRWallFunction; value uniform {k:.6g}; }}\n"
        f"    top {{ type kqRWallFunction; value uniform {k:.6g}; }}\n"
        f"    sides {{ type kqRWallFunction; value uniform {k:.6g}; }}\n")
    if kind == "kOmegaSST":
        out["omega"] = _field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", f"uniform {omega:.6g}",
            f"    inlet {{ type fixedValue; value uniform {omega:.6g}; }}\n"
            f"    outlet {{ type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g}; }}\n"
            f"    base {{ type omegaWallFunction; value uniform {omega:.6g}; }}\n"
            f"    top {{ type omegaWallFunction; value uniform {omega:.6g}; }}\n"
            f"    sides {{ type omegaWallFunction; value uniform {omega:.6g}; }}\n")
    else:
        out["epsilon"] = _field("volScalarField", "epsilon", "[0 2 -3 0 0 0 0]", f"uniform {epsilon:.6g}",
            f"    inlet {{ type fixedValue; value uniform {epsilon:.6g}; }}\n"
            f"    outlet {{ type inletOutlet; inletValue uniform {epsilon:.6g}; value uniform {epsilon:.6g}; }}\n"
            f"    base {{ type epsilonWallFunction; value uniform {epsilon:.6g}; }}\n"
            f"    top {{ type epsilonWallFunction; value uniform {epsilon:.6g}; }}\n"
            f"    sides {{ type epsilonWallFunction; value uniform {epsilon:.6g}; }}\n")
    out["nut"] = _field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0",
        "    inlet { type calculated; value uniform 0; }\n"
        "    outlet { type calculated; value uniform 0; }\n"
        "    base { type nutkWallFunction; value uniform 0; }\n"
        "    top { type nutkWallFunction; value uniform 0; }\n"
        "    sides { type nutkWallFunction; value uniform 0; }\n")
    out["alphat"] = _field("volScalarField", "alphat", "[1 -1 -1 0 0 0 0]", "uniform 0",
        "    inlet { type calculated; value uniform 0; }\n"
        "    outlet { type calculated; value uniform 0; }\n"
        "    base { type compressible::alphatWallFunction; value uniform 0; }\n"
        "    top { type compressible::alphatWallFunction; value uniform 0; }\n"
        "    sides { type compressible::alphatWallFunction; value uniform 0; }\n")
    return out


def build(case: Path, p, iters):
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "system" / "blockMeshDict").write_text(blockMeshDict(p))
    (case / "system" / "controlDict").write_text(controlDict(iters))
    (case / "system" / "fvSchemes").write_text(fvSchemes())
    (case / "system" / "fvSolution").write_text(fvSolution_cp())
    (case / "system" / "fvOptions").write_text(fvOptions_cp(p))
    (case / "constant" / "thermophysicalProperties").write_text(thermo(p.get("fluid", "water")))
    (case / "constant" / "turbulenceProperties").write_text(turb(p.get("turb", "kOmegaSST")))
    (case / "constant" / "g").write_text(g_dict())
    z = case / "0"; z.mkdir(exist_ok=True)
    for name, txt in fields(p).items():
        (z / name).write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--iters", type=int, default=800)
    a = ap.parse_args()
    p = json.loads(Path(a.params).read_text())
    build(Path(a.case), p, a.iters)
    print(f"[build_coldplate] case ready: {a.case}")


if __name__ == "__main__":
    main()
