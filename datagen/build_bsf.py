"""Assemble a single-region buoyantSimpleFoam case (SIMSHIFT-style heatsink).

Air box with the heatsink as a heated-wall obstacle carved by snappyHexMesh.
Reuses the exact v2412 buoyantSimpleFoam recipe from the circuitBoardCooling
tutorial (heRhoThermo perfectGas air; p_rgh 0.7 / U 0.3 / h 0.7 relaxation),
switched to kOmegaSST to match SIMSHIFT (turbulentKE + turbulentOmega metadata).

Domain (metres):  x streamwise [0,Lx], y vertical [0,Ly] (g=-y), z span [0,Lz].
Patches: inlet(x-) outlet(x+) floor(y-) farfield(y+,z-,z+) + heatsink(from STL).

Usage:
  python build_bsf.py --case <dir> --params <params.json> --iters 3000
The STL must already be at <case>/constant/triSurface/heatsink.stl.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

HDR = ("FoamFile\n{{\n    version 2.0; format ascii; class {cls}; "
       "object {obj};\n}}\n")
def hdr(cls, obj): return HDR.format(cls=cls, obj=obj)


# ---------------- mesh ----------------
def blockMeshDict(d):
    Lx, Ly, Lz = d["Lx"], d["Ly"], d["Lz"]
    h = d["bg_cell"]                                  # background cell size (m)
    nx, ny, nz = round(Lx / h), round(Ly / h), round(Lz / h)
    v = [(0,0,0),(Lx,0,0),(Lx,Ly,0),(0,Ly,0),(0,0,Lz),(Lx,0,Lz),(Lx,Ly,Lz),(0,Ly,Lz)]
    vs = "\n".join(f"    ({x} {y} {z})" for x, y, z in v)
    return hdr("dictionary", "blockMeshDict") + f"""
scale 1;
vertices
(
{vs}
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet    {{ type patch; faces ( (0 4 7 3) ); }}
    outlet   {{ type patch; faces ( (1 2 6 5) ); }}
    floor    {{ type wall;  faces ( (0 1 5 4) ); }}
    farfield {{ type patch; faces ( (3 7 6 2) (0 3 2 1) (4 5 6 7) ); }}
);
mergePatchPairs ();
"""


def snappyDict(d):
    # refinement box tightly around the heatsink
    m = 0.02
    x0, x1 = d["cx"] - d["L"]/2 - m, d["cx"] + d["L"]/2 + m
    y0, y1 = -0.005, d["t_b"] + d["h_f"] + m
    z0, z1 = d["cz"] - d["W"]/2 - m, d["cz"] + d["W"]/2 + m
    lvl = d["snap_level"]
    nlayers = int(d.get("n_layers", 3))              # prism layers on the heated wall
    px, py, pz = 0.02, d["Ly"] - 0.02, 0.02          # locationInMesh: fluid point
    return hdr("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh true;
snap            true;
addLayers       {"true" if nlayers > 0 else "false"};

geometry
{{
    heatsink.stl {{ type triSurfaceMesh; name heatsink; }}
    refBox {{ type searchableBox; min ({x0} {y0} {z0}); max ({x1} {y1} {z1}); }}
}};

castellatedMeshControls
{{
    maxLocalCells 2000000;
    maxGlobalCells 8000000;
    minRefinementCells 10;
    nCellsBetweenLevels 3;
    maxLoadUnbalance 0.1;
    resolveFeatureAngle 30;
    allowFreeStandingZoneFaces true;
    features ( {{ file "heatsink.eMesh"; level {lvl}; }} );
    refinementSurfaces
    {{
        heatsink {{ level ({lvl} {lvl}); patchInfo {{ type wall; }} }}
    }}
    refinementRegions
    {{
        refBox {{ mode inside; levels ((1e15 {max(lvl-1,1)})); }}
    }}
    locationInMesh ({px} {py} {pz});
}}

snapControls
{{
    nSmoothPatch 3; tolerance 2.0; nSolveIter 50; nRelaxIter 5;
    nFeatureSnapIter 10; implicitFeatureSnap false; explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes true;
    layers {{ heatsink {{ nSurfaceLayers {nlayers}; }} }}
    expansionRatio 1.3;
    finalLayerThickness 0.4; minThickness 0.1;
    nGrow 0; featureAngle 60; nRelaxIter 5;
    nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10;
    maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65; maxBoundarySkewness 20; maxInternalSkewness 4;
    maxConcave 80; minVol 1e-13; minTetQuality 1e-15; minArea -1;
    minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02;
    minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4;
    errorReduction 0.75;
}}

mergeTolerance 1e-6;
"""


def sfeDict():
    return hdr("dictionary", "surfaceFeatureExtractDict") + """
heatsink.stl
{
    extractionMethod extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    writeObj no;
}
"""


# ---------------- physics dicts ----------------
# fluid thermophysical properties (const transport, hConst; ~330 K)
FLUIDS = {
    "air":    dict(eos="perfectGas", mol=28.96, Cp=1004.4, mu=1.831e-5, Pr=0.705, rho=None),
    "water":  dict(eos="rhoConst",   mol=18.0,  Cp=4181.0, mu=8.9e-4,   Pr=6.1,   rho=997.0),
    # engineered liquid coolants
    "oil":    dict(eos="rhoConst",   mol=170.0, Cp=1900.0, mu=0.02,     Pr=292.0, rho=850.0),   # mineral/immersion oil (high Pr)
    "glycol": dict(eos="rhoConst",   mol=62.0,  Cp=3300.0, mu=0.0035,   Pr=29.0,  rho=1070.0),  # 50/50 ethylene-glycol/water
}
LIQUIDS = ("water", "oil", "glycol")

def thermo(fluid="air"):
    f = FLUIDS[fluid]
    # perfectGas derives rho from p (needs no eos block); rhoConst needs an explicit rho
    eos_block = "" if f["rho"] is None else f"    equationOfState {{ rho {f['rho']}; }}\n"
    return hdr("dictionary", "thermophysicalProperties") + f"""
thermoType
{{
    type heRhoThermo; mixture pureMixture; transport const; thermo hConst;
    equationOfState {f['eos']}; specie specie; energy sensibleEnthalpy;
}}
mixture
{{
    specie {{ molWeight {f['mol']}; }}
    thermodynamics {{ Cp {f['Cp']}; Hf 0; }}
    transport {{ mu {f['mu']}; Pr {f['Pr']}; }}
{eos_block}}}
"""

def turb(kind="kOmegaSST"):
    if kind == "laminar":
        return hdr("dictionary", "turbulenceProperties") + "\nsimulationType laminar;\n"
    return hdr("dictionary", "turbulenceProperties") + \
        f"\nsimulationType RAS;\nRAS {{ RASModel {kind}; turbulence on; printCoeffs on; }}\n"

def g_dict():
    return (hdr("uniformDimensionedVectorField", "g") +
            "\ndimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);\n")


# ---------------- system ----------------
def controlDict(iters):
    return hdr("dictionary", "controlDict") + f"""
application buoyantSimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {iters};
deltaT 1; writeControl timeStep; writeInterval {iters}; purgeWrite 1;
writeFormat ascii; writePrecision 7; writeCompression off;
timeFormat general; runTimeModifiable true;
"""

def fvSchemes():
    return hdr("dictionary", "fvSchemes") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,K)      bounded Gauss upwind;
    div(phi,h)      bounded Gauss upwind;
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,Ekp)    bounded Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
"""

def fvOptions(p):
    # bound T and |U| during the transient. Floor at ambient: the fluid can't be
    # colder than its coldest boundary (inlet=ambient), so clamp there to kill the
    # small numerical sub-ambient undershoot seen on high-aspect layer cells.
    tlo, thi = p["T_amb"], p["T_wall"] + 20
    return hdr("dictionary", "fvOptions") + f"""
limitT
{{ type limitTemperature; active true; selectionMode all; min {tlo}; max {thi}; }}
limitU
{{ type limitVelocity; active true; selectionMode all; max 50; }}
"""

def fvSolution():
    return hdr("dictionary", "fvSolution") + """
solvers
{
    p_rgh { solver GAMG; tolerance 1e-7; relTol 0.01; smoother DICGaussSeidel; }
    "(U|h|k|epsilon|omega)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-7; relTol 0.01; }
}
SIMPLE
{
    momentumPredictor no; nNonOrthogonalCorrectors 1; pRefCell 0; pRefValue 0;
    // U not listed: momentumPredictor=no means U is never solved as a predictor,
    // so a U residual would never exist and convergence could never trip.
    residualControl { p_rgh 1e-3; h 1e-3; "(k|epsilon|omega)" 1e-2; }
}
relaxationFactors
{
    rho 1.0; p_rgh 0.3; U 0.2; h 0.5; "(k|epsilon|omega)" 0.3;
}
"""


# ---------------- 0/ fields ----------------
def _field(cls, obj, dims, internal, bf):
    return (hdr(cls, obj) + f"\ndimensions {dims};\ninternalField {internal};\n"
            f"boundaryField\n{{\n{bf}}}\n")

def fields(p):
    U, Tamb, Tw = p["u_in"], p["T_amb"], p["T_wall"]
    kind = p.get("turb", "kOmegaSST")
    I, Cmu = 0.05, 0.09
    k = max(1.5 * (I * max(U, 1e-3)) ** 2, 1e-4)
    # turbulence length scale from the tallest geometry feature (fins, else base/plate/cube),
    # floored so finless shapes (h_f=0) don't divide by zero
    l = 0.1 * max(p.get("h_f", 0.0), p.get("t_b", 0.0), 0.005)
    omega = math.sqrt(k) / (Cmu ** 0.25 * l)
    epsilon = Cmu ** 0.75 * k ** 1.5 / l
    out = {}
    out["U"] = _field("volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)",
        f"    inlet {{ type fixedValue; value uniform ({U} 0 0); }}\n"
        f"    outlet {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}\n"
        f"    floor {{ type noSlip; }}\n"
        f"    farfield {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}\n"
        f"    heatsink {{ type noSlip; }}\n")
    out["T"] = _field("volScalarField", "T", "[0 0 0 1 0 0 0]", f"uniform {Tamb}",
        f"    inlet {{ type fixedValue; value uniform {Tamb}; }}\n"
        f"    outlet {{ type inletOutlet; inletValue uniform {Tamb}; value uniform {Tamb}; }}\n"
        f"    floor {{ type zeroGradient; }}\n"
        f"    farfield {{ type inletOutlet; inletValue uniform {Tamb}; value uniform {Tamb}; }}\n"
        f"    heatsink {{ type fixedValue; value uniform {Tw}; }}\n")
    out["p_rgh"] = _field("volScalarField", "p_rgh", "[1 -1 -2 0 0 0 0]", "uniform 1e5",
        f"    inlet {{ type fixedFluxPressure; gradient uniform 0; value uniform 1e5; }}\n"
        f"    outlet {{ type fixedValue; value uniform 1e5; }}\n"
        f"    floor {{ type fixedFluxPressure; gradient uniform 0; value uniform 1e5; }}\n"
        f"    farfield {{ type prghPressure; p uniform 1e5; value uniform 1e5; }}\n"
        f"    heatsink {{ type fixedFluxPressure; gradient uniform 0; value uniform 1e5; }}\n")
    out["p"] = _field("volScalarField", "p", "[1 -1 -2 0 0 0 0]", "uniform 1e5",
        "    inlet { type calculated; value uniform 1e5; }\n"
        "    outlet { type calculated; value uniform 1e5; }\n"
        "    floor { type calculated; value uniform 1e5; }\n"
        "    farfield { type calculated; value uniform 1e5; }\n"
        "    heatsink { type calculated; value uniform 1e5; }\n")
    if kind == "laminar":
        return out                                   # no turbulence fields
    out["k"] = _field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"uniform {k:.6g}",
        f"    inlet {{ type fixedValue; value uniform {k:.6g}; }}\n"
        f"    outlet {{ type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g}; }}\n"
        f"    floor {{ type kqRWallFunction; value uniform {k:.6g}; }}\n"
        f"    farfield {{ type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g}; }}\n"
        f"    heatsink {{ type kqRWallFunction; value uniform {k:.6g}; }}\n")
    if kind == "kOmegaSST":
        out["omega"] = _field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", f"uniform {omega:.6g}",
            f"    inlet {{ type fixedValue; value uniform {omega:.6g}; }}\n"
            f"    outlet {{ type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g}; }}\n"
            f"    floor {{ type omegaWallFunction; value uniform {omega:.6g}; }}\n"
            f"    farfield {{ type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g}; }}\n"
            f"    heatsink {{ type omegaWallFunction; value uniform {omega:.6g}; }}\n")
    else:  # kEpsilon
        out["epsilon"] = _field("volScalarField", "epsilon", "[0 2 -3 0 0 0 0]", f"uniform {epsilon:.6g}",
            f"    inlet {{ type fixedValue; value uniform {epsilon:.6g}; }}\n"
            f"    outlet {{ type inletOutlet; inletValue uniform {epsilon:.6g}; value uniform {epsilon:.6g}; }}\n"
            f"    floor {{ type epsilonWallFunction; value uniform {epsilon:.6g}; }}\n"
            f"    farfield {{ type inletOutlet; inletValue uniform {epsilon:.6g}; value uniform {epsilon:.6g}; }}\n"
            f"    heatsink {{ type epsilonWallFunction; value uniform {epsilon:.6g}; }}\n")
    out["nut"] = _field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0",
        "    inlet { type calculated; value uniform 0; }\n"
        "    outlet { type calculated; value uniform 0; }\n"
        "    floor { type nutkWallFunction; value uniform 0; }\n"
        "    farfield { type calculated; value uniform 0; }\n"
        "    heatsink { type nutkWallFunction; value uniform 0; }\n")
    out["alphat"] = _field("volScalarField", "alphat", "[1 -1 -1 0 0 0 0]", "uniform 0",
        "    inlet { type calculated; value uniform 0; }\n"
        "    outlet { type calculated; value uniform 0; }\n"
        "    floor { type compressible::alphatWallFunction; value uniform 0; }\n"
        "    farfield { type calculated; value uniform 0; }\n"
        "    heatsink { type compressible::alphatWallFunction; value uniform 0; }\n")
    return out


def build(case: Path, p, iters):
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "system" / "blockMeshDict").write_text(blockMeshDict(p))
    (case / "system" / "snappyHexMeshDict").write_text(snappyDict(p))
    (case / "system" / "surfaceFeatureExtractDict").write_text(sfeDict())
    (case / "system" / "controlDict").write_text(controlDict(iters))
    (case / "system" / "fvSchemes").write_text(fvSchemes())
    (case / "system" / "fvSolution").write_text(fvSolution())
    (case / "system" / "fvOptions").write_text(fvOptions(p))
    (case / "constant" / "thermophysicalProperties").write_text(thermo(p.get("fluid", "air")))
    (case / "constant" / "turbulenceProperties").write_text(turb(p.get("turb", "kOmegaSST")))
    (case / "constant" / "g").write_text(g_dict())
    z = case / "0"; z.mkdir(exist_ok=True)
    for name, txt in fields(p).items():
        (z / name).write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    a = ap.parse_args()
    p = json.loads(Path(a.params).read_text())
    build(Path(a.case), p, a.iters)
    print(f"[build_bsf] case ready: {a.case}")


if __name__ == "__main__":
    main()
