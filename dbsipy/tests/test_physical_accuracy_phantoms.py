from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nb
import pytest
import torch

from dbsipy.core.fast_DBSI import DBSIpy
from dbsipy.misc.models.Linear_Models import NODDIOptimizer


@dataclass(frozen=True)
class _Paths:
    dwi: Path
    mask: Path
    bval: Path
    bvec: Path
    cfg: Path


def _unit_vectors(n: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3)).astype(np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v.astype(np.float32)


def _abcd_like_acquisition(*, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return an ABCD-like multi-shell acquisition (bvals, bvecs).

    This is intentionally small for unit testing while still resembling a
    modern multi-shell protocol.

    Shells: b=0 plus {500, 1000, 2000, 3000} s/mm^2.
    """

    n_b0 = 3
    shells: list[tuple[int, int]] = [
        (500, 6),
        (1000, 10),
        (2000, 10),
        (3000, 15),
    ]

    bvals_parts: list[np.ndarray] = [np.zeros((n_b0,), dtype=np.int32)]
    bvecs_parts: list[np.ndarray] = [np.zeros((n_b0, 3), dtype=np.float32)]

    for i, (b, n_dir) in enumerate(shells):
        bvals_parts.append(np.full((n_dir,), int(b), dtype=np.int32))
        bvecs_parts.append(_unit_vectors(int(n_dir), seed=int(seed) + 11 * (i + 1)))

    bvals = np.concatenate(bvals_parts)
    bvecs = np.vstack(bvecs_parts)
    assert bvecs.shape[0] == bvals.shape[0]
    return bvals, bvecs


def _write_bvals_bvecs(out_bval: Path, out_bvec: Path, bvals: np.ndarray, bvecs: np.ndarray) -> None:
    out_bval.write_text(" ".join(str(int(x)) for x in bvals.tolist()) + "\n", encoding="utf-8")

    # DBSIpy loader supports 3xN or Nx3; write 3xN (classic FSL).
    bvec_3xn = bvecs.T
    lines = [" ".join(f"{float(x):.8f}" for x in row.tolist()) for row in bvec_3xn]
    out_bvec.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dwi_and_mask(tmp_path: Path, signal: np.ndarray) -> tuple[Path, Path]:
    """Write a tiny 1x1x1xN DWI and a 1x1x1 mask."""
    assert signal.ndim == 1
    data = signal.astype(np.float32)[None, None, None, :]
    affine = np.eye(4, dtype=np.float32)

    dwi_path = tmp_path / "dwi.nii.gz"
    nb.save(nb.Nifti1Image(data, affine), str(dwi_path))

    mask_path = tmp_path / "mask.nii.gz"
    nb.save(nb.Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), affine), str(mask_path))

    return dwi_path, mask_path


def _make_cfg_from_template(tmp_path: Path, *, engine: str, paths: _Paths, overrides: dict[str, dict[str, str]] | None = None) -> None:
    template = Path(__file__).resolve().parents[1] / "configs" / "Template_All_Engines_Minimal.ini"
    cfg = configparser.ConfigParser()
    cfg.read(template, encoding="utf-8")

    cfg["INPUT"]["dwi_file"] = str(paths.dwi)
    cfg["INPUT"]["mask_file"] = str(paths.mask)
    cfg["INPUT"]["bval_file"] = str(paths.bval)
    cfg["INPUT"]["bvec_file"] = str(paths.bvec)

    cfg["GLOBAL"]["model_engine"] = str(engine)

    # Keep tests deterministic + fast.
    cfg["DEVICE"]["DEVICE"] = "cpu"
    cfg["DEVICE"]["HOST"] = "cpu"
    cfg["DEBUG"]["verbose"] = "False"

    if overrides:
        for section, section_overrides in overrides.items():
            if not cfg.has_section(section):
                cfg.add_section(section)
            for k, v in section_overrides.items():
                cfg[section][k] = str(v)

    with open(paths.cfg, "w", encoding="utf-8") as f:
        cfg.write(f)


def _run_pipeline(cfg_path: Path) -> DBSIpy:
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path, encoding="utf-8")
    dbsi = DBSIpy(cfg)
    dbsi.load()
    dbsi.calc()
    return dbsi


def _simulate_dti_signal(bvals: np.ndarray, bvecs: np.ndarray, *, s0: float, evals_mm2_s: tuple[float, float, float]) -> np.ndarray:
    """Simulate single-tensor signal with principal axes aligned to x/y/z."""
    l1, l2, l3 = evals_mm2_s
    D = np.diag([l1, l2, l3]).astype(np.float64)
    g = bvecs.astype(np.float64)
    b = bvals.astype(np.float64)

    # Signal attenuation: exp(-b * g^T D g)
    gtDg = np.einsum("ni,ij,nj->n", g, D, g)
    atten = np.exp(-b * gtDg)
    return (float(s0) * atten).astype(np.float32)


def _fa_from_evals(evals: tuple[float, float, float]) -> float:
    l1, l2, l3 = (float(x) for x in evals)
    md = (l1 + l2 + l3) / 3.0
    num = (l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2
    den = l1**2 + l2**2 + l3**2
    if den <= 0:
        return 0.0
    return float(np.sqrt(1.5 * num / den))


def _simulate_noddi_signal(
    bvals_s_mm2: np.ndarray,
    bvecs: np.ndarray,
    *,
    s0: float,
    v_ic: float,
    v_iso: float,
    od: float,
    direction_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0),
    d_ic: float = 1.7e-3,
    d_iso: float = 3.0e-3,
    d_ec_par_init: float = 1.5e-3,
) -> np.ndarray:
    """Simulate NODDI attenuation using the same differentiable forward model used in fitting."""

    # Convert a direction vector into (theta, phi).
    n = np.asarray(direction_xyz, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    theta = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    phi = float(np.arctan2(n[1], n[0]))

    bvals = torch.from_numpy(bvals_s_mm2.astype(np.float32))
    bvecs_t = torch.from_numpy(bvecs.astype(np.float32))

    init = {
        "v_ic": torch.tensor([float(v_ic)], dtype=torch.float32),
        "v_iso": torch.tensor([float(v_iso)], dtype=torch.float32),
        "od": torch.tensor([float(od)], dtype=torch.float32),
        "d_ec_par": torch.tensor([float(d_ec_par_init)], dtype=torch.float32),
        "theta": torch.tensor([theta], dtype=torch.float32),
        "phi": torch.tensor([phi], dtype=torch.float32),
    }

    model = NODDIOptimizer(
        n_voxels=1,
        initial_params=init,
        bvals=bvals,
        bvecs=bvecs_t,
        watson_lut=None,
        d_ic=float(d_ic),
        d_iso=float(d_iso),
        use_tortuosity=True,
        device="cpu",
    )

    with torch.no_grad():
        atten = model(voxel_idx=None).detach().cpu().numpy().reshape(-1)

    return (float(s0) * atten).astype(np.float32)


@pytest.mark.accuracy
def test_dti_adam_recovers_tensor_metrics(tmp_path: Path) -> None:
    # Acquisition: ABCD-like multi-shell (DTI uses b < dti_bval_cutoff).
    bvals, bvecs = _abcd_like_acquisition(seed=1)

    # Ground truth tensor in mm^2/s.
    evals = (1.7e-3, 0.3e-3, 0.3e-3)
    s0 = 1000.0
    signal = _simulate_dti_signal(bvals, bvecs, s0=s0, evals_mm2_s=evals)

    dwi_path, mask_path = _write_dwi_and_mask(tmp_path, signal)
    bval_path = tmp_path / "bvals"
    bvec_path = tmp_path / "bvecs"
    _write_bvals_bvecs(bval_path, bvec_path, bvals, bvecs)

    cfg_path = tmp_path / "cfg.ini"
    paths = _Paths(dwi=dwi_path, mask=mask_path, bval=bval_path, bvec=bvec_path, cfg=cfg_path)

    _make_cfg_from_template(
        tmp_path,
        engine="DTI",
        paths=paths,
        overrides={
            "DTI": {
                "dti_fit_method": "ADAM",
                # Enough iterations to converge on a trivial 1-voxel phantom.
                "dti_lr": "0.001",
                "dti_epochs": "200",
            }
        },
    )

    dbsi = _run_pipeline(cfg_path)

    ad_hat = float(dbsi.params["dti_axial"].pmap[0, 0].detach().cpu().item())
    rd_hat = float(dbsi.params["dti_radial"].pmap[0, 0].detach().cpu().item())
    fa_hat = float(dbsi.params["dti_fa"].pmap[0, 0].detach().cpu().item())

    # Convert ground truth to um^2/ms (DBSIpy convention).
    ad_true = float(evals[0] * 1e3)
    rd_true = float(((evals[1] + evals[2]) / 2.0) * 1e3)
    fa_true = _fa_from_evals(tuple(x * 1e3 for x in evals))

    # ADAM-based DTI fitting is iterative and may not hit the exact closed-form
    # solution even in noise-free settings. Keep tolerances wide enough to avoid
    # spurious failures while still catching major regressions.
    assert abs(ad_hat - ad_true)/ad_true < 0.05
    assert abs(rd_hat - rd_true)/rd_true < 0.05
    assert abs(fa_hat - fa_true)/fa_true < 0.05


@pytest.mark.accuracy
def test_noddi_recovers_basic_fractions(tmp_path: Path) -> None:
    # ABCD-like multi-shell helps the optimizer distinguish compartments.
    bvals, bvecs = _abcd_like_acquisition(seed=2)

    # Ground truth NODDI params.
    s0 = 1000.0
    v_ic = 0.60
    v_iso = 0.10
    od = 0.20
    ndi_true = (1.0 - v_iso) * v_ic

    signal = _simulate_noddi_signal(bvals, bvecs, s0=s0, v_ic=v_ic, v_iso=v_iso, od=od)

    dwi_path, mask_path = _write_dwi_and_mask(tmp_path, signal)
    bval_path = tmp_path / "bvals"
    bvec_path = tmp_path / "bvecs"
    _write_bvals_bvecs(bval_path, bvec_path, bvals, bvecs)

    cfg_path = tmp_path / "cfg.ini"
    paths = _Paths(dwi=dwi_path, mask=mask_path, bval=bval_path, bvec=bvec_path, cfg=cfg_path)

    _make_cfg_from_template(
        tmp_path,
        engine="NODDI",
        paths=paths,
        overrides={
            "NODDI": {
                "noddi_lr": "0.002",
                "noddi_epochs": "500",
                "noddi_d_ic": "0.0017",
                "noddi_d_iso": "0.003",
                "noddi_use_tortuosity": "False",
            },
            "GLOBAL": {
                # NODDI expects attenuation-like fitting; keep the default semantics explicit.
                "signal_normalization": "max",
            },
        },
    )

    dbsi = _run_pipeline(cfg_path)

    ndi_hat = float(dbsi.params["noddi_ndi"].pmap[0, 0].detach().cpu().item())
    fiso_hat = float(dbsi.params["noddi_fiso"].pmap[0, 0].detach().cpu().item())
    odi_hat = float(dbsi.params["noddi_odi"].pmap[0, 0].detach().cpu().item())

    # Keep tolerances modest: this is a nonconvex fit and is intentionally optional.
    assert (abs(ndi_hat - ndi_true)/ndi_true) < 0.05
    assert (abs(fiso_hat - v_iso)/v_iso) < 0.1
    assert (abs(odi_hat - od)/od) < 0.05


@pytest.mark.accuracy
@pytest.mark.parametrize("engine", ["DBSI", "IA"])
def test_dbsi_family_isotropic_phantom_is_mostly_isotropic(tmp_path: Path, engine: str) -> None:
    # Simple isotropic free-water phantom: signal depends on b only.
    bvals, bvecs = _abcd_like_acquisition(seed=4)

    s0 = 1000.0
    d_iso = 3.0e-3  # mm^2/s (free water)
    signal = (s0 * np.exp(-bvals.astype(np.float64) * d_iso)).astype(np.float32)

    dwi_path, mask_path = _write_dwi_and_mask(tmp_path, signal)
    bval_path = tmp_path / "bvals"
    bvec_path = tmp_path / "bvecs"
    _write_bvals_bvecs(bval_path, bvec_path, bvals, bvecs)

    cfg_path = tmp_path / "cfg.ini"
    paths = _Paths(dwi=dwi_path, mask=mask_path, bval=bval_path, bvec=bvec_path, cfg=cfg_path)

    _make_cfg_from_template(
        tmp_path,
        engine=engine,
        paths=paths,
        overrides={
            "GLOBAL": {
                "max_group_number": "1",
                "fiber_threshold": "0.01",
                "signal_normalization": "max",
            },
            "OPTIMIZER": {
                # Keep runtime bounded; 1 voxel should converge quickly.
                "step_1_epochs": "120",
                "step_2_epochs": "200",
                "step_2_patience": "40",
                "step_2_min_delta": "1e-6",
            },
            "STEP_1": {
                "step_1_axial": "0.0017",
                "step_1_radial": "0.0002",
            },
        },
    )

    dbsi = _run_pipeline(cfg_path)

    # Expect most signal to be assigned to isotropic compartments.
    iso_frac = float(dbsi.params["isotropic_fraction"].pmap[0, 0].detach().cpu().item())
    fiber_frac = float(dbsi.params["fiber_0d_fraction"].pmap[0, 0].detach().cpu().item())

    assert iso_frac > 0.90
    assert fiber_frac < 0.10


def _simulate_single_fiber_plus_isotropic(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    *,
    s0: float,
    fiber_fraction: float,
    evals_fiber_mm2_s: tuple[float, float, float],
    d_iso_mm2_s: float,
) -> np.ndarray:
    """Simulate a simple 2-compartment voxel: single-fiber tensor + isotropic diffusion."""

    fiber_fraction = float(np.clip(fiber_fraction, 0.0, 1.0))
    iso_fraction = 1.0 - fiber_fraction

    s_fiber = _simulate_dti_signal(bvals, bvecs, s0=s0, evals_mm2_s=evals_fiber_mm2_s).astype(np.float64)
    s_iso = (float(s0) * np.exp(-bvals.astype(np.float64) * float(d_iso_mm2_s))).astype(np.float64)

    s = fiber_fraction * s_fiber + iso_fraction * s_iso
    return s.astype(np.float32)


@pytest.mark.accuracy
@pytest.mark.parametrize("engine", ["DBSI", "IA"])
def test_dbsi_family_single_fiber_phantom_is_mostly_fiber(tmp_path: Path, engine: str) -> None:
    # Acquisition: ABCD-like multi-shell.
    bvals, bvecs = _abcd_like_acquisition(seed=5)

    # Ground truth: strong single-fiber tensor + small isotropic fraction.
    s0 = 1000.0
    fiber_fraction = 0.85
    evals = (1.7e-3, 0.3e-3, 0.3e-3)  # mm^2/s
    d_iso = 3.0e-3  # mm^2/s

    signal = _simulate_single_fiber_plus_isotropic(
        bvals,
        bvecs,
        s0=s0,
        fiber_fraction=fiber_fraction,
        evals_fiber_mm2_s=evals,
        d_iso_mm2_s=d_iso,
    )

    dwi_path, mask_path = _write_dwi_and_mask(tmp_path, signal)
    bval_path = tmp_path / "bvals"
    bvec_path = tmp_path / "bvecs"
    _write_bvals_bvecs(bval_path, bvec_path, bvals, bvecs)

    cfg_path = tmp_path / "cfg.ini"
    paths = _Paths(dwi=dwi_path, mask=mask_path, bval=bval_path, bvec=bvec_path, cfg=cfg_path)

    _make_cfg_from_template(
        tmp_path,
        engine=engine,
        paths=paths,
        overrides={
            "GLOBAL": {
                "max_group_number": "1",
                "fiber_threshold": "0.01",
                "signal_normalization": "max",
            },
            "OPTIMIZER": {
                # Keep runtime bounded; 1 voxel should converge quickly.
                "step_1_epochs": "150",
                "step_2_epochs": "250",
                "step_2_patience": "50",
                "step_2_min_delta": "1e-6",
            },
            "STEP_1": {
                "step_1_axial": "0.0017",
                "step_1_radial": "0.0003",
            },
        },
    )

    dbsi = _run_pipeline(cfg_path)

    fiber_frac_hat = float(dbsi.params["fiber_0d_fraction"].pmap[0, 0].detach().cpu().item())
    iso_frac_hat = float(dbsi.params["isotropic_fraction"].pmap[0, 0].detach().cpu().item())
    fa_hat = float(dbsi.params["fiber_0d_fa"].pmap[0, 0].detach().cpu().item())

    # These are approximate: DBSI/IA solves a nonconvex optimization and this is opt-in.
    assert fiber_frac_hat > 0.65
    assert iso_frac_hat < 0.35
    assert fa_hat > 0.50

    if engine == "IA":
        ia_frac = float(dbsi.params["fiber_0d_IA_fraction"].pmap[0, 0].detach().cpu().item())
        ea_frac = float(dbsi.params["fiber_0d_EA_fraction"].pmap[0, 0].detach().cpu().item())
        assert ia_frac >= 0.0
        assert ea_frac >= 0.0
        # IA/EA should decompose the same fiber compartment.
        assert abs((ia_frac + ea_frac) - fiber_frac_hat) < 0.15
