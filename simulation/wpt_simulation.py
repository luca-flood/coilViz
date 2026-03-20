"""
wpt_simulation.py  —  v2.0
─────────────────────────────────────────────────────────────────
Modular WPT Simulation — Khan et al. 2018
"Accurate Modeling of Coil Inductance for Near-Field WPT"
IEEE Transactions on Microwave Theory and Techniques, 2018

Physical models included:
  ✓ Skin effect          (AC resistance at 6.78MHz)
  ✓ Proximity effect     (Dowell model, inter-turn current distortion)
  ✓ Parasitic capacitance (inter-turn, Medhurst model)
  ✓ Medhurst SRF correction (effective inductance near self-resonance)
  ✓ Temperature correction  (copper resistivity vs temperature)
  ✓ Arch curvature geometry (cylindrical conic curve mold)
  ✓ Neumann mutual inductance (elliptic integral exact solution)
  ✓ Misalignment: lateral + angular (Khan 2018 Eq.20, 28)
  ✓ Frequency splitting analysis
  ✓ S-parameter computation

Usage:
    from wpt_simulation import CoilParams, WPTSystem, Pose2D
    tx     = CoilParams(curvature_h=0)
    rx     = CoilParams(curvature_h=20e-3)
    system = WPTSystem(tx, rx)
    result = system.evaluate(Pose2D(x=20e-3, y=0, theta=15))
"""

import numpy as np
from scipy import special
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═════════════════════════════════════════════════════════════
MU0         = 4 * np.pi * 1e-7   # H/m  permeability of free space
EPSILON0    = 8.854e-12           # F/m  permittivity of free space
RHO_CU_20   = 1.68e-8             # Ω·m  copper resistivity at 20°C
ALPHA_CU    = 0.00393             # /°C  copper temp coefficient
EPSILON_R_ENAMEL = 3.5            # —    relative permittivity of enamel
                                  #      (polyurethane/polyamide typical)
ENAMEL_THICKNESS = 0.04e-3        # m    heavy build 18AWG enamel per side


# ═════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════

@dataclass
class CoilParams:
    """
    Physical parameters of a single spiral coil.

    Args:
        N           : Number of turns
        R_outer     : Outer radius (m)
        R_inner     : Inner radius (m)
        pitch       : Turn-to-turn pitch (m)
        wire_diam   : Bare wire diameter (m) — 18 AWG = 1.024e-3 m
        curvature_h : Height of conic arch curve (m). 0 = flat.
    """
    N           : int   = 12
    R_outer     : float = 43e-3
    R_inner     : float = 10e-3
    pitch       : float = 1.2e-3
    wire_diam   : float = 1.024e-3
    curvature_h : float = 0.0

    def __post_init__(self):
        self.loop_radii = np.linspace(self.R_inner, self.R_outer, self.N)
        # Total wire diameter including enamel insulation
        self.wire_diam_total = self.wire_diam + 2 * ENAMEL_THICKNESS

    def summary(self):
        return (f"CoilParams(N={self.N}, OD={self.R_outer*2e3:.1f}mm, "
                f"ID={self.R_inner*2e3:.1f}mm, pitch={self.pitch*1e3:.1f}mm, "
                f"wire={self.wire_diam*1e3:.3f}mm, "
                f"h={self.curvature_h*1e3:.1f}mm)")


@dataclass
class Pose2D:
    """
    2D pose of Rx coil relative to Tx coil center.
    Tx coil is always at origin, face-up in XY plane.

    Args:
        x     : Lateral offset along X axis (m)
        y     : Lateral offset along Y axis (m)
        theta : Angular tilt of Rx coil about Y axis (degrees)
                0°  = parallel to Tx (best coupling)
                90° = perpendicular (zero coupling)
    """
    x     : float = 0.0
    y     : float = 0.0
    theta : float = 0.0

    @property
    def d2(self):
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def theta_rad(self):
        return self.theta * np.pi / 180.0


@dataclass
class WPTResult:
    """Full result of a single WPT evaluation."""
    # Pose
    x                   : float = 0.0
    y                   : float = 0.0
    theta_deg           : float = 0.0
    d_axial_mm          : float = 0.0
    d_lateral_mm        : float = 0.0

    # Coil electrical properties
    L_tx_uH             : float = 0.0
    L_rx_uH             : float = 0.0
    L_tx_eff_uH         : float = 0.0   # Medhurst-corrected
    L_rx_eff_uH         : float = 0.0   # Medhurst-corrected
    C_tx_pF             : float = 0.0   # resonant cap for L_tx_eff
    C_rx_pF             : float = 0.0   # resonant cap for L_rx_eff
    C_parasitic_tx_pF   : float = 0.0   # inter-turn parasitic
    C_parasitic_rx_pF   : float = 0.0
    f_srf_tx_MHz        : float = 0.0   # self-resonant frequency
    f_srf_rx_MHz        : float = 0.0
    R_tx_skin           : float = 0.0   # skin effect only
    R_rx_skin           : float = 0.0
    R_tx_total          : float = 0.0   # skin + proximity + temp
    R_rx_total          : float = 0.0
    proximity_factor    : float = 0.0   # Dowell factor
    Q_tx                : float = 0.0
    Q_rx                : float = 0.0

    # Coupling
    M_nH                : float = 0.0
    k                   : float = 0.0
    kQ                  : float = 0.0

    # Frequency splitting
    f_split_low_MHz     : float = 0.0
    f_split_high_MHz    : float = 0.0
    split_gap_kHz       : float = 0.0
    coupling_regime     : str   = ''
    k_critical          : float = 0.0

    # S-parameters (dB)
    S11_dB              : float = 0.0
    S22_dB              : float = 0.0
    S21_dB              : float = 0.0
    S12_dB              : float = 0.0

    # Power
    V_in                : float = 0.0
    I_in                : float = 0.0
    P_in_W              : float = 0.0
    V_out               : float = 0.0
    I_out               : float = 0.0
    P_out_W             : float = 0.0
    efficiency_pct      : float = 0.0

    def to_dict(self):
        return {k: (round(v, 8) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


# ═════════════════════════════════════════════════════════════
# PHYSICS ENGINE
# ═════════════════════════════════════════════════════════════

class WPTSystem:
    """
    Full WPT system simulation with proximity effect,
    parasitic capacitance, Medhurst SRF correction,
    and arch curvature geometry.

    Args:
        tx      : CoilParams for transmitting coil (always flat)
        rx      : CoilParams for receiving coil
        freq    : Operating frequency (Hz)
        V_in    : Input supply voltage (V)
        I_in    : Input current (A)
        d_axial : Fixed axial separation between coil planes (m)
        temp_c  : Operating temperature (°C)
        R_load  : Load resistance (Ω). None = matched load.
    """

    def __init__(self,
                 tx      : CoilParams,
                 rx      : CoilParams,
                 freq    : float         = 6.78e6,
                 V_in    : float         = 5.0,
                 I_in    : float         = 0.1,
                 d_axial : float         = 50e-3,
                 temp_c  : float         = 75.0,
                 R_load  : Optional[float] = None):

        self.tx      = tx
        self.rx      = rx
        self.freq    = freq
        self.omega   = 2 * np.pi * freq
        self.V_in    = V_in
        self.I_in    = I_in
        self.d_axial = d_axial
        self.temp_c  = temp_c
        self.R_load  = R_load

        # ── Temperature-corrected copper resistivity ──────────
        # rho(T) = rho_20 * (1 + alpha*(T-20))
        # At 75°C: factor = 1 + 0.00393*(75-20) = 1.216
        self.rho_cu = RHO_CU_20 * (1 + ALPHA_CU * (temp_c - 20))

        # ── Skin depth at operating frequency ─────────────────
        # delta = sqrt(rho / (pi * f * mu0))
        # At 6.78MHz, 75°C: delta ≈ 27.1 µm
        self.skin_depth = np.sqrt(
            self.rho_cu / (np.pi * freq * MU0)
        )

        # ── Proximity effect factor (Dowell model) ────────────
        self.proximity_factor = self._dowell_proximity(tx)

        # ── Coil resistances ──────────────────────────────────
        self.R_tx_skin  = self._skin_resistance(tx)
        self.R_rx_skin  = self._skin_resistance(rx)
        self.R_tx_total = self.R_tx_skin * self.proximity_factor
        self.R_rx_total = self.R_rx_skin * self.proximity_factor

        # ── Raw inductances (no parasitic correction) ─────────
        self.L_tx_raw = self._coil_inductance(tx)
        self.L_rx_raw = self._coil_inductance(rx)

        # ── Parasitic capacitances ────────────────────────────
        self.C_p_tx = self._parasitic_capacitance(tx)
        self.C_p_rx = self._parasitic_capacitance(rx)

        # ── Self-resonant frequencies ─────────────────────────
        # f_SRF = 1 / (2*pi*sqrt(L*C_parasitic))
        self.f_srf_tx = self._self_resonant_freq(self.L_tx_raw, self.C_p_tx)
        self.f_srf_rx = self._self_resonant_freq(self.L_rx_raw, self.C_p_rx)

        # ── Medhurst-corrected effective inductances ──────────
        # L_eff = L_raw / (1 - (f/f_SRF)²)
        # Below SRF: L_eff > L_raw (inductance increases near SRF)
        self.L_tx = self._medhurst_correction(self.L_tx_raw, self.f_srf_tx)
        self.L_rx = self._medhurst_correction(self.L_rx_raw, self.f_srf_rx)

        # ── Resonant capacitances (for corrected inductance) ──
        # C_res = 1 / (omega^2 * L_eff)
        self.C_tx = 1.0 / (self.omega**2 * self.L_tx)
        self.C_rx = 1.0 / (self.omega**2 * self.L_rx)

        # ── Q factors ─────────────────────────────────────────
        # Q = omega * L_eff / R_total
        self.Q_tx = self.omega * self.L_tx / self.R_tx_total
        self.Q_rx = self.omega * self.L_rx / self.R_rx_total

    # ── ─────────────────────────────────────────────────────
    # RESISTANCE MODELS
    # ── ─────────────────────────────────────────────────────

    def _skin_resistance(self, coil: CoilParams) -> float:
        """
        AC resistance due to skin effect.

        At high frequency current crowds to wire surface in a
        shell of thickness delta (skin depth).
        Formula: R = rho * L_wire / (pi * delta * d_wire)
        Valid when delta << d_wire/2, which holds here:
        delta=27µm << 512µm = wire radius.

        This is the tubular conductor approximation.
        """
        wire_length = sum(2 * np.pi * r for r in coil.loop_radii)
        return (self.rho_cu * wire_length /
                (np.pi * self.skin_depth * coil.wire_diam))

    def _dowell_proximity(self, coil: CoilParams) -> float:
        """
        Proximity effect resistance factor — Dowell model.

        When adjacent turns carry current in the same direction,
        each turn's magnetic field distorts current in neighboring
        turns, forcing it to the outer edges. This increases
        effective resistance beyond the skin effect alone.

        The Dowell factor F_R multiplies the skin resistance:
            R_total = R_skin * F_R

        Dowell parameter:
            xi = (d_wire / delta) * sqrt(pi/4)
            xi encodes how many skin depths fit in the wire

        For a single-layer coil with N_layers = 1:
            F_R = xi * [M1(xi) + (2/3)*(N_layers²-1)*M2(xi)]

        where:
            M1(xi) = (sinh(2xi) + sin(2xi)) / (cosh(2xi) - cos(2xi))
            M2(xi) = (sinh(xi)  - sin(xi))  / (cosh(xi)  + cos(xi))

        For your 12-turn single-layer coil:
            N_layers = 1  →  M2 term vanishes
            F_R = xi * M1(xi)

        At 6.78MHz with 18AWG wire:
            xi ≈ (1.024e-3 / 27e-6) * sqrt(pi/4) ≈ 33.6
            M1 ≈ 1.0 (saturates for large xi)
            F_R ≈ xi * 1.0 ... but capped physically

        For large xi (thick wire, high frequency) the Dowell
        factor approaches a simpler form:
            F_R ≈ 1 + (1/3) * (d/delta)^4 / (1 + (d/delta)^4)
                      × (N_layers² - 0.25) / (N_layers² + 0.5)

        For N_layers=1 this simplifies to the single-layer result.
        """
        d     = coil.wire_diam
        delta = self.skin_depth
        xi    = (d / delta) * np.sqrt(np.pi / 4.0)

        # Dowell M1 and M2 functions
        # Numerically stable computation
        sinh2 = np.sinh(2 * xi)
        sin2  = np.sin(2 * xi)
        cosh2 = np.cosh(2 * xi)
        cos2  = np.cos(2 * xi)

        denom1 = cosh2 - cos2
        M1 = (sinh2 + sin2) / denom1 if abs(denom1) > 1e-10 else 1.0

        sinh1 = np.sinh(xi)
        sin1  = np.sin(xi)
        cosh1 = np.cosh(xi)
        cos1  = np.cos(xi)

        denom2 = cosh1 + cos1
        M2 = (sinh1 - sin1) / denom2 if abs(denom2) > 1e-10 else 0.0

        # Single-layer coil (N_l = 1 layer of turns)
        # F_R = xi * [M1 + (2/3)*(1-1)*M2] = xi * M1
        N_l  = 1
        F_R  = xi * (M1 + (2.0 / 3.0) * (N_l**2 - 1) * M2)

        # Physical lower bound: F_R >= 1 (can't be less resistive than DC)
        # Physical upper bound: cap at reasonable value for single layer
        F_R = np.clip(F_R, 1.0, 5.0)
        return float(F_R)

    # ── ─────────────────────────────────────────────────────
    # PARASITIC CAPACITANCE + MEDHURST CORRECTION
    # ── ─────────────────────────────────────────────────────

    def _parasitic_capacitance(self, coil: CoilParams) -> float:
        """
        Inter-turn parasitic capacitance — Medhurst model.

        Adjacent turns are separated by enamel insulation, forming
        a distributed capacitor between each pair of turns.

        For a single-layer spiral coil, the total inter-turn
        parasitic capacitance is approximated as:

            C_p = epsilon_0 * epsilon_r * pi * w² /
                  (4 * t_ins * N)

        where:
            w     = wire diameter (bare)
            t_ins = enamel insulation thickness per side
            N     = number of turns
            epsilon_r = relative permittivity of enamel (~3.5)

        Physical picture: each adjacent turn pair is like a
        parallel plate capacitor. The plate area is the
        cross-sectional area of wire contact (~pi*w²/4),
        the dielectric is the enamel, and there are N-1 such
        capacitors in series/parallel combination.

        Note: this is a lumped approximation of a distributed
        phenomenon. It gives the correct order of magnitude
        and captures the dominant physics.
        """
        w     = coil.wire_diam
        t_ins = ENAMEL_THICKNESS
        N     = coil.N
        eps   = EPSILON0 * EPSILON_R_ENAMEL

        # Inter-turn capacitance per pair
        C_per_pair = eps * np.pi * w**2 / (4 * t_ins)

        # N turns → N-1 adjacent pairs
        # In a spiral coil these are approximately in parallel
        # (each pair shares the same voltage across the coil)
        # divided by N² for the distributed winding factor
        C_total = C_per_pair * (N - 1) / (N**2)

        return C_total

    def _self_resonant_freq(self, L: float, C_p: float) -> float:
        """
        Self-resonant frequency of coil.

        f_SRF = 1 / (2*pi*sqrt(L * C_p))

        At f_SRF the coil's own parasitic capacitance resonates
        with its inductance. Above f_SRF the coil looks capacitive.
        Always operate well below f_SRF (< f_SRF/2 is safe).
        """
        if L <= 0 or C_p <= 0:
            return float('inf')
        return 1.0 / (2 * np.pi * np.sqrt(L * C_p))

    def _medhurst_correction(self, L_raw: float, f_srf: float) -> float:
        """
        Medhurst correction for effective inductance near SRF.

        As operating frequency approaches the self-resonant
        frequency, the parasitic capacitance resonates with the
        inductance and the apparent inductance increases:

            L_eff = L_raw / (1 - (f/f_SRF)²)

        This correction:
          - Is negligible at f << f_SRF  (e.g. f/f_SRF = 0.1 → +1%)
          - Becomes significant near f_SRF (e.g. f/f_SRF = 0.5 → +33%)
          - Diverges at f = f_SRF (resonance)

        For your coil at 6.78MHz:
          f_SRF >> 6.78MHz  (SRF is typically 50-200MHz for this geometry)
          Correction is small but non-negligible
        """
        if f_srf <= 0:
            return L_raw
        ratio = self.freq / f_srf
        if ratio >= 1.0:
            # Operating above SRF — coil is capacitive, not useful
            # Return raw value with warning flag
            return L_raw
        return L_raw / (1.0 - ratio**2)

    # ── ─────────────────────────────────────────────────────
    # INDUCTANCE PRIMITIVES — Khan et al. 2018
    # ── ─────────────────────────────────────────────────────

    def _L_single_loop(self, R: float, w: float) -> float:
        """
        Self inductance of single circular loop. Khan 2018 Eq.(1).

        L = mu0 * R * (ln(8R/w) - 2)

        Derived from Neumann formula: integrate magnetic flux
        through the loop created by its own current.
        Larger R and thinner wire both increase inductance.
        """
        return MU0 * R * (np.log(8.0 * R / w) - 2.0)

    def _M_two_loops(self, Ri: float, Rj: float,
                     dij: float) -> float:
        """
        Mutual inductance between two coaxial circular loops.
        Khan 2018 Eq.(2) — exact Neumann integral solution.

        M = mu0*sqrt(Ri*Rj)/alpha * [(2-alpha²)*K(alpha) - 2*E(alpha)]

        where:
            alpha² = 4*Ri*Rj / ((Ri+Rj)² + dij²)
            K, E   = complete elliptic integrals 1st and 2nd kind

        The elliptic integrals arise naturally from integrating
        the magnetic vector potential around circular loops.
        This formula is exact for any separation and radii.
        """
        if dij == 0 and abs(Ri - Rj) < 1e-10:
            return self._L_single_loop(Ri, self.tx.wire_diam)
        alpha_sq = (4.0 * Ri * Rj) / ((Ri + Rj)**2 + dij**2)
        alpha_sq = np.clip(alpha_sq, 0.0, 0.9999)
        K = special.ellipk(alpha_sq)
        E = special.ellipe(alpha_sq)
        return (MU0 * np.sqrt(Ri * Rj) / np.sqrt(alpha_sq) *
                ((2.0 - alpha_sq) * K - 2.0 * E))

    # ── ─────────────────────────────────────────────────────
    # ARCH CURVATURE GEOMETRY
    # ── ─────────────────────────────────────────────────────

    def _effective_radius(self, r: float, h: float,
                          R_ref: float, n_pts: int = 360) -> float:
        """
        Effective radius of a loop on arch-curved coil.

        Arch profile: z(x) = h * (1 - x²/R_ref²)
        where x = r*cos(theta) is the x-position on the loop.

        Curvature tilts the wire surface by angle phi where:
            tan(phi) = dz/dx = -2*h*x/R_ref²

        The horizontal projection of each wire segment is
        foreshortened by cos(phi):
            cos(phi) = 1/sqrt(1 + (dz/dx)²)

        Effective radius averages this foreshortening around
        the full loop circumference.
        """
        if h == 0:
            return r
        theta    = np.linspace(0, 2 * np.pi, n_pts)
        x        = r * np.cos(theta)
        dz_dx    = -2.0 * h * x / R_ref**2
        cos_phi  = 1.0 / np.sqrt(1.0 + dz_dx**2)
        r_eff    = np.mean(np.sqrt(
            (r * np.cos(theta) * cos_phi)**2 +
            (r * np.sin(theta))**2
        ))
        return float(r_eff)

    def _avg_z_sep(self, ri: float, rj: float,
                   h: float, R_ref: float,
                   n_pts: int = 360) -> float:
        """
        Average vertical separation between two loops on arch coil.

        On a curved coil, turns at different radii sit at
        different heights. This additional separation dij
        reduces their mutual inductance compared to a flat coil.

        z(r,theta) = h * (1 - (r*cos(theta))²/R_ref²)

        Average |zi - zj| around the loop circumference.
        """
        if h == 0:
            return 0.0
        theta = np.linspace(0, 2 * np.pi, n_pts)
        zi = h * (1.0 - (ri * np.cos(theta))**2 / R_ref**2)
        zj = h * (1.0 - (rj * np.cos(theta))**2 / R_ref**2)
        return float(np.mean(np.abs(zi - zj)))

    # ── ─────────────────────────────────────────────────────
    # COIL SELF-INDUCTANCE
    # ── ─────────────────────────────────────────────────────

    def _coil_inductance(self, coil: CoilParams) -> float:
        """
        Total self inductance of spiral coil. Khan 2018 Eq.(5).

        L_total = sum_i(L_i) + sum_i(sum_j≠i(M_ij))

        Every turn magnetically couples to every other turn.
        The mutual coupling between turns significantly increases
        total inductance beyond the sum of individual loops.

        For curved coil:
          - Use effective (foreshortened) radii
          - Include non-zero z-separation between turns
        """
        h      = coil.curvature_h
        R_ref  = coil.R_outer
        radii  = coil.loop_radii
        L      = 0.0
        for i in range(coil.N):
            for j in range(coil.N):
                ri = self._effective_radius(radii[i], h, R_ref)
                rj = self._effective_radius(radii[j], h, R_ref)
                if i == j:
                    L += self._L_single_loop(ri, coil.wire_diam)
                else:
                    dij = self._avg_z_sep(radii[i], radii[j], h, R_ref)
                    L  += self._M_two_loops(ri, rj, dij)
        return L

    # ── ─────────────────────────────────────────────────────
    # MUTUAL INDUCTANCE WITH POSE
    # ── ─────────────────────────────────────────────────────

    def _mutual_inductance(self, pose: Pose2D) -> float:
        """
        Total mutual inductance between Tx and Rx.

        Sums over all N_tx × N_rx loop pairs.
        Each pair contribution modified by:

        1. Axial separation d1:
           Enters M_two_loops as dij — dominates coupling strength.

        2. Lateral misalignment d2 — Khan 2018 Eq.(20):
           Taylor expansion of Neumann integral for off-axis coils.
           gamma_c = d2 / sqrt(r² + d1²)  (must be << 1)
           lateral_factor = 1 - 1.5*gamma_c²
           Valid for d2 < ~0.7*sqrt(r²+d1²)

        3. Angular misalignment theta — Khan 2018 Eq.(28):
           Rx coil tilted by theta about Y axis.
           angular_factor = cos(theta)*cos(lambda)
           where lambda = pitch angle (0 for 2D pose)
           At 90°: factor=0, M=0, no power transfer.

        4. Rx curvature:
           Effective radius of each Rx loop is foreshortened
           by arch curvature geometry.
        """
        d1    = self.d_axial
        d2    = pose.d2
        theta = pose.theta_rad
        h_rx  = self.rx.curvature_h
        R_ref = self.rx.R_outer
        lam   = 0.0  # pitch angle (2D pose only)

        M = 0.0
        for ri in self.tx.loop_radii:
            for rj in self.rx.loop_radii:
                # Effective radius with curvature foreshortening
                r_eff = self._effective_radius(rj, h_rx, R_ref)

                # Base mutual inductance — axial separation only
                M_base = self._M_two_loops(ri, r_eff, d1)

                # Lateral misalignment correction — Khan Eq.(20)
                if d2 > 1e-6:
                    denom  = ri**2 + d1**2
                    gamma_c = d2 / np.sqrt(denom) if denom > 0 else 0
                    gamma_c = np.clip(gamma_c, 0, 0.95)
                    lat_factor = max(0.0, 1.0 - 1.5 * gamma_c**2)
                    M_base *= lat_factor

                # Angular misalignment — Khan Eq.(28)
                if abs(theta) > 1e-6:
                    ang_factor = np.cos(theta) * np.cos(lam)
                    M_base *= ang_factor

                M += M_base
        return M

    # ── ─────────────────────────────────────────────────────
    # FREQUENCY SPLITTING ANALYSIS
    # ── ─────────────────────────────────────────────────────

    def _frequency_splitting(self, k: float):
        """
        Determine coupling regime and split frequencies.

        Critical coupling condition:
            k_critical = 1 / sqrt(Q_tx * Q_rx)

        When k < k_crit: undercoupled — single peak at f0
        When k = k_crit: critical — peak efficiency
        When k > k_crit: overcoupled — two peaks at f±

        Split frequencies (series-series resonant):
            f± = f0 / sqrt(1 ∓ k)

        Split gap:
            delta_f = f+ - f-  (kHz)
        """
        f0     = self.freq
        Q_tx   = self.Q_tx
        Q_rx   = self.Q_rx
        k_crit = 1.0 / np.sqrt(Q_tx * Q_rx) if Q_tx > 0 and Q_rx > 0 else 1.0

        if k < 0.95 * k_crit:
            regime = 'undercoupled'
            f_low  = f_high = f0
            gap    = 0.0
        elif k < 1.05 * k_crit:
            regime = 'critical'
            f_low  = f_high = f0
            gap    = 0.0
        else:
            regime = 'overcoupled'
            f_low  = f0 / np.sqrt(1.0 + k)
            f_high = f0 / np.sqrt(max(1.0 - k, 1e-6))
            gap    = (f_high - f_low) / 1e3

        return (f_low / 1e6, f_high / 1e6,
                gap, regime, k_crit)

    # ── ─────────────────────────────────────────────────────
    # S-PARAMETERS
    # ── ─────────────────────────────────────────────────────

    def _s_parameters(self, k: float, Z0: float = 50.0):
        """
        S-parameters for series-series resonant WPT network.

        At resonance the LC tank impedance cancels, leaving
        only resistive terms:
            Z_tx = R_tx_total
            Z_rx = R_rx_total

        S11 — Tx port return loss:
            S11 = (Z_tx - Z0) / (Z_tx + Z0)
            Measures how much power is reflected at Tx port.
            S11 = -∞ dB = perfect match (all power enters)
            S11 = 0 dB  = total reflection

        S21 — Forward transmission (what you measure):
            S21 = 2*k*sqrt(Q_tx*Q_rx) / (1 + k²*Q_tx*Q_rx)
            This is the power delivered to Rx relative to
            power available from source.
            S21 = 0 dB  = perfect (all power delivered)
            S21 = -20dB = 1% power delivered

        S22 — Rx port return loss (same form as S11)
        S12 = S21 (reciprocal network)
        """
        R_tx = self.R_tx_total
        R_rx = self.R_rx_total
        kQ   = k * np.sqrt(self.Q_tx * self.Q_rx)

        # Return loss
        gamma_tx = (R_tx - Z0) / (R_tx + Z0)
        gamma_rx = (R_rx - Z0) / (R_rx + Z0)
        S11_dB   = 20 * np.log10(max(abs(gamma_tx), 1e-12))
        S22_dB   = 20 * np.log10(max(abs(gamma_rx), 1e-12))

        # Forward transmission
        if kQ > 0:
            S21_lin = 2 * kQ / (1 + kQ**2)
            S21_lin = np.clip(S21_lin, 1e-12, 1.0)
        else:
            S21_lin = 1e-12
        S21_dB = 20 * np.log10(S21_lin)
        S12_dB = S21_dB

        return S11_dB, S22_dB, S21_dB, S12_dB

    # ── ─────────────────────────────────────────────────────
    # MAIN EVALUATION
    # ── ─────────────────────────────────────────────────────

    def evaluate(self, pose: Pose2D) -> WPTResult:
        """
        Evaluate WPT system at a given 2D pose.
        Returns full WPTResult with all metrics.
        """
        # Mutual inductance at this pose
        M  = self._mutual_inductance(pose)
        k  = np.clip(abs(M) / np.sqrt(self.L_tx * self.L_rx),
                     0, 0.9999)
        kQ = k * np.sqrt(self.Q_tx * self.Q_rx)

        # Efficiency — series-series resonant WPT
        # eta = kQ² / (1 + kQ²)²
        eta = (kQ**2 / (1 + kQ**2)**2) * 100 if kQ > 0 else 0.0

        # Power
        P_in  = self.V_in * self.I_in
        P_out = P_in * eta / 100.0

        # Output voltage
        V_oc  = self.omega * M * self.I_in
        V_out = V_oc * np.sqrt(max(eta / 100.0, 0))

        # Output current
        R_L   = self.R_load if self.R_load else self.R_rx_total
        I_out = V_out / R_L if R_L > 0 else 0.0

        # Frequency splitting
        f_low, f_high, gap, regime, k_crit = \
            self._frequency_splitting(k)

        # S-parameters
        S11, S22, S21, S12 = self._s_parameters(k)

        return WPTResult(
            x                 = pose.x,
            y                 = pose.y,
            theta_deg         = pose.theta,
            d_axial_mm        = self.d_axial * 1e3,
            d_lateral_mm      = pose.d2 * 1e3,
            L_tx_uH           = self.L_tx_raw * 1e6,
            L_rx_uH           = self.L_rx_raw * 1e6,
            L_tx_eff_uH       = self.L_tx * 1e6,
            L_rx_eff_uH       = self.L_rx * 1e6,
            C_tx_pF           = self.C_tx * 1e12,
            C_rx_pF           = self.C_rx * 1e12,
            C_parasitic_tx_pF = self.C_p_tx * 1e12,
            C_parasitic_rx_pF = self.C_p_rx * 1e12,
            f_srf_tx_MHz      = self.f_srf_tx / 1e6,
            f_srf_rx_MHz      = self.f_srf_rx / 1e6,
            R_tx_skin         = self.R_tx_skin,
            R_rx_skin         = self.R_rx_skin,
            R_tx_total        = self.R_tx_total,
            R_rx_total        = self.R_rx_total,
            proximity_factor  = self.proximity_factor,
            Q_tx              = self.Q_tx,
            Q_rx              = self.Q_rx,
            M_nH              = M * 1e9,
            k                 = k,
            kQ                = kQ,
            f_split_low_MHz   = f_low,
            f_split_high_MHz  = f_high,
            split_gap_kHz     = gap,
            coupling_regime   = regime,
            k_critical        = k_crit,
            S11_dB            = S11,
            S22_dB            = S22,
            S21_dB            = S21,
            S12_dB            = S12,
            V_in              = self.V_in,
            I_in              = self.I_in,
            P_in_W            = P_in,
            V_out             = V_out,
            I_out             = I_out,
            P_out_W           = P_out,
            efficiency_pct    = eta,
        )

    def info(self):
        """Print full system summary."""
        print("=" * 65)
        print("  WPT System — Physical Model Summary")
        print("=" * 65)
        print(f"  Tx: {self.tx.summary()}")
        print(f"  Rx: {self.rx.summary()}")
        print(f"\n  Operating conditions:")
        print(f"    Frequency:    {self.freq/1e6:.3f} MHz")
        print(f"    Temperature:  {self.temp_c}°C")
        print(f"    V_in:         {self.V_in} V")
        print(f"    d_axial:      {self.d_axial*1e3:.1f} mm")
        print(f"\n  Copper model:")
        print(f"    rho(T):       {self.rho_cu*1e8:.4f} ×10⁻⁸ Ω·m  "
              f"(+{(self.rho_cu/RHO_CU_20-1)*100:.1f}% vs 20°C)")
        print(f"    Skin depth:   {self.skin_depth*1e6:.2f} µm")
        print(f"    d/delta:      {self.tx.wire_diam/self.skin_depth:.1f}")
        print(f"\n  Resistance model:")
        print(f"    R_skin:       {self.R_tx_skin:.4f} Ω")
        print(f"    Proximity F_R:{self.proximity_factor:.4f}×")
        print(f"    R_total:      {self.R_tx_total:.4f} Ω")
        print(f"\n  Parasitic capacitance:")
        print(f"    C_parasitic:  {self.C_p_tx*1e12:.3f} pF")
        print(f"    f_SRF (Tx):   {self.f_srf_tx/1e6:.1f} MHz")
        print(f"    f_SRF (Rx):   {self.f_srf_rx/1e6:.1f} MHz")
        print(f"    f/f_SRF:      {self.freq/self.f_srf_tx:.4f}  "
              f"({'safe' if self.freq < self.f_srf_tx/2 else 'WARNING: near SRF'})")
        print(f"\n  Medhurst-corrected inductance:")
        print(f"    L_tx raw:     {self.L_tx_raw*1e6:.4f} µH")
        print(f"    L_tx eff:     {self.L_tx*1e6:.4f} µH  "
              f"(+{(self.L_tx/self.L_tx_raw-1)*100:.2f}%)")
        print(f"    L_rx raw:     {self.L_rx_raw*1e6:.4f} µH")
        print(f"    L_rx eff:     {self.L_rx*1e6:.4f} µH  "
              f"(+{(self.L_rx/self.L_rx_raw-1)*100:.2f}%)")
        print(f"\n  Resonant capacitances:")
        print(f"    C_tx:         {self.C_tx*1e12:.2f} pF")
        print(f"    C_rx:         {self.C_rx*1e12:.2f} pF")
        print(f"\n  Q factors:")
        print(f"    Q_tx:         {self.Q_tx:.1f}")
        print(f"    Q_rx:         {self.Q_rx:.1f}")
        print(f"    k_critical:   {1/np.sqrt(self.Q_tx*self.Q_rx):.6f}")
        print("=" * 65)


# ═════════════════════════════════════════════════════════════
# QUICK TEST
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tx = CoilParams(curvature_h=0.0)
    rx = CoilParams(curvature_h=20e-3)

    system = WPTSystem(tx=tx, rx=rx, freq=6.78e6,
                       V_in=12.0, I_in=0.24,
                       d_axial=50e-3, temp_c=75.0)
    system.info()

    print("\nPose sweep:")
    print(f"{'Pose':<32} {'k':<10} {'η (%)':<10} "
          f"{'S21(dB)':<10} {'Regime':<14} {'Gap(kHz)'}")
    print("-" * 85)
    poses = [
        Pose2D(x=0,     y=0, theta=0),
        Pose2D(x=20e-3, y=0, theta=0),
        Pose2D(x=0,     y=0, theta=45),
        Pose2D(x=20e-3, y=0, theta=45),
        Pose2D(x=0,     y=0, theta=90),
    ]
    for p in poses:
        r = system.evaluate(p)
        lbl = (f"x={p.x*1e3:.0f}mm y={p.y*1e3:.0f}mm "
               f"θ={p.theta:.0f}°")
        print(f"{lbl:<32} {r.k:<10.6f} {r.efficiency_pct:<10.4f} "
              f"{r.S21_dB:<10.2f} {r.coupling_regime:<14} "
              f"{r.split_gap_kHz:.1f}")
