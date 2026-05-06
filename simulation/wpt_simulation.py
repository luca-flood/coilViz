"""
wpt_simulation.py  v3.2
────────────────────────────────────────────────────────────────
Modular WPT Simulation — Khan et al. 2018

9 ablation configurations:
  1. full_model                  all effects enabled, perfect resonance
  2. no_skin_effect              DC resistance used instead of AC skin
  3. no_proximity_effect         F_R = 1 (no inter-turn distortion)
  4. no_parasitic_cap            C_p = 0, Medhurst disabled
  5. no_medhurst_correction      L_eff = L_raw
  6. no_curvature_geometry       curved coil treated as flat
  7. no_temp_correction          20C resistivity used
  8. fixed_6p78MHz_no_retune     Rx cap tuned for FLAT coil once,
                                 never retuned as curvature changes
  9. fixed_120pF_rx              Rx cap = 120pF (your real hardware:
                                 4x30pF in parallel), never retuned
"""

import numpy as np
from scipy import special
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ════════════════════════════════════════════════════════════
MU0              = 4 * np.pi * 1e-7
EPSILON0         = 8.854e-12
RHO_CU_20        = 1.68e-8
ALPHA_CU         = 0.00393
EPSILON_R_ENAMEL = 3.5
ENAMEL_THICKNESS = 0.04e-3


# ════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════

@dataclass
class CoilParams:
    N           : int   = 12
    R_outer     : float = 43e-3
    R_inner     : float = 10e-3
    pitch       : float = 1.2e-3
    wire_diam   : float = 1.024e-3
    curvature_h : float = 0.0

    def __post_init__(self):
        self.loop_radii = np.linspace(self.R_inner, self.R_outer, self.N)

    def summary(self):
        return (f"CoilParams(N={self.N}, OD={self.R_outer*2e3:.1f}mm, "
                f"ID={self.R_inner*2e3:.1f}mm, h={self.curvature_h*1e3:.1f}mm)")


@dataclass
class Pose2D:
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
class AblationConfig:
    """
    Controls which physical effects are active.

    rx_cap_mode:
      'perfect'  — Rx cap always tuned to match L_rx exactly (default)
      'fixed_flat' — Rx cap tuned for FLAT coil once, held fixed
      'fixed_120pF' — Rx cap = 120pF always (real hardware)
    """
    name                   : str   = 'full_model'
    use_skin_effect        : bool  = True
    use_proximity_effect   : bool  = True
    use_parasitic_cap      : bool  = True
    use_medhurst           : bool  = True
    use_curvature_geometry : bool  = True
    use_temp_correction    : bool  = True


    def description(self):
        removed = []
        if not self.use_skin_effect:        removed.append('skin')
        if not self.use_proximity_effect:   removed.append('proximity')
        if not self.use_parasitic_cap:      removed.append('parasitic_cap')
        if not self.use_medhurst:           removed.append('medhurst')
        if not self.use_curvature_geometry: removed.append('curvature_geom')
        if not self.use_temp_correction:    removed.append('temp_correction')

        tag = f" [removed: {', '.join(removed)}]" if removed else " [all effects]"
        return self.name + tag


# ── Predefined ablation set ───────────────────────────────────
ABLATIONS = [
    # All physical effects active — baseline for comparison
    AblationConfig(name='full_model'),

    # Remove skin effect — uses DC resistance instead
    # Shows: does skin resistance change how curvature affects η?
    AblationConfig(name='no_skin_effect',
                   use_skin_effect=False),

    # Remove proximity effect — F_R = 1.0
    # Shows: does inter-turn current distortion affect curvature impact?
    AblationConfig(name='no_proximity_effect',
                   use_proximity_effect=False),

    # Remove parasitic capacitance — C_p = 0, Medhurst disabled
    # Shows: do inter-turn capacitances affect curvature impact?
    AblationConfig(name='no_parasitic_cap',
                   use_parasitic_cap=False,
                   use_medhurst=False),

    # Remove Medhurst correction — L_eff = L_raw
    # Shows: does SRF-induced inductance change affect curvature?
    AblationConfig(name='no_medhurst_correction',
                   use_medhurst=False),

    # Remove curvature geometry — curved coil treated as flat
    # KEY ABLATION: isolates the pure geometric effect of arch curvature.
    # Difference between this and full_model = curvature geometry effect.
    AblationConfig(name='no_curvature_geometry',
                   use_curvature_geometry=False),

    # Remove temperature correction — use 20C resistivity
    # Shows: does thermal resistance change affect curvature impact?
    AblationConfig(name='no_temp_correction',
                   use_temp_correction=False),
]


@dataclass
class WPTResult:
    ablation_name          : str   = ''
    ablation_description   : str   = ''
    conic_curve_mm         : float = 0.0
    x                      : float = 0.0
    y                      : float = 0.0
    theta_deg              : float = 0.0
    d_axial_mm             : float = 0.0
    d_lateral_mm           : float = 0.0
    alignment              : str   = ''
    group                  : str   = ''
    L_tx_raw_uH            : float = 0.0
    L_rx_raw_uH            : float = 0.0
    L_tx_eff_uH            : float = 0.0
    L_rx_eff_uH            : float = 0.0
    C_tx_pF                : float = 0.0
    C_rx_pF                : float = 0.0
    f_resonant_tx_MHz      : float = 0.0
    f_resonant_rx_MHz      : float = 0.0
    rx_detuning_kHz        : float = 0.0
    C_parasitic_tx_pF      : float = 0.0
    C_parasitic_rx_pF      : float = 0.0
    f_srf_tx_MHz           : float = 0.0
    f_srf_rx_MHz           : float = 0.0
    R_tx_dc                : float = 0.0
    R_tx_skin              : float = 0.0
    R_tx_total             : float = 0.0
    R_rx_dc                : float = 0.0
    R_rx_skin              : float = 0.0
    R_rx_total             : float = 0.0
    proximity_factor       : float = 0.0
    temp_factor            : float = 0.0
    Q_tx                   : float = 0.0
    Q_rx                   : float = 0.0
    M_nH                   : float = 0.0
    k                      : float = 0.0
    kQ                     : float = 0.0
    k_critical             : float = 0.0
    f_split_low_MHz        : float = 0.0
    f_split_high_MHz       : float = 0.0
    split_gap_kHz          : float = 0.0
    coupling_regime        : str   = ''
    S11_dB                 : float = 0.0
    S22_dB                 : float = 0.0
    S21_dB                 : float = 0.0
    S12_dB                 : float = 0.0
    V_in                   : float = 0.0
    I_in                   : float = 0.0
    P_in_W                 : float = 0.0
    V_out                  : float = 0.0
    I_out                  : float = 0.0
    P_out_W                : float = 0.0
    efficiency_pct         : float = 0.0

    def to_dict(self):
        return {k: (round(v, 8) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


# ════════════════════════════════════════════════════════════
# PHYSICS ENGINE
# ════════════════════════════════════════════════════════════

class WPTSystem:
    """
    Full WPT system with ablation support.

    rx_cap_mode logic:
      'perfect'    — C_rx = 1/(w²*L_rx) always, zero detuning
      'fixed_flat' — C_rx computed for flat coil once, held fixed
                     as curvature increases L_rx changes but C_rx stays
      'fixed_120pF'— C_rx = 120pF always (your 4x30pF hardware)
    """

    # Class-level cache for flat coil capacitance
    # Computed once and shared across all instances with fixed_flat mode
    _flat_C_rx_cache = {}

    def __init__(self,
                 tx         : CoilParams,
                 rx         : CoilParams,
                 freq       : float           = 6.78e6,
                 V_in       : float           = 12.0,
                 I_in       : float           = 0.24,
                 d_axial    : float           = 50e-3,
                 temp_c     : float           = 75.0,
                 R_load     : Optional[float] = None,
                 ablation   : AblationConfig  = None,
):
        self.tx      = tx
        self.rx      = rx
        self.freq    = freq
        self.omega   = 2 * np.pi * freq
        self.V_in    = V_in
        self.I_in    = I_in
        self.d_axial = d_axial
        self.temp_c  = temp_c
        self.R_load  = R_load
        self.abl     = ablation or AblationConfig()

        # Temperature
        self.temp_factor = (1 + ALPHA_CU * (temp_c - 20)
                            if self.abl.use_temp_correction else 1.0)
        self.rho_cu = RHO_CU_20 * self.temp_factor

        # Skin depth
        self.skin_depth = np.sqrt(self.rho_cu / (np.pi * freq * MU0))

        # Proximity factor
        self.proximity_factor = (self._dowell(tx)
                                 if self.abl.use_proximity_effect else 1.0)

        # Resistances
        self.R_tx_dc    = self._dc_r(tx)
        self.R_rx_dc    = self._dc_r(rx)
        self.R_tx_skin  = self._skin_r(tx) if self.abl.use_skin_effect else self.R_tx_dc
        self.R_rx_skin  = self._skin_r(rx) if self.abl.use_skin_effect else self.R_rx_dc
        self.R_tx_total = self.R_tx_skin * self.proximity_factor
        self.R_rx_total = self.R_rx_skin * self.proximity_factor

        # Inductances
        self.L_tx_raw = self._L_coil(tx)
        self.L_rx_raw = self._L_coil(rx)

        # Parasitic caps
        self.C_p_tx = self._C_parasitic(tx) if self.abl.use_parasitic_cap else 0.0
        self.C_p_rx = self._C_parasitic(rx) if self.abl.use_parasitic_cap else 0.0

        # SRF
        self.f_srf_tx = self._srf(self.L_tx_raw, self.C_p_tx)
        self.f_srf_rx = self._srf(self.L_rx_raw, self.C_p_rx)

        # Medhurst correction
        use_med  = self.abl.use_medhurst and self.abl.use_parasitic_cap
        self.L_tx = self._medhurst(self.L_tx_raw, self.f_srf_tx) if use_med else self.L_tx_raw
        self.L_rx = self._medhurst(self.L_rx_raw, self.f_srf_rx) if use_med else self.L_rx_raw

        # ── Perfect resonance — always assumed ───────────────
        # Both coils tuned to their optimal resonant frequency.
        # C = 1/(omega^2 * L_eff). At resonance LC impedance = 0.
        # This isolates curvature geometry from hardware constraints.
        self.C_tx          = 1.0 / (self.omega**2 * self.L_tx)
        self.C_rx          = 1.0 / (self.omega**2 * self.L_rx)
        self.f_resonant_tx = freq
        self.f_resonant_rx = freq
        self.rx_detuning   = 0.0

        # Q factors — perfect resonance, no detuning penalty
        self.Q_tx     = self.omega * self.L_tx / self.R_tx_total
        self.Q_rx     = self.omega * self.L_rx / self.R_rx_total
        self.Q_rx_eff = self.Q_rx

    # ── Resistance ──────────────────────────────────────────

    def _dc_r(self, coil):
        wl = sum(2*np.pi*r for r in coil.loop_radii)
        return self.rho_cu * wl / (np.pi * (coil.wire_diam/2)**2)

    def _skin_r(self, coil):
        wl = sum(2*np.pi*r for r in coil.loop_radii)
        return self.rho_cu * wl / (np.pi * self.skin_depth * coil.wire_diam)

    def _dowell(self, coil):
        xi    = (coil.wire_diam / self.skin_depth) * np.sqrt(np.pi/4)
        sinh2 = np.sinh(2*xi); sin2 = np.sin(2*xi)
        cosh2 = np.cosh(2*xi); cos2 = np.cos(2*xi)
        d1 = cosh2 - cos2
        M1 = (sinh2+sin2)/d1 if abs(d1)>1e-10 else 1.0
        sinh1=np.sinh(xi); sin1=np.sin(xi)
        cosh1=np.cosh(xi); cos1=np.cos(xi)
        d2 = cosh1 + cos1
        M2 = (sinh1-sin1)/d2 if abs(d2)>1e-10 else 0.0
        return float(np.clip(xi*(M1+(2/3)*(1-1)*M2), 1.0, 5.0))

    # ── Parasitic cap + Medhurst ─────────────────────────────

    def _C_parasitic(self, coil):
        eps = EPSILON0 * EPSILON_R_ENAMEL
        C   = eps * np.pi * coil.wire_diam**2 / (4 * ENAMEL_THICKNESS)
        return C * (coil.N-1) / coil.N**2

    def _srf(self, L, Cp):
        if L<=0 or Cp<=0: return float('inf')
        return 1.0 / (2*np.pi*np.sqrt(L*Cp))

    def _medhurst(self, L_raw, f_srf):
        if np.isinf(f_srf) or f_srf<=0: return L_raw
        r = self.freq / f_srf
        return L_raw if r>=1.0 else L_raw/(1.0-r**2)

    # ── Inductance ───────────────────────────────────────────

    def _L_loop(self, R, w):
        return MU0 * R * (np.log(8*R/w) - 2.0)

    def _M_loops(self, Ri, Rj, dij):
        if dij==0 and abs(Ri-Rj)<1e-10:
            return self._L_loop(Ri, self.tx.wire_diam)
        a2 = np.clip(4*Ri*Rj/((Ri+Rj)**2+dij**2), 0, 0.9999)
        K  = special.ellipk(a2)
        E  = special.ellipe(a2)
        return MU0*np.sqrt(Ri*Rj)/np.sqrt(a2)*((2-a2)*K-2*E)

    def _eff_r(self, r, h, R_ref, n=360):
        if h==0 or not self.abl.use_curvature_geometry: return r
        t  = np.linspace(0, 2*np.pi, n)
        x  = r*np.cos(t)
        dz = -2*h*x/R_ref**2
        cp = 1.0/np.sqrt(1+dz**2)
        return float(np.mean(np.sqrt((r*np.cos(t)*cp)**2+(r*np.sin(t))**2)))

    def _avg_z(self, ri, rj, h, R_ref, n=360):
        if h==0 or not self.abl.use_curvature_geometry: return 0.0
        t  = np.linspace(0, 2*np.pi, n)
        zi = h*(1-(ri*np.cos(t))**2/R_ref**2)
        zj = h*(1-(rj*np.cos(t))**2/R_ref**2)
        return float(np.mean(np.abs(zi-zj)))

    def _L_coil(self, coil):
        h=coil.curvature_h; R=coil.R_outer; rad=coil.loop_radii; L=0.0
        for i in range(coil.N):
            for j in range(coil.N):
                ri=self._eff_r(rad[i],h,R); rj=self._eff_r(rad[j],h,R)
                L += (self._L_loop(ri,coil.wire_diam) if i==j
                      else self._M_loops(ri,rj,self._avg_z(rad[i],rad[j],h,R)))
        return L

    # ── Mutual inductance ────────────────────────────────────

    def _mutual(self, pose):
        d1=self.d_axial; d2=pose.d2; theta=pose.theta_rad
        h_rx=self.rx.curvature_h; R_ref=self.rx.R_outer; M=0.0
        for ri in self.tx.loop_radii:
            for rj in self.rx.loop_radii:
                re  = self._eff_r(rj,h_rx,R_ref)
                Mb  = self._M_loops(ri,re,d1)
                if d2>1e-6:
                    gc  = np.clip(d2/np.sqrt(ri**2+d1**2),0,0.95)
                    Mb *= max(0.0,1-1.5*gc**2)
                if abs(theta)>1e-6:
                    Mb *= np.cos(theta)
                M  += Mb
        return M

    # ── Frequency splitting ──────────────────────────────────

    def _split(self, k):
        f0  = self.freq
        k_c = (1/np.sqrt(self.Q_tx*self.Q_rx_eff)
               if self.Q_tx>0 and self.Q_rx_eff>0 else 1.0)
        if   k < 0.95*k_c: return f0/1e6,f0/1e6,0.0,'undercoupled',k_c
        elif k < 1.05*k_c: return f0/1e6,f0/1e6,0.0,'critical',k_c
        else:
            fl=f0/np.sqrt(1+k); fh=f0/np.sqrt(max(1-k,1e-6))
            return fl/1e6,fh/1e6,(fh-fl)/1e3,'overcoupled',k_c

    # ── S-parameters ────────────────────────────────────────

    def _sparams(self, k, Z0=50.0):
        kQ  = k*np.sqrt(self.Q_tx*self.Q_rx_eff)
        gTx = (self.R_tx_total-Z0)/(self.R_tx_total+Z0)
        gRx = (self.R_rx_total-Z0)/(self.R_rx_total+Z0)
        S11 = 20*np.log10(max(abs(gTx),1e-12))
        S22 = 20*np.log10(max(abs(gRx),1e-12))
        s21 = np.clip(2*kQ/(1+kQ**2),1e-12,1.0) if kQ>0 else 1e-12
        S21 = 20*np.log10(s21)
        return S11,S22,S21,S21

    # ── Main evaluate ────────────────────────────────────────

    def evaluate(self, pose: Pose2D,
                 alignment: str = '',
                 group: str = '',
                 conic_curve_mm: float = 0.0) -> WPTResult:
        M   = self._mutual(pose)
        k   = np.clip(abs(M)/np.sqrt(self.L_tx*self.L_rx),0,0.9999)
        kQ  = k*np.sqrt(self.Q_tx*self.Q_rx_eff)
        eta = (kQ**2/(1+kQ**2)**2)*100 if kQ>0 else 0.0
        P_in  = self.V_in*self.I_in
        P_out = P_in*eta/100.0
        V_out = self.omega*M*self.I_in*np.sqrt(max(eta/100,0))
        R_L   = self.R_load or self.R_rx_total
        I_out = V_out/R_L if R_L>0 else 0.0
        fl,fh,gap,regime,k_c = self._split(k)
        S11,S22,S21,S12      = self._sparams(k)
        return WPTResult(
            ablation_name        = self.abl.name,
            ablation_description = self.abl.description(),
            conic_curve_mm       = conic_curve_mm,
            x=pose.x, y=pose.y, theta_deg=pose.theta,
            d_axial_mm           = self.d_axial*1e3,
            d_lateral_mm         = pose.d2*1e3,
            alignment=alignment, group=group,
            L_tx_raw_uH          = self.L_tx_raw*1e6,
            L_rx_raw_uH          = self.L_rx_raw*1e6,
            L_tx_eff_uH          = self.L_tx*1e6,
            L_rx_eff_uH          = self.L_rx*1e6,
            C_tx_pF              = self.C_tx*1e12,
            C_rx_pF              = self.C_rx*1e12,
            f_resonant_tx_MHz    = self.f_resonant_tx/1e6,
            f_resonant_rx_MHz    = self.f_resonant_rx/1e6,
            rx_detuning_kHz      = self.rx_detuning/1e3,
            C_parasitic_tx_pF    = self.C_p_tx*1e12,
            C_parasitic_rx_pF    = self.C_p_rx*1e12,
            f_srf_tx_MHz         = (self.f_srf_tx/1e6 if not np.isinf(self.f_srf_tx) else 9999.0),
            f_srf_rx_MHz         = (self.f_srf_rx/1e6 if not np.isinf(self.f_srf_rx) else 9999.0),
            R_tx_dc=self.R_tx_dc, R_tx_skin=self.R_tx_skin, R_tx_total=self.R_tx_total,
            R_rx_dc=self.R_rx_dc, R_rx_skin=self.R_rx_skin, R_rx_total=self.R_rx_total,
            proximity_factor=self.proximity_factor, temp_factor=self.temp_factor,
            Q_tx=self.Q_tx, Q_rx=self.Q_rx_eff,
            M_nH=M*1e9, k=k, kQ=kQ, k_critical=k_c,
            f_split_low_MHz=fl, f_split_high_MHz=fh,
            split_gap_kHz=gap, coupling_regime=regime,
            S11_dB=S11, S22_dB=S22, S21_dB=S21, S12_dB=S12,
            V_in=self.V_in, I_in=self.I_in, P_in_W=P_in,
            V_out=V_out, I_out=I_out, P_out_W=P_out,
            efficiency_pct=eta,
        )

    def info(self):
        print(f"  [{self.abl.name}]  "
              f"R={self.R_rx_total:.4f}Ω  Q_eff={self.Q_rx_eff:.1f}  "
              f"L={self.L_rx*1e6:.4f}µH  C={self.C_rx*1e12:.2f}pF  "
              f"f_res={self.f_resonant_rx/1e6:.4f}MHz  "
              f"detune={self.rx_detuning/1e3:+.1f}kHz")


# ════════════════════════════════════════════════════════════
# QUICK TEST
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tx = CoilParams(curvature_h=0.0)

    print("Ablation comparison — h=20mm, x=0,y=0,θ=0°,d=50mm\n")
    print(f"{'Ablation':<32} {'R_Ω':>6} {'Q_eff':>7} "
          f"{'f_res_rx':>10} {'detune':>10} {'η%':>8}")
    print("-"*78)

    # Compute flat coil L for fixed_flat mode
    flat_sys = WPTSystem(tx=CoilParams(), rx=CoilParams(),
                         ablation=AblationConfig())
    flat_L_rx = flat_sys.L_rx

    for abl in ABLATIONS:
        rx  = CoilParams(curvature_h=20e-3)
        sys = WPTSystem(tx=tx, rx=rx, ablation=abl,
                        V_in=12.0, I_in=0.24, d_axial=50e-3,
                        flat_L_rx=flat_L_rx)
        r   = sys.evaluate(Pose2D(0,0,0), conic_curve_mm=20)
        print(f"{abl.name:<32} {r.R_rx_total:>6.3f} {r.Q_rx:>7.1f} "
              f"{r.f_resonant_rx_MHz:>10.4f} {r.rx_detuning_kHz:>+10.1f} "
              f"{r.efficiency_pct:>8.4f}")
