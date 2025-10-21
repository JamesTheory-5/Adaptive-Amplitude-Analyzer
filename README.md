# Adaptive-Amplitude-Analyzer

love it. let’s build a **single, stateful adaptive amplitude analyzer** that *changes its own time-constant in real time* based on what the signal is doing. this is the kind of thing you actually slot into synth/FX control paths when you need **fast transients**, **stable sustains**, and **low noise chatter**—without hand-tuning per patch.

below you’ll get (1) the concept, (2) equations, and (3) a clean python reference you can drop into a project.

---

# concept (what it does)

* **goal:** track amplitude with **low latency on rises** and **noise-immune stability on decays/sustains**.
* **how:** compute a base measure (RMS or ABS), then run a **time-varying 1-pole lowpass** where the **effective τ** adapts to:

  1. *slew* (how fast the input is changing)
  2. *SNR* (avoid chasing noise floor)
  3. *crest factor* (peaky vs sustained)
  4. *direction* (attack vs release)

---

# math (how it thinks)

let ( m[n] ) be your *raw measure* (choose ABS or RMS):

* **ABS:** ( m[n]=|x[n]| )
* **RMS (power-root):** ( m[n] = \sqrt{\text{EWMA}_\tau{x^2}} )

adaptive envelope ( e[n] ) is a 1-pole with variable ( \alpha[n] ):

[
e[n] = (1-\alpha[n]),m[n] + \alpha[n],e[n-1],\quad
\alpha[n] = \exp!\left(-\frac{1}{\tau_\text{eff}[n],f_s}\right)
]

effective time constant:

[
\tau_\text{eff}[n] =
\begin{cases}
\tau_\text{att} \cdot g_\text{slew}[n]\cdot g_\text{crest}[n], & m[n] > e[n-1] \
\tau_\text{rel} \cdot g_\text{snr}[n]\cdot g_\text{crest}[n],  & \text{otherwise}
\end{cases}
]

adaptation terms (bounded ([\tau_{\min},\tau_{\max}])):

* **slew gain** (faster when surprise ↑):
  [
  g_\text{slew}[n]=\frac{1}{1+k_\text{d},\frac{|m[n]-m[n-1]|}{\max(m[n],\epsilon)}}
  ]
* **snr gain** (slower when near noise floor ( \sigma )):
  [
  \text{snr}[n]=\frac{m[n]}{\sigma[n]+\epsilon},\quad
  g_\text{snr}[n]=1 + \frac{k_\text{snr}}{1+\text{snr}[n]}
  ]
  where ( \sigma[n] ) is a slow EWMA of (|x|) (noise floor proxy).
* **crest gain** (faster for peaky material, slower for dense):
  [
  \text{crest}[n]=\frac{p[n]}{r[n]+\epsilon},\quad
  g_\text{crest}[n]=\frac{1}{1+k_\text{c},(\text{crest}[n]-C_0)_+}
  ]
  with ( p[n] ) a fast peak follower, ( r[n] ) an RMS EWMA, and (C_0) your “neutral” crest (e.g. 3).

these are simple, monotone, and bounded → robust in practice.

---

# code (drop-in python)

```python
import numpy as np

class AdaptiveAmplitudeAnalyzer:
    """
    Adaptive amplitude envelope for audio DSP control.

    Features
    --------
    - Mode: 'abs' or 'rms' (raw measure before adaptive smoothing)
    - Attack/Release times: baseline τ; auto-adapts around them
    - Adaptive knobs:
        * slew-based acceleration (k_d)
        * SNR-based damping near noise floor (k_snr)
        * crest-based shaping (k_c, C0)
    - Streaming/stateful: safe to call across blocks

    Notes
    -----
    e[n] = (1 - alpha[n]) * m[n] + alpha[n] * e[n-1]
    alpha[n] = exp(-1 / (tau_eff[n] * fs))

    tau_eff is chosen per-sample from attack/release and then
    modulated by slew/SNR/crest terms, and clamped to [tau_min, tau_max].
    """

    def __init__(
        self,
        fs=48000,
        mode="rms",              # 'abs' or 'rms'
        tau_att=0.005,           # baseline attack  (s)
        tau_rel=0.080,           # baseline release (s)
        tau_min=0.002,           # hard clamp (s)
        tau_max=0.500,           # hard clamp (s)
        # RMS internals (if mode='rms'):
        tau_rms=0.010,           # RMS detector EWMA τ (s)
        # Noise floor estimator (slow abs-EWMA):
        tau_noise=0.200,         # (s)
        # Crest trackers:
        tau_peak=0.010,          # fast peak follower τ (s)
        tau_r_crest=0.050,       # RMS for crest denominator (s)
        # Adaptive strengths:
        k_d=4.0,                 # slew accel strength
        k_snr=1.0,               # noise damping strength
        k_c=0.5,                 # crest shaping strength
        C0=3.0,                  # neutral crest factor
        eps=1e-12
    ):
        self.fs = float(fs)
        self.mode = mode
        self.tau_att = tau_att
        self.tau_rel = tau_rel
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_rms = tau_rms
        self.tau_noise = tau_noise
        self.tau_peak = tau_peak
        self.tau_r_crest = tau_r_crest
        self.k_d = k_d
        self.k_snr = k_snr
        self.k_c = k_c
        self.C0 = C0
        self.eps = eps

        # states
        self._e = 0.0          # adaptive envelope
        self._m_prev = 0.0     # previous measure
        self._rms2 = 0.0       # rms EWMA of x^2
        self._noise = 0.0      # noise floor proxy (EWMA of |x|)
        self._peak = 0.0       # fast peak follower
        self._r_crest = 0.0    # rms EWMA for crest denominator

        # precompute exp coeff helpers
        self._exp = np.exp

    # --- helpers: one-pole EWMA update with tau ---
    def _ewma(self, state, x, tau):
        a = self._exp(-1.0 / (tau * self.fs))
        return (1 - a) * x + a * state

    def _alpha_from_tau(self, tau):
        tau = np.clip(tau, self.tau_min, self.tau_max)
        return self._exp(-1.0 / (tau * self.fs))

    # --- process a single sample ---
    def step(self, x):
        ax = abs(x)

        # noise floor proxy (slow)
        self._noise = self._ewma(self._noise, ax, self.tau_noise)

        # raw measure m: ABS or RMS
        if self.mode == "abs":
            m = ax
        elif self.mode == "rms":
            self._rms2 = self._ewma(self._rms2, x * x, self.tau_rms)
            m = np.sqrt(max(self._rms2, self.eps))
        else:
            raise ValueError("mode must be 'abs' or 'rms'")

        # crest trackers
        # fast peak follower (attack-fast, release via same τ here)
        self._peak = max(ax, self._ewma(self._peak, ax, self.tau_peak))
        self._r_crest = self._ewma(self._r_crest, x * x, self.tau_r_crest)
        crest = self._peak / (np.sqrt(max(self._r_crest, self.eps)))

        # adaptation terms
        # 1) slew acceleration (large when m jumps)
        rel_slew = abs(m - self._m_prev) / max(m, self.eps)
        g_slew = 1.0 / (1.0 + self.k_d * rel_slew)

        # 2) SNR damping (slow near noise)
        snr = m / (self._noise + self.eps)
        g_snr = 1.0 + (self.k_snr / (1.0 + snr))  # -> ~1 at high SNR, >1 at low

        # 3) crest shaping (faster for peaky signals beyond C0)
        crest_excess = max(0.0, crest - self.C0)
        g_crest = 1.0 / (1.0 + self.k_c * crest_excess)

        # choose baseline tau (directional) then modulate
        rising = m > self._e
        tau_base = self.tau_att if rising else self.tau_rel
        tau_eff = tau_base * (g_slew if rising else g_snr) * g_crest
        tau_eff = np.clip(tau_eff, self.tau_min, self.tau_max)

        # 1-pole with variable alpha
        alpha = self._alpha_from_tau(tau_eff)
        self._e = (1 - alpha) * m + alpha * self._e

        # book-keeping
        self._m_prev = m

        return self._e, tau_eff

    # --- process a block ---
    def process(self, x):
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        env = np.zeros(N, dtype=float)
        tau_eff_arr = np.zeros(N, dtype=float)
        for n in range(N):
            env[n], tau_eff_arr[n] = self.step(x[n])
        return env, tau_eff_arr
```

---

## usage: three sensible presets

```python
import numpy as np
import matplotlib.pyplot as plt

fs = 48000
t = np.linspace(0, 1.0, fs, endpoint=False)
# test: AM tone + noise
car = np.sin(2*np.pi*220*t)
mod = 0.2 + 0.8 * (0.5*(1+np.sin(2*np.pi*2*t)))
x = mod * car + 0.01*np.random.randn(len(t))

# 1) “fast musical” (synth envelopes / auto-wah control)
fast = AdaptiveAmplitudeAnalyzer(fs=fs, mode='abs',
                                 tau_att=0.003, tau_rel=0.080,
                                 k_d=6.0, k_snr=1.0, k_c=0.4, C0=3.0)
e1, _ = fast.process(x)

# 2) “stable AGC” (dynamics sidechain)
agc = AdaptiveAmplitudeAnalyzer(fs=fs, mode='rms',
                                tau_att=0.010, tau_rel=0.150,
                                k_d=3.0, k_snr=1.5, k_c=0.3, C0=2.5)
e2, _ = agc.process(x)

# 3) “noisy live input” (robust near floor)
robust = AdaptiveAmplitudeAnalyzer(fs=fs, mode='rms',
                                   tau_att=0.005, tau_rel=0.200,
                                   k_d=4.0, k_snr=2.5, k_c=0.6, C0=3.5)
e3, _ = robust.process(x)

plt.figure(figsize=(10,4))
plt.plot(t, x, alpha=0.35, label='signal')
plt.plot(t, e1, label='fast musical')
plt.plot(t, e2, label='stable agc')
plt.plot(t, e3, label='robust near floor')
plt.legend(); plt.tight_layout(); plt.show()
```

---

## why this works in synth/FX contexts

* **snappy on transients**: slew term collapses τ on big upward motion.
* **calm on tails & noise**: SNR term expands τ near the floor, avoiding chatter.
* **material-aware**: crest term biases τ based on peaky vs dense content.
* **directional**: independent attack/release baselines like analog followers.
* **stateful**: drop into block processing; it carries memory correctly.

---

## quick tuning cheatsheet

* need more snap? ↓ `tau_att`, ↑ `k_d`
* too fluttery in silence? ↑ `tau_noise`, ↑ `k_snr`
* harsh on dense pads? ↑ `k_c` or ↑ `tau_rel`
* overall too laggy? ↓ `tau_rel`, ↓ `tau_max` (keep some headroom)

---

want a **Faust version** of this (using `fi.avg_tau` and a little state machine), or a **Numba**-accelerated variant for realtime python?
