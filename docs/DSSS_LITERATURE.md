# DSSS detection and blind estimation — the literature, read for §B6 (2026-10-29)

Bill asked (2026-10-29) for the DSSS section of `ATKDSP_PLAN.md` to be
improved comprehensively, with the arXiv, IEEE and Chinese signal-processing
literature reviewed for ideas, and with one rule: **any detector built on a
method from a Chinese journal must be identifiable as such in ATK** (its own
drop-down, or a flag). So every entry below carries a provenance tag —
the venue, year and the authors' country as the paper states it — and the
plan's §B6 makes that tag a field of every kernel's `describe()` and result
struct. Tags: **CN** China, **US**, **FR** France, **FI** Finland, **SG**
Singapore, **ES** Spain, **DE** Germany, **UK**, **BE** Belgium, **TR**
Turkey, **IR** Iran, **AU** Australia, **ATK** ours. Where a paper's
affiliation was not visible to us the tag says "(affiliation to confirm)".

Evidence grades: **S** simulation only; **B** an SDR bench (wired or
over-the-air in a lab); **R** real captures or multiple real datasets. Every
blind-DSSS result found is S or B. Operating points are therefore design
targets to be *measured* in atkdsp's own harness (the phase-3 discipline:
binomial bands, a planted known-good emitter), never copied.

How this was gathered: arXiv's search page and API refuse automated
fetches, so papers were found by web search and then read on their own
pages; the HAL (univ-brest) deposits of Burel's group are behind a bot
wall, so Burel's statistics come from three sources that reproduce them
with equations (a METU master's thesis, Le Nir & Scheers at the Royal
Military Academy, Belgium, and a 2025 JMSE paper). Full-text access per
paper is in §H at the end. Nothing here is from a search snippet alone.

---

## A. Blind detection — "is there a spread signal here?"

### A1. Burel's fluctuation-of-correlation detector — **FR 2000** (G. Burel, Univ. Brest; reproduced with equations by the METU thesis **TR**, Le Nir & Scheers **BE 2015**, JMSE **2025**)

- Window p of T samples, P windows: `r(j,p) = (1/T) Σ_t y(pT+t)·y*(pT+t+j)`;
  the statistic is the second-order moment of the estimator,
  `ρ(j) = (1/P) Σ_p |r(j,p)|²`.
- Noise alone: `E|r(j,p)|² ≈ σ⁴/T`, flat in j for j ≠ 0 — the floor is the
  estimator's own variance, which is what makes the threshold analytic.
  With a DSSS signal, `s(t)·s*(t+T_s)` is coherent over a whole symbol, so ρ
  peaks at every multiple of the symbol period T_s although E[r] itself is
  zero (the data signs average it away; the fluctuation does not).
- Output SNR, as the thesis reproduces it:
  `SNR_out(dB) ≈ 4·SNR_in(dB) + 20·log10(W·T_s) + 10·log10(M) − 3`,
  with W·T_s ≈ chips per symbol and M the number of windows. **Each dB of
  input costs four of output; windows buy 10·log10(M).** That is the sizing
  law a UI can show ("at this SNR you need N windows").
- Threshold: floor mean + 4σ from the off-peak lags; false alarm 2 % by
  threshold alone, 0 % with the periodicity test (accept T_s only when ≥ 6
  peaks are equally spaced; the thesis takes the mode of inter-peak
  distances over 100 threshold levels). Le Nir: with L known choose
  L < T < 2L and require the top peak at L; unknown, require the peaks to
  be multiples of the first.
- Assumptions: none on the code (short or long, any user count — all users
  at one T_s add to the same peak; multirate gives several peak families),
  BPSK or QPSK chips, **carrier-offset tolerant** (an offset multiplies
  r(j,p) by e^{j2πΔf·j}; |·|² removes it — JWCN 2013 confirms), **no chip
  timing needed**, works oversampled, T_s comes out in samples.
- Reported (thesis; 64-chip BPSK, fs 200 MHz, 20 MHz receiver): 20 windows
  → detection at −11 dB; 50 windows → Pd = 1 at −13 dB (period right 99 %),
  Pd = 0.2 at −15 dB, lowest success −16.6 dB; period std ≈ 0.03 µs on
  20.48 µs; 4-path multipath costs ~2 dB; UMTS 256-chip QPSK at −5 dB with
  20 windows. Grade **S**.
- Cost: one FFT autocorrelation per window, |·|², a running mean —
  trivially streaming and chunk-invariant.
- **Adds:** this is the plan's step 1 and it survives the review intact.
  The sizing law and the equal-spacing periodicity test are new to the
  plan.

### A2. Covariance absolute-value (CAV) test — **SG 2008** (Zeng & Liang, I2R; arXiv 0808.2562)

- Single channel. L lags over Ns samples → L×L Toeplitz sample covariance;
  `T1 = (1/L) Σ_all |r_nm|`, `T2 = (1/L) Σ_diag |r_nn|`; signal if
  `T1/T2 > γ1`, `γ1 = [1 + (L−1)√(2/(Ns·π))] / [1 − Q⁻¹(Pfa)·√(2/Ns)]` — the
  threshold depends on L, Ns and Pfa only. **No noise-power input.**
- Needs correlated signal samples (oversampling or chip shaping: at exactly
  one sample per chip with rectangular chips it is blind — document that);
  white noise (an appendix pre-whitens coloured noise).
- Reported: L = 8–10, Ns = 5·10⁴, Pfa 0.1 measured 0.099 where an energy
  detector under 0.5 dB noise uncertainty gave 0.47; beats the ideal energy
  detector at −20 dB on oversampled narrowband signals; a closed-form
  sample count for a target (Pfa, Pd, SNR). Grade **S**.
- **Adds:** a "something structured is here" gate from the same lag
  products A1 already computes, holding its Pfa where a radiometer
  collapses. It does not find the period; A1 still does.

### A3. Eigenvalue tests MME / EME — **SG 2008/2009** (Zeng & Liang; arXiv 0804.2960) and the identifiability rule — **US 2007** (Rao & Edelman, MIT; arXiv 0705.2605)

- Stack L consecutive samples of M channels into ML-vectors, covariance
  over Ns snapshots. MME: `λmax/λmin > γ1`,
  `γ1 = ((√Ns+√ML)²/(√Ns−√ML)²)·(1 + (√Ns+√ML)^(−2/3)·(Ns·ML)^(−1/6)·F1⁻¹(1−Pfa))`,
  F1 = Tracy–Widom order 1 (F1⁻¹: 0.90 → 0.45, 0.95 → 0.98, 0.99 → 2.02).
  EME: mean power / λmin with a Gaussian threshold. A 2018 Tehnički
  vjesnik paper refines the centering (μ, σ with the ½ corrections) and
  reaches Pd = 1 at −15 dB with n = 1000, p = 6, Pfa 0.1.
- Reported: Ns 4·10⁴–1.8·10⁵ at −20 dB, L = 8–10, Pfa 0.1; MME > EME; both
  hold under 0.5–2 dB noise uncertainty. Grade **S**.
- **Rao–Edelman:** a population eigenvalue is identifiable from sample
  eigenvalues only if it exceeds `σ²(1 + √(m/n))`, m the vector dimension,
  n the snapshots; below that no eigenvalue method can see it and adding
  dimensions hurts.
- **Adds:** (i) a noise-power-free presence test with an analytic
  threshold, for one channel (L lags) or the Kraken's five; (ii) the
  design rule for the eigenvector step of code recovery — with m = samples
  per code period and n = periods, the rank-1 DSSS eigenvalue must clear
  σ²(1 + √(m/n)), which gives the **minimum number of code periods** for a
  chip SNR, and the honest "need more periods" refusal.

### A4. Chi-square cyclostationarity test, single and multiple cycles — **FI/US 2007** (Lundén, Koivunen, Huttunen, Poor; arXiv 0707.0909)

- Cyclic autocorrelation `R(α,τ) = (1/M) Σ x(t)·x^(*)(t+τ)·e^{−j2παt}`;
  stack real and imaginary parts at N lags into r (1×2N); Σ from
  frequency-smoothed cyclic periodograms; GLRT `T(α) = M·r·Σ⁻¹·rᵀ ~ χ²_2N`
  under H0. Several cycles: `D_s = Σ_α T(α) ~ χ²_(2N·Nα)`;
  `D_m = max_α T(α)` with CDF `(1 − e^{−x/2} Σ_{n<N} (x/2)ⁿ/n!)^Nα` — reject
  when the CDF exceeds 1 − p.
- Reported: OFDM at −7 to −9 dB, Pfa 0.05, 100 symbols. Grade **S**; the
  null distribution is general.
- **Adds:** replaces "a line above the floor after delay-multiply" with a
  stated Pfa at α = k·f_chip and α = k/T_code — and the max-statistic CDF
  gives the correct Pfa when *scanning many candidate chip rates*, which is
  exactly what a chip-rate search does.

### A5. Cyclic autocorrelation DSSS detector — **(EURASIP JWCN 2013, affiliation to confirm)**

- `λ = (T/N0²) Σ_{n=1..N} |R̂(0; n·T_b)|²`; H0 mean N, variance N (Gaussian
  threshold). 31-chip, N = 100 lags, T = 10⁴ samples: recognition > 0.9 at
  Pfa 10⁻³ and −13 dB; needs T_b known; cost N·T(1 + log₂T). Carrier
  offset does not change the recognition probability. Grade **S**.
- **Adds:** a confirmation statistic once the period is known; the CFO
  statement supports A1.

### A6. Fourth-order (Jarque–Bera) Gaussianity score — **ES 2025** (Cuevas, Santamaría et al.; arXiv 2510.25751)

- A warden against a 31-chip PN with Grassmannian codewords on the chips:
  skewness + kurtosis over 5·10⁴ chip-rate samples, a 2-dof χ² test; chosen
  over Shapiro–Wilk and others for cost. An energy detector's performance
  depends only on the spreading length, so the fourth-order statistic is
  what tells PSK-chip DSSS from Gaussian noise. OTA on a B210 matched the
  simulation qualitatively. Grade **B** (figures only, no SNR in text).
- **Adds:** a 20-line secondary indicator, sensitive to the chip
  constellation; Gaussian-looking chips defeat it, which is itself a
  classification hint.

### A7. Degree of cyclostationarity scalar — **US 2026** (Saha, Dagefu, Spasojević et al., ARL; arXiv 2602.17008)

- `DCS = Σ_{α≠0} ∫|S^α(f)|² df / ∫|S⁰(f)|² df` from a finite-time spectrum;
  threshold by Monte-Carlo search. Grade **S**; no analytic threshold.
- **Adds:** display only — a "cyclic energy fraction" meter, not a
  decision statistic.

### A8. Covert-communication wardens as reference detectors — **US 2015/2025** (Bash, Goeckel, Guha, Towsley; Bali, Bullock, Bash; arXiv 1506.00066, 2508.08380, 2506.02297)

- Spread spectrum lowers PSD, not power: a radiometer with known σ_w² sees
  power outside ±c·σ_w²·√n, hence O(√n) covert bits; the warden's
  noise-power uncertainty breaks that. The 2025 bench (X310, 12.5 MSPS,
  6·10⁶–7·10⁷ samples, 1000 trials per point) implements radiometer versus
  structure-matched LRT with empirically set Pfa* = 0.1 and measured Pmd;
  "the radiometer performs poorly even against a careless transmitter".
  Grade **B**.
- **Adds:** the design principle — every atkdsp detector states its
  behaviour under noise-power uncertainty (why A2/A3/A4, which need no
  noise power, matter) — and a sound recipe for empirical threshold
  calibration.

### A9. Reverse engineering of communications networks, a survey — **IR 2017** (Teimouri, Kakaei Motlagh; arXiv 1704.05432)

- Maps the chain and its sources: Tsatsanis–Giannakis 1997 (blind symbol
  recovery without the PN), Burel–Bouder 2000 (eigenvector sequence
  estimate, period known), Bouder 2002 (period + timing), Koivisto–Koivunen
  2007 (asynchronous multi-user short codes), Xu 2007 / Dong–Li 2008 (long-
  code period by delay-multiply), Daneshmand 2009 (periodic long code), Qiu
  2010 (short/long classification), Vlok–Olivier 2012/2014 (weak-signal
  presence, sequence-length at low SNR), Uysal–Filik 2015 (sparse long-code
  presence); and for LFSR reconstruction from noisy bits: Cluzeau 2007
  (algebraic, needs input bias), Liu 2011/12 (bias estimation), Xie 2014,
  Gautier 2002, Liu 2013 / Ma 2013 (polynomial + initial state). No
  numbers.
- **Adds:** the reading list, and the pointer that Berlekamp–Massey alone
  does not cover the noisy-chip case (see §D).

## B. Symbol period and timing

### B1. Period from ρ's peak spacing; timing from the covariance norm — **FR 2002** (Bouder, Azou, Burel; restated by Le Nir & Scheers **BE 2015**)

- Period: peak spacing of A1's ρ(j), refined by the parabolic peak.
- Timing offset, CFO-robust: `y(θ,p) = [y(pN+θ) … y((p+1)N+θ−1)]ᵀ`,
  `R_y(θ) = (1/P) Σ_p y(θ,p)·y(θ,p)ᴴ`, `θ_opt = argmax_θ ‖R_y(θ)‖_F`;
  cheaper: `‖R‖_F² = (1/P²) Σ_{p,q} |y_pᴴ y_q|²`, or track λmax only.
  Alternatively from the one-symbol-window EVD: `λ1 ≈ (L−t0)σ_s² + σ_n²`,
  `λ2 ≈ t0·σ_s² + σ_n²` → t0 and the SNR from the two top eigenvalues
  (Atlantis Press 2015).
- **Adds:** the timing search the plan's step 4 needs before any
  symbol-aligned method (and before Fast-ICA), done CFO-robustly.

### B2. Long-code symbol duration — **(Springer WPC 2020, affiliation to confirm; abstract only)**

- Sample covariances over several trial window lengths, diagonal dropped,
  second-order statistics of the squared Frobenius norm; peaks at the true
  T_s. Grade **S**, abstract only.
- **Adds:** the long-code symbol-rate estimate the plan's step 1 "long
  codes" line needs (hence the processing gain G = Rc/Rs). To be rebuilt
  from the idea, since the equations were not accessible.

### B3. Code-difference function for chip transitions — **(MDPI JMSE 2025, affiliation to confirm)**

- Accumulate half-window sums `Sum_a − Sum_b`, which peak at chip
  transitions (A²·Nc/4), then autocorrelate and take the second-order
  moment for the code period; 7-stage m-sequence, correct period 92 % at
  −14 dB with 5 accumulations. Grade **S**.
- **Adds:** a chip-transition statistic that is an alternative to
  delay-multiply for the chip rate; cross-check only.

## C. The code — blind spreading-sequence estimation

### C1. Eigenvector of period-long windows — **FR 2000** (Burel & Bouder); wide-window / SVD variants — **CN 2012** (Tongxin Xuebao / Journal on Communications), **(Atlantis Press 2015, affiliation to confirm)**

- Windows of N = T_s samples with unknown offset t0: `y_m = a_m·h0 +
  a_{m+1}·h−1 + n`, h0 = [c(t0..N−1), 0…], h−1 = [0…, c(0..t0−1)];
  `R = σ_a²(h0h0ᴴ + h−1h−1ᴴ) + σ_n²I` → the two dominant eigenvectors ≈ h0,
  h−1 with eigenvalues ∝ (N−t0), t0. Concatenate, take sign(). The
  relative sign between the two pieces is resolved by symbol consistency,
  or avoided by the **wide window**: windows of 2N overlapping by N; the
  principal eigenvector v (length 2N) holds one contiguous code;
  `k_max = argmax_k Σ_{i=k}^{k+N−1} |v(i)|²`; `ĉ = sign(v(k_max..k_max+N−1))`.
  The 2012 Chinese paper does it as an SVD of the stacked-window matrix
  (left singular vector = code, right = symbols, "no phase ambiguity").
- Assumptions: short code (period = symbol), T_s known (from A1/B1), chip-
  rate or oversampled, one dominant user (several users → one eigenvector
  each), **CFO removed first** (see C2), flat channel (in multipath the
  eigenvector is code ⊛ channel).
- Reported: 64-chip recovered at −5 dB (thesis); 127-chip Gold, 200
  windows, 100 % at −6 dB and correct k_max at −9 dB (Atlantis, with PAST).
  Grade **S**.
- **Adds:** the wide-window cut (no two-vector sign ambiguity, timing for
  free) replaces the plan's two-eigenvector alignment.

### C2. Carrier offset estimated *on the eigenvector* — **BE 2015** (Le Nir & Scheers, Royal Military Academy)

- Theorem: with offset α cycles per sample the dominant eigenvector is
  `v̂ = diag(1, e^{j2πα}, …, e^{j2πα(N−1)})·v_clean`. Estimate α from v̂²
  (BPSK) with a single-tone estimator (Kay's weighted phase average);
  CRB = 6/(ρ·L·(N²−1)); **attains the CRB at −8 dB on the eigenvector
  versus +12 dB on the raw signal's square**; |α| < 0.25 for BPSK. Then
  derotate and re-sign. Grade **S**.
- **Adds:** the 20 dB answer to "the eigen method needs the carrier
  removed first": estimate the offset where the processing gain has
  already been collected.

### C3. PAST subspace tracking instead of an EVD — **(Atlantis Press 2015)**, **CN 2012** (Systems Engineering and Electronics: nonperiodic / long-code PN)

- Projection-approximation subspace tracking gives the principal vector in
  O(N) per window (≈ 2PQ flops against O(N³) for an EVD); the 2012 paper
  uses a sliding window + PAST, then sorts and splices segment estimates
  for a nonperiodic (long) code, given code duration, chip rate and symbol
  rate. Grade **S**, abstract only for the 2012 paper.
- **Adds:** the streaming principal-vector tracker that makes long
  periods (N in the thousands) feasible in C without LAPACK, and the splice
  idea for long codes.

### C4. Two-cluster alternating sign-despread (k-means) — **CN 2021** (Ye et al., 信号处理 / Journal of Signal Processing; abstract only); EVD timing + two matrices — **CN 2022** (Ye et al., same journal; abstract only)

- 2021: estimate period, chip rate and start point; one segment per code
  period; 2-cluster k-means → cluster label = information bit; PN =
  coherent accumulation of sign-corrected segments. Equivalent to
  `ĉ ← Σ_m sign(y_mᵀ ĉ)·y_m`, iterated from the eigenvector: O(M·L) per
  pass, no EVD; claimed fast and noise-robust.
- 2022: eigenvalues of a first autocorrelation matrix give the out-of-step
  (timing) point; an eigenvector gives the information sequence;
  eigenvectors of a second matrix give the PN; claims better timing at low
  SNR than prior methods.
- Grade **S**; full texts behind a JS loader / 404.
- **Adds:** the decision-directed refinement the plan proposed is this
  method; **tag it CN** in the provenance field. The vote margin per chip
  (ours) is what makes it honest.

### C5. Fast-ICA on the symbol × chip matrix — **CN 2023** (MDPI *Information* 14(2):112, QPSK-DSSS); **UK 2019** (Lancaster, multi-rate long/short-code CDMA)

- MDPI: `r(k) = A[a(k)P1(k) + b(k)P2(k)] + n`, reshaped M symbols × L chips
  → R = A·S·P + N, an instantaneous two-source mixture. Whiten
  `Z = Λ_s^{−1/2}U_sᴴR`, maximise |kurt|: `W ← E{Z(WᵀZ)³} − 3W`, normalise,
  Gram–Schmidt deflation; `P̃ = sign(WᵀZ)`. Assumes L, chip rate and
  carrier known, symbol-aligned segmentation; sign/permutation ambiguity
  not addressed. Cost O(M³ + (M²+2M+24)L) vs EVD O(L³ + ML² + 4L) — cheaper
  when M < L. Results: L = 127–2047, M = 200; BER < 2 % above −5 dB; at
  −10 dB M ≈ 750 symbols gives a near-perfect code; residual CFO ≤ 100 Hz
  at 1 Mchip/s tolerated, 500–2000 Hz degrades; ≈ 5 % lower BER than EVD
  below −14 dB, equal above. Grade **S**.
- Lancaster: users on OVSF short codes under m-sequence scramblers at
  several rates; cut into long-code segments and fragments so the mixing is
  time-invariant, Fast-ICA with deflation per fragment, concatenate, double
  delay-and-multiply to strip the short code, **triple correlation**
  `C(p,q) = (1/L) Σ_m c(m)c(m+p)c(m+q)` matched against a precomputed table
  of all m-sequences of that degree (cyclotomic-coset heads), vote; a
  similarity matrix removes recovered users, repeat per rate. 100·1023
  chips, 100 runs, −16 to −10 dB, ≈ 13 dB better than their baseline at
  90 % correct; the table "is complicated to build". Grade **S**.
- **Adds:** the QPSK-chip and two-user short-code case (Fast-ICA,
  **CN**), and the one worked multi-user long-code method (**UK**), which
  is heavy and reference-only.

### C6. SVD of the window matrix — **CN 2014** (Shen & Wang, JEIT; abstract); principal eigenvector denoised, chip rate *and* PN from it — **CN 2017** (IET Signal Processing, Crossref abstract only)

- 2014: left singular vector = PN, right = information sequence, no
  synchronisation needed, chip rate and period known. 2017: segment, EVD,
  denoise the principal eigenvector, read the chip rate and the PN from it;
  a lower-complexity variant does better at low SNR. Grade **S**; no
  equations accessible.
- **Adds:** reading the chip rate off the eigenvector is a cross-check
  for the line scan once a code is in hand.

### C7. Periodic long code with a residual carrier — **CN 2019** (Yu, Zhang et al., 信号处理; abstract only)

- Model the periodic long code + residual carrier as a virtual multi-user
  short-code signal; EVD + a "fuzzy unitary matrix" gives a first estimate
  with the carrier still present; a DPLL tracks and removes it; the
  segmentation-phase ambiguity is resolved with m/Gold structure. Chip rate
  and period required. Grade **S**, abstract only.
- **Adds:** the idea that a residual carrier can be tracked *after* a
  coarse code estimate; details inaccessible, so reference only.

### C8. Subspace (polyphase) blind identification — **US 1997** (Tsatsanis & Giannakis; summary only)

- Chip-rate signal as a single-input P-output polyphase system h_i(l) =
  h(lP + i) with h = code ⊛ multipath; the Moulines subspace identifier
  (noise eigenvectors orthogonal to the channel's Toeplitz structure →
  smallest eigenvector of a stacked quadratic form), then zero-forcing
  recovers the symbols. Needs P, chip-rate sampling, an LTI channel (CFO ≈
  0), one user; output is code ⊛ channel.
- **Adds:** the multipath-tolerant counterpart of C1; reference only.

### C9. A blind CDMA receiver by ICA on chip-rate vectors — **US 2014** (Albataineh, Salem; arXiv 1408.0196)

- ICA columns are the users' spreading waveforms; three variants plus
  semi-blind RAKE hybrids; BER versus users/SNR in simulation only. Grade
  **S**.
- **Adds:** confirms ICA = multi-user alternative to one eigenvector; weak.

## D. The generator polynomial from noisy chips

### D1. Parity-check counting and fast correlation attacks — **FR 2004** (Mitton, DCSSI) with the standard Cluzeau / Meier–Staffelbach forms (**FR / CH**)

- Model `z_n = x_n ⊕ e_n`, P(e = 1) = p, bias ε = 1 − 2p. A weight-w
  multiple Q of the feedback polynomial gives a parity check whose noisy
  version holds with probability ½(1 + ε^w) (piling-up).
- **Identify the polynomial:** for a candidate Q of weight w over N
  positions, `Z_Q = Σ_n (−1)^{z_n ⊕ z_{n+i1} ⊕ …}`; true Q: E[Z] = N·ε^w,
  Var ≈ N; wrong Q: E[Z] = 0, Var = N. Decide Z > k√N (k ≈ 4–5 over many
  candidates) ⇒ **N ≈ k²/ε^{2w}**. For an m-sequence of degree r search the
  φ(2^r − 1)/r primitive polynomials directly (e.g. r = 10: 60 candidates;
  trinomials w = 3 need N ≈ 25/(1 − 2p)⁶ → p = 0.1: ≈ 95 chips; p = 0.3:
  ≈ 6100). A Walsh–Hadamard transform batches this over all low-degree
  candidates (a **CN 2019** sys-ele paper derives the misjudgment
  probability versus N, bias and weight).
- **Correct the chips:** per-bit statistic N_{0,k} = satisfied weight-k
  checks involving bit n; flip when `Σ_k N_{0,k}·atanh(ε^{k−1}) <
  ½ Σ_k N_k·atanh(ε^{k−1})`; iterate. Needs
  `Σ_k ε^{2(k−2)} N_k(n) > 1`; length bound `L > ((q−2)!)^{1/(q−1)} ·
  2^{r/(q−1)} · ε^{−2}`, information bound L > 2r/ε². Example r = 63,
  x⁶³ + x + 1: |ε| 0.5 → 640 bits, 0.3 → 1728, 0.2 → 5640, 0.1 → 28128.
- **Adds:** the honest end of the chain. After C1–C4 the chip error rate
  is low (< 5–10 %), so a few hundred chips and the counting statistic name
  the polynomial where Berlekamp–Massey on the raw noisy chips cannot (BM
  needs 2r error-free chips). The linear-complexity profile remains the
  tell that BM was lied to.

## E. Known-code acquisition and GNSS integrity (the half of B6 that is not blind)

### E1. Parallel code-phase search and its GLRT threshold — GNSS-SDR documentation (**ES**, CTTC), Borre et al. (**DK/US**), the Doppler-bin rule — **DE 2011/2014** (Geiger & Vogel; arXiv 1107.1972)

- PCPS: per Doppler bin f, `x[k] = x_in[k]·e^{−j2πfkT_s}`, `X = FFT(x)`,
  `Y = X·conj(D)` (D = FFT of the local code, precomputed — the conjugate
  is required), `R = IFFT(Y)/K`, `S = |R|²`; non-coherent sum over dwells,
  or a doubled coherent length to straddle a bit transition. Doppler grid
  N = ⌈2·f_max/step⌉ (defaults 5000 / 500 Hz; step ≈ 2/(3·T_coh)), a
  second pass at 125 Hz around the peak.
- GNSS-SDR's statistic `Γ = 2·K·S_max/P_in`, P_in the mean input power —
  CFAR: each normalised cell is Exp(1) under H0 (χ² with 2 dof; 2d dof for
  d dwells), so `γ = −ln(1 − (1 − Pfa)^{1/N_cells})`, N_cells = N_bins·K.
  Borre's peak-to-second-peak ratio (largest cell outside ±1 chip in the
  same bin; declare above ≈ 2.5) and the Tong up/down counter are the
  textbook alternatives.
- Geiger & Vogel: cell statistic |X|² non-central χ² with non-centrality
  2·T·C/N0·sinc²(Δf·T)·R²; expected cell energy over a uniform Doppler bin
  in closed form; cell Pfa = e^{−β/σ²}, Pd = Marcum Q1; global Pfa =
  1 − (1 − Pfa)^{NK}. **At 40 dB-Hz and T = 1 ms the best bin width is
  W·T = 0.5–0.7 (500–667 Hz)**; narrower bins raise adjacent-bin energy and
  global false alarms. Grade **S** (analysis).
- **Adds:** the acquisition kernel the plan's step 5 needs, with a
  threshold that states its Pfa and a defensible Doppler grid.

### E2. Post-detection combiners across unknown data bits — **ES 2018** (Gómez-Casco, López-Salcedo, Seco-Granados, UAB; arXiv 1810.04085, TAES)

- Correlator outputs `y_k = A·d_k·e^{jφ} + w_k` over N_nc blocks with
  unknown bits d_k and phase. Benchmarks: NPDI Σ|y_k|², DPDI |Σ y_k y*_{k−1}|,
  GPDIT, NPDISD (Σ|y_k|² + |Σ y_k²|). Proposed: BAPDI (a sum of Bessel I0
  terms over bit patterns), **MBAPDI = max over patterns of
  √(a_m² + b_m²)** (SNR-free), and the GLRT `Σ ln cosh((2A/σ²)(I_k cos φ +
  Q_k sin φ))` with `φ = ½·atan2(2ΣI_kQ_k, Σ(I_k² − Q_k²))`, high-SNR form
  `Σ |I_k cos φ + Q_k sin φ|`. N_nc 5–15; at Pfa 10⁻³ the proposed set beats
  NPDISD; coherent, DPDI and GPDIT collapse once bits are present. Grade
  **S**.
- **Adds:** the right combiner for any sign-ambiguous accumulation —
  weak-signal acquisition, and **verifying a blindly estimated code across
  data-modulated periods** (NPDI is the worst choice; DPDI breaks with
  bits).

### E3. Delay-and-multiply bit stripping before long accumulation — **CN 2020** (BeiDou acquisition, *Sensors*; arXiv 2003.00459)

- Delay-and-multiply the IF signal to remove the secondary-code and
  data-bit sign flips, then accumulate variable-length blocks (length
  corrected for code Doppler) before the FFT search; reaches 35 dB-Hz at
  ≈ 27.5 % of the computation of non-coherent integration. Grade **S**.
- **Adds:** the same nonlinearity used for the chip-rate line, applied as
  a bit-transition remover; **tag CN**.

### E4. C/N0 estimators and their validity — Van Dierendonck NWPR, VSM, Beaulieu, M2M4 (**US/CA**; numbers from *Sensors* 2020 and Falletti et al. **IT**)

- NWPR per 20 ms epoch of M 1-ms correlations: NBP = (ΣI)² + (ΣQ)², WBP =
  Σ(I² + Q²), μ = mean(NBP/WBP), C/N0 = (1/T_coh)·(μ − 1)/(M − μ). VSM:
  Z = I² + Q²; P̂_d = √(Z̄² − σ̂_Z²), P̂_n = Z̄ − P̂_d, C/N0 = (1/T_coh)·P̂_d/P̂_n.
  Measured: at 55 dB-Hz both have 1.2 dB std at 0.5 s averaging, 0.5 dB at
  5 s; **at 18 dB-Hz both bias low by ≈ 8 dB with 3–6 dB std** — unreliable
  below ≈ 25 dB-Hz; Beaulieu and M2M4 unbiased within 0.1 dB; NWPR/SNV
  underestimate under residual phase noise. Grade **B/R**.
- **Adds:** the C/N0 the acquisition and tracking kernels report, with a
  "> 25 dB-Hz" validity flag, and M2M4 as the estimator of choice.

### E5. GNSS power-distortion spoofing monitor — **US 2017/2018** (Wesson, Gross, Humphreys, Evans; Gross, Kilic, Humphreys; arXiv 1702.06554, 1712.04501, TAES)

- Power P_k = 10·log10 mean|r|² over 200 ms in 2 MHz (or the AGC
  set-point), σ_P ≈ 0.4 dB. Distortion `D_k = |ξ(−τ_d) − ξ(τ_d)|/σ_N0`, τ_d =
  0.15 chip, 41 taps over ±1 chip at 100 Hz averaged to 10 Hz; under H0 D_k
  is Rayleigh with σ_d² = 8τ_d. Four hypotheses (clean / multipath / spoof
  / jam) with Bayes-risk regions. PD-ML replaces D_k with the post-fit
  residual xᵀQ⁻¹x of a single-signal fit (amplitude, phase, code phase; 11
  taps). **Real data: TEXBAT's six spoofing recordings + 14 multipath + 4
  jamming + 3 clean.** PD: every attack flagged, < 0.6 % false alarm, but
  82 % of spoofing labelled jamming; PD-ML: 82.9 % spoofing right, 56 %
  multipath right, 0.31 % multipath-as-spoof. Grade **R** — the only item
  in this review with multi-dataset real-capture results.
- **Adds:** the RF detector behind `gnss_status.py`'s DENIED / UNTRUSTED
  states, two-tap version first, 11-tap ML upgrade.

### E6. Jamming and spoofing from COTS receivers — **(2025, Jammertest; arXiv 2509.13600, affiliation to confirm)**; C/N0 versus antenna tilt — **(2025; arXiv 2510.16229)**

- u-blox F9P: received power from the spectrum output (AGC-adjusted,
  temperature-calibrated), RFI flag at 1.92 dB above nominal (Gaussian fit
  to 10 days, Pfa 10⁻⁶/s); C/N0 ceiling ≈ 53 dB-Hz; jamming: C/N0 falls 1
  dB per dB of power; spoofing: C/N0 elevated relative to power. Jammertest
  2024: > 99 % no-RFI, 95–100 % jamming, **meaconing spoof only 53–67 %
  (SBAS) and ≈ 0–3 % (GPS)**. Grade **R**, honest about weak spoof
  detection. The tilt heuristic (real PRNs change 4–6 dB with tilt,
  spoofed 0.5–1.5 dB) has no rates reported. Grade **B**.
- **Adds:** field-calibrated thresholds for the receiver-observable
  version (ATK's GPS dongles report C/N0), and the warning that
  C/N0-only spoof detection is weak.

## F. Multi-antenna (the Kraken's five channels)

### F1. Canonical-correlation cyclic significance test — **US 2012** (Urriza, Rebeiz, Cabric, UCLA; arXiv 1211.0313 / 1210.8176, JSAC)

- M antennas, N samples: R_xx the sample covariance at lag τ0, R^α the
  sample cyclic cross-covariance at (α0, τ0) (xᵀ instead of xᴴ for
  conjugate cycles); the eigenvalues μ_i² of `R_xx⁻¹ R^α R_xx⁻¹ R^αᴴ` are
  squared canonical correlations; `T = −(N − M − 1)·ln Π(1 − μ_i²) ~ χ²`
  with M² (non-conjugate) or M(M + 1) (conjugate) dof. **The threshold
  depends on M only** — not N, not the noise — and the test is invariant to
  any linear noise transform (spatially coloured noise is free). BPSK at
  α0 = 2f_c, τ0 = 0; N = 1000–4000 at −10 dB, Pfa 0.1; ≈ 16 k complex
  multiplies against 296 k for SCF-based combining; mild loss under 3 dB
  noise uncertainty. Grade **S**; the null distribution is exact
  asymptotically.
- **Adds:** the cheapest calibrated DSSS detector for a five-channel
  capture; the conjugate cycle at 2f_c is present for BPSK chips and absent
  for QPSK chips, which classifies the chip modulation by itself.

### F2. Whitening with a noise-only covariance; condition-number tests — **(2019/2025; arXiv 1902.02483, 2411.17939, affiliations to confirm)**

- Whiten the sensing covariance with a noise-only training covariance
  (n, p ≥ m snapshots), test λmax/λmin or the largest generalised
  eigenvalue; closed-form finite-sample Pfa/Pd from random-matrix theory;
  the SCN keeps CFAR when the noise covariance is scaled by an unknown
  factor. Grade **S**.
- **Adds:** the right handling of noise coloured by an SDR's decimation
  filters, which breaks the white-noise assumption of A2/A3; needs a
  noise-only capture, which ATK can take (the Kraken's calibration path).

### F3. One-bit eigenvalue-moment-ratio sensing — **(2021; arXiv 2104.03481)**

- Sign-quantise I and Q; the one-bit covariance has unit diagonal;
  `ξ = 1 + (1/m) Σ` of the squared upper-triangle entries; m·n·(ξ − 1) ~ χ²
  with q = m(2m − 1) dof. Costs 2–3 dB against full precision. Grade **S**.
- **Adds:** low priority; a sign-only correlator for a very cheap front end.

## G. Not DSSS, but the templates are useful

- **Starlink Ku downlink blind pipeline** (arXiv 2210.11578, **US**): cyclic
  autocorrelation peak → N and N_g, frame period snapped to an integer Hz,
  cross-frame repetition → sync sequences, a k-means cluster score for
  timing/CFO. A good template for blind structure discovery in general.
- **LoRa SF/BW classification from STFT instantaneous frequency** (arXiv
  2309.04088): reliable only above −5 dB on synthetic data — the dechirp
  method in the plan's B6b is the better detector.
- **CycloDSP** (arXiv 2405.16911, **IT**): streaming second-order cyclic
  correlation estimators as a GNU Radio module, validated live on GMSK
  telemetry; an architecture reference for stream-mode cyclic correlation
  at a fixed α list.
- **Multi-cycle ratio detector for OFDM under noise uncertainty** (arXiv
  1311.6449); **sub-Nyquist cyclic-spectrum recovery** (arXiv 1604.02659);
  **single-channel co-frequency PSK separation** (arXiv 2609.11280); DS/CDMA
  receivers that assume known codes (arXiv 2412.12840, 1301.5006); a
  covert-comms warden with a parametric H0 and a learned H1 density (arXiv
  2602.09763) — portable idea, unproven here; a CNN on the spectral
  correlation function (arXiv 2003.08359, **TR**), no DSSS class.
- **Featureless signalling** (arXiv 1807.07260, **AU**, Shakeel, DST): claims
  DSSS with a repeating PN is exposed by ACF/PACF spikes at the code period
  and replaces the PN with an autoencoder waveform; no detector ROCs — the
  reminder that the PACF of the chip stream is a cheap period finder.

## H. What could and could not be read

- **Full text:** arXiv 0808.2562, 0804.2960, 0705.2605, 0707.0909,
  1211.0313 / 1210.8176, 1810.04085, 1107.1972, 2003.00459, 1702.06554,
  1712.04501, 2509.13600, 2510.16229, 2510.25751, 2602.17008, 1807.07260,
  2508.08380, 2506.02297, 1506.00066, 1408.0196, 1704.05432, 2104.03481,
  2411.17939, 1902.02483, 2405.16911 and the §G items; the Lancaster eprint
  (PDF text garbled, equations partly reconstructed); MDPI *Information*
  2023; the METU thesis (Burel's detector with equations and Monte-Carlo
  tables); Le Nir & Scheers 2015 (RMA); Atlantis Press 2015; EURASIP JWCN
  2013; Tehnički vjesnik 2018; Mitton / DCSSI 2004; GNSS-SDR docs; CEA
  patent US20100189161; MDPI JMSE 2025; MDPI Sensors 2020.
- **Abstract only:** IET SP 2017 (Shen & Wang); 信号处理 2019, 2021, 2022
  (full text is a JS loader, PDF link 404); JEIT 2014; sys-ele 2012, 2017,
  2019; IEICE 2005; Tongxin Xuebao 2012; Springer WPC 2020; Tsatsanis &
  Giannakis 1997.
- **Blocked:** every HAL / univ-brest deposit of Burel, Bouder, Azou and
  Berder (a bot wall), Semantic Scholar, PMC. **Not found** under its
  title: "A low-complexity near-optimal algorithm for blind estimation of
  PN sequences in DSSS" (2017); the nearest is a 2008 China Science Paper
  Online preprint (Wang, Wang, Chang) proposing two add/multiply-only
  statistics for synchronisation and sequence estimation, abstract only.
- Consequence for the plan: every **CN**-tagged method except the MDPI
  Fast-ICA paper is known to us from its abstract and from papers that
  cite it. Where §B6 adopts one (the k-means refinement, PAST for long
  codes, delay-multiply bit stripping) it is rebuilt from the idea and
  measured in our harness; the provenance tag says where the idea came
  from, and the measurement says whether ours works.

## I. What the review changes in §B6 — the short list

1. **Thresholds with a stated Pfa everywhere.** A2's covariance test as
   the presence gate (no noise power), A4's χ² cycle test for the chip-rate
   line (with the max-statistic correction for a scan), A3's Tracy–Widom
   threshold and the Rao–Edelman rule for the eigen step, E1's GLRT
   threshold for acquisition. The phase-3 discipline (binomial bands)
   measures each.
2. **The eigen step done right:** wide-window cut (C1), CFO from the
   eigenvector (C2), PAST for long N (C3), k-means / alternating sign
   refinement with a vote margin (C4, **CN**), Fast-ICA for QPSK chips and
   two users (C5, **CN**), MBAPDI / GLRT combiners to verify a code across
   data bits (E2).
3. **The polynomial through chip errors:** parity-check counting with
   N ≈ k²/ε^{2w}, a Walsh–Hadamard batch over low-degree candidates, FCA
   flips if the chip error rate is above ≈ 5 % (D1) — Berlekamp–Massey
   only on chips the vote margin certifies.
4. **The Kraken's channels as a detector:** F1 (threshold from the channel
   count alone, BPSK/QPSK chip classification for free), A3 on the spatial
   covariance; F2's whitening when a noise-only capture exists.
5. **GNSS integrity from the RF:** E5's power-distortion monitor is the
   detector behind DENIED / UNTRUSTED; E4's C/N0 validity flag; E6's
   warning that C/N0 alone detects meaconing poorly.
6. **Provenance as a field**, per Bill's rule.
7. **What did not survive:** the two-eigenvector alignment of the first
   draft (replaced by the wide window), Berlekamp–Massey as the only
   polynomial step, "a line above the floor" as a decision rule, and GPS
   as the only real acceptance (it is the hardest case and blind recovery
   cannot do it; 802.11b Barker first).

## J. Addendum, later the same evening — the 2025 crop Bill's search turned up

Bill sent a Google search ("2026 papers on new dsss detection methods");
the result page itself cannot be fetched, so the same query was run here.
What is new since the review above, all **CN**, all simulation or a
single RTL-SDR capture:

- **Intelligent detection of DSSS under false-alarm constraints with a
  noise score-pool threshold calibration — CN 2025** (Zhang, Tang, Sun;
  National University of Defense Technology, Changsha; *JEIT* 48(12),
  doi 10.11999/JEIT260414). An improved 1-D ResNet-18 on raw I/Q (1×7
  input kernel, no initial max-pool, no downsampling in the first
  residual stage, to keep weak temporal features; unnormalised input
  ≈ 1 dB better than RMS-normalised). **The useful part is not the
  network but the calibration:** the trained detector is run over a
  large pool of noise-only inputs, and the decision threshold is the
  quantile of that score pool at the target Pfa — an empirical CFAR
  wrapper around a classifier. Simulated GPS L1 C/A only; Pd = 1 at
  −8 dB for Pfa 0.01; ≈ 3–4 dB over autocorrelation detection; tested
  under coloured noise it was not trained on. Grade **S**. *Adds:* the
  score-pool calibration is exactly phase 3's measured-threshold
  discipline applied to a learned score, so any learned handle Undertow
  ever adds gets its Pfa the same way; tag **CN** if adopted.
- **Time–frequency energy squeeze for chip-rate estimation — CN 2025**
  (Zhao, Sun, Cai, Liu; Guilin University of Electronic Technology;
  *Sensors* 25(3):596). Morlet CWT, time-reassignment by a group-delay
  operator iterated to convergence (a "time-reassigned
  multisynchrosqueezing transform"), then the chip transitions are read
  as impulses whose spectrum has lines at k·Rc. Blind in chip rate and
  code; compared only against two wavelet power-spectrum methods, with
  delay-multiply and the cyclic spectrum discussed qualitatively ("heavy");
  Simulink sweeps plus one RTL-SDR capture. Grade **S/B**. *Adds:* a
  chip-transition sharpener before the α-spectrum; the delay-multiply
  and FDIFF nonlinearities already do that job in phase 3, so this is a
  cross-check candidate, not a replacement.
- **DSSS detection based on CNN — CN 2023** (*Sensors* 23(15):6691) and
  **Detection of DSSS signals based on deep learning (2022)**: classifiers
  on simulated data; no calibrated Pfa; **S**. Reference only.
- **Wavelet decomposition + delay multiplication — CN 2018** (IEEE
  8538308): a wavelet band split before delay-multiply — the scale idea
  in a crude form, abstract only.
- **PN-sequence period estimation for underwater DSSS at low SNR — 2025**
  (*JMSE* 13(12):2318) is the code-difference-function paper already in
  §B3.

**And Bill's question — "cyclostationary detection of DSSS: lots of
papers?"** Yes; it is the mainstream of the field, from Gardner's 1988
"Signal interception: a unifying theoretical framework for feature
detection" onward — chip-rate cyclic features, the spectral correlation
function, the strip spectral correlation analyser for wideband survey,
and a long tail of low-SNR variants. Phase 3's `atkdsp_cyclic` *is* that
method in streaming form (one Goertzel per candidate rate, the coherence
test, measured thresholds). What that literature does not do, and what
B6 and B6c add on top of it: a chip-rate *scan* with a true Pfa over the
scan (literature A4's max-statistic form), the scale constraint that
ties the line's position to the channel it is found in, the second
receiver, and the long watch with drift compensation.

## K. The neural-network line (Bill: "I remember reading papers on this … I know some use CNN")

What exists, read tonight:
- **DSSS Signal Detection Based on CNN — CN 2023** (Gu, Liu, Xu, Zhang,
  Lu; Zhejiang Sci-Tech / Zhejiang University; *Sensors* 23(15):6691).
  Raw I/Q, 1×2048×2, power-normalised; a 6-layer 1-D CNN with global
  average pooling and a softmax yes/no. Trained on BPSK, 127-chip
  m-sequence, 1 Mc/s at 2 MSPS, −20 to +10 dB; tested on 255–2047 chips,
  Gold codes and QPSK. Pd = 1 at −8 dB (BPSK, 127) against the plain
  single-capture autocorrelation detector's −4 dB — "≈ 4 dB"; at −14 dB
  Pd 0.71 vs 0; QPSK 4 dB worse. **No false-alarm rate, no ROC, no
  threshold rule** — the paper says it did not study the noise-only
  case. Simulation only. Grade **S**.
- **Intelligent detection with a noise score-pool CFAR calibration — CN
  2025** (NUDT; §J above): the same idea with a 1-D ResNet-18, and the
  one thing the line was missing — a stated Pfa, from the quantile of
  the network's scores over a noise-only pool. −8 dB at Pfa 0.01 on
  simulated GPS C/A; 3–4 dB over autocorrelation. Grade **S**.
- Cited by both and not read in full: Wei et al. 2022 (raw I/Q into a
  CNN, and a "CNN-CORR" hybrid that feeds the autocorrelation in to cut
  cost); an energy-detector-then-cyclic-spectrum-image CNN for DSSS UWB;
  Tekbiyik et al. 2020 (**TR**, arXiv 2003.08359) on spectral-
  correlation-function images; DetecSHUN (WiSec 2024, "signals hidden
  under the noise" — paywalled, not read); a 2025 ResNet/DenseNet
  communication-signal detector (*SN Applied Sciences*).

What the line is worth, honestly:
1. **The baseline they beat is weak.** "4 dB over autocorrelation" is
   against the one-capture autocorrelation test; Burel's detector with
   50 windows reaches −13 dB on a 64-chip code (literature A1), i.e. the
   classical fourth-order path already sits 5 dB past the CNN's −8 dB
   when it is allowed to integrate. None of the CNN papers compares
   against Burel at matched integration, the eigenvalue tests, or the
   χ² cycle test.
2. **A network on raw I/Q learns fourth-order features.** It has one
   receiver and no code; the only statistics available to it are the
   same repetition, transition and correlation structures the classical
   detectors compute explicitly. It can learn the best *combination* of
   them for the training distribution — which is why it edges the plain
   autocorrelation test — and it cannot cross the second-order line that
   needs an independent copy of the signal (B6c). Physics does not care
   what the detector is made of.
3. **Trained on one chip rate, one length, one shape**, with
   generalisation measured only across lengths and Gold/QPSK; nothing on
   unknown chip rate or bandwidth, which is the B6c problem. A survey
   detector would need one network per scale or a scale-invariant input —
   the α-spectrum and the ACF are exactly those inputs, computed by the
   kernel.
4. **False alarms:** only the 2025 paper states a Pfa, and its mechanism
   — the score-pool quantile — is the measured-threshold discipline phase
   3 already uses, applied to a learned score.

Where a network does belong, then (B6c says the same): **above the
kernel, not inside it.** atkdsp computes the calibrated statistics per
cell (ACF triangle, α-spectrum lines, PSD shape fit, cross-spectrum,
period peak); a small learned *fuser* on those features — trained on the
harness's synthetic signals the way TorchSig trains, in ATK's Python
beside the diffusion toolkit, calibrated by a noise score pool, and
tagged with its provenance (the calibration idea is **CN 2025**; the fuser
is **ATK**) — can replace hand-set fusion weights and classify what fired
(chip shape family, chip modulation, long or short code). It never
replaces the statistics, it never runs in C99, and it is judged in the
same harness, against the same planted emitter, by the same three numbers
(depth, time-to-detect, false alarms over a night).

## L. Addendum, 2026-10-30 — blind chip-rate estimation, and two Frobenius ideas

Bill asked (02:41) for a search on blind chip-rate detectors, after the
bootstrap (`atkdsp.h` §30) showed the chip count is the hard part below the
floor: in a survey cell the filter cuts the code's spectrum at ~2/3 of the
chip rate, so the chip-rate line is weak.

- **Bandpass filtering before the cyclic moment — KR 2020** (Kim, Yoon, Kang,
  Moon, Park; *J. KICS* 45(3), 568–576; abstract read). A DSSS signal's
  cyclic spectrum has lines at every multiple of the code rate, which bury
  the chip-rate line; they restrict the search to a band set by a bandwidth
  estimate. **Confirms** the trap the bootstrap met, and the survey's scale
  window is the same remedy. No SNR floor stated.
- **Carrier-aggregated DSSS, cyclostationary — TR 2024** (Görçin, ITU /
  TÜBİTAK BİLGEM; *Electronics* 13(18), 3743). Carrier count, spacing and
  chip rate jointly; tested to 0 dB only. Above the floor.
- **Time–frequency energy squeeze — CN 2025** (Zhao, Sun, Cai, Liu, Guilin UET;
  *Sensors* 25(3), 596). Already in §J; rectangular chips only.
- **Le Nir & Scheers — BE** (Royal Military Academy; *Electronics Letters*).
  The eigenvector-carrier idea `dsss_code` uses; it does **not** estimate the
  chip rate (one sample a chip assumed).
- Reading the chip rate off the principal eigenvector: **CN 2017** (§C6). The
  bootstrap's two scores of the eigenvector — the chip-centre envelope and
  the transition line, decoded and confirmed by naming — are ATK's form.

Nothing found handles band-limited chips below the floor.

**Two ideas Bill relayed (03:08), sound, for `dsss_code` (v0.23):**
1. **2-adic decimation averaging.** For an m-sequence of length L = 2^r − 1,
   c[2^m n mod L] is a cyclic shift of c[n] (the Frobenius automorphism). So
   the r decimations of the soft-chip vector, aligned by their cyclic shifts
   (found by correlation, which is itself evidence of an m-sequence), give
   each chip r looks at *different* samples — independent noise — about
   10·log₁₀ r dB per chip (7 dB at 31 chips, 10 dB at 1023). Not for Gold
   codes (each component shifts differently). The technique is known in
   coding as permutation / automorphism-ensemble decoding.
2. **The Frobenius parity stack.** Over GF(2), g(x)^(2^m) = g(x^(2^m)): every
   parity check's squares are checks too, at power-of-two lags — the
   low-weight-multiple trick of fast correlation attacks (Meier &
   Staffelbach, CH 1989; already cited for parity counting). Summing the
   parity statistic over the orbit gives about √r more deflection when
   naming the generator through chip errors.

Rejected from the same exchange, with reasons in the conversation: quantile /
Wasserstein envelope tests (a robust variant of the chip-rate envelope
feature, not deeper), modulational-instability "amplification" (it amplifies
noise in its gain band equally), a Bessel-score band concentration (linear at
low SNR: a radiometer), a two-channel geometric phase (a high-order statistic
from three samples; the averaged cross-spectrum is the right two-receiver
statistic), and min-sum parity on raw chips (the piling-up lemma it cites is
the cubic loss).
