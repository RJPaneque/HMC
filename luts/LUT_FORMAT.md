# Look-up table format

The binary layout of the tables of the Zenodo record and the sampling contract that `hmc.x` follows (the reference implementation is
`luts/src/lutgen/lut_contract.py`).

### LUT format

#### 1. Design goals

The LUT stores, for each material and grid energy, the conditional
distribution of the scattering angle and energy loss of surviving
positrons. The format is designed so that:

- every surviving history contributes to the tables (no subsampling);
- quantile functions are monotone, so they can be sampled by inverse-CDF
  interpolation;
- the angle and the energy loss are sampled jointly: the energy loss is
  conditioned on the angle bin;
- stopped histories contribute too: where a positron that does not survive
  the step ends up is stored with the same construction (§3bis), so the
  engine can place annihilations without a hinge-point approximation;
- each row carries an explicit validity code, so gaps are visible rather
  than padded with zeros;
- dimensions are stored in the file header, not as compile-time constants.

#### 2. Quantities

For a positron of kinetic energy E₀ entering a fixed path length STEP_CM of
material m, the engine needs:

* `p_ann(m, E₀)`: probability the positron stops (reaches ~0 kinetic energy)
  within the step.
* the joint distribution of (θ, u) for *surviving* positrons, where
  θ ∈ [0, π] is the polar deflection from the entry direction and
  **u = ΔE_kin / E₀ ∈ [0, 1)** is the fractional kinetic-energy loss.
* the joint distribution of (z, r) for *stopped* positrons: the end
  position relative to the start of the step, z ∈ [−STEP_CM, STEP_CM] along
  the entry direction and r ∈ [0, STEP_CM] perpendicular to it (azimuth
  uniform by symmetry).
* ρ_nom(m): the mass density the material was simulated at [g/cm³], so a
  voxel of known density can scale the step without the user supplying
  ratios.

u replaces ΔE in eV. Reasons: it is bounded, varies slowly with E₀ along the
log-E grid (so linear interpolation of quantile functions between rows and
neighbour pooling across rows are both well-behaved), and it makes rows of
different materials directly comparable. The engine recovers ΔE = u·E
with the *actual* particle energy E, not the grid energy.

#### 3. Representation: conditional quantiles

Per (material m, grid energy i), with N_θ ≥ 3 and N_u ≥ 3:

* `theta_icdf[m, i, j]`, j = 0 … N_θ−1: quantiles of θ at probability
  levels q_j = j / (N_θ − 1). Non-decreasing in j.
  Interior knots are the sample quantiles. The two end knots are
  *mean-preserving* (see "Edge knots" below), not the sample minimum and
  maximum.
* `u_icdf[m, i, b, k]`, b = 0 … N_θ−2, k = 0 … N_u−1: quantiles of u at
  levels q_k = k / (N_u − 1), computed over the survivors whose θ falls in
  θ-quantile bin b, i.e. between the sample θ quantiles at levels b/(N_θ−1)
  and (b+1)/(N_θ−1) (bins are assigned before the end knots are moved).
  Non-decreasing in k for fixed b. End knots are mean-preserving as for θ.

Edge knots. Linear interpolation spreads each interval's probability mass
uniformly, so the sampled mean within an interval is its midpoint. For interior
intervals this matches the data to within noise. The two edge intervals end at
the sample extremes, which are outliers of long tails. With the raw minimum and
maximum, the top θ interval alone biased the LUT mean θ high by up to 7 % in
soft tissue at 4.6 MeV and by up to 59 % in air, and mean u by 4–27 %.
`lut_contract.mean_preserving_ends` therefore sets

    knot[0]   = clamp(2·mean(x ≤ q₁)      − q₁,      lo, q₁)
    knot[N−1] = clamp(2·mean(x ≥ q_{N−2}) − q_{N−2}, q_{N−2}, hi)

with [lo, hi] = [0, π] for θ and [0, 1) for u. Each edge interval then
reproduces its events' conditional mean. The rest of the tail beyond the new end
knot is not represented, at a cost of 1/(N−1) of the probability. The sampling
contract (§4) is unchanged.

Each θ bin holds ≈ n_surv / (N_θ − 1) events; with n_surv = 10⁵ and
N_θ = 128 that is ≈ 790 events per conditional ICDF.

##### 3bis. Stopping position

Per (m, i), from the histories that stopped within the step, the same
construction on (z, r) with N_z ≥ 3 and N_r ≥ 3 knots:

* `z_icdf[m, i, j]`, j = 0 … N_z−1: quantiles of z, mean-preserving end
  knots clamped to [−STEP_CM, STEP_CM].
* `r_icdf[m, i, b, k]`, b = 0 … N_z−2, k = 0 … N_r−1: quantiles of r
  conditioned on the z-quantile bin b, end knots clamped to [0, STEP_CM].

Rows are classified like the survivors' (§5) with their own threshold
N_FULL_STOP (default 500) and the same K_MAX, into `svalid`; a row with no
stopped history is DEAD (zeros) and the engine falls back to the hinge point
there. At low energies every history stops and the tables are FULL; near the
annihilation threshold they are POOLED; at MeV energies the few stops are
in-flight annihilations, which the tables describe as well. Defaults
N_z = 32, N_r = 16: 116·(32 + 31·16)·4 B ≈ 245 KB per material, 7.3 MB for 30.

Quantile estimator: linear interpolation of the order statistics
(`numpy.quantile(..., method="linear")`). Pooled rows (§5) use the same
estimator on the pooled sample.

Resolution. A piecewise-linear ICDF with N equiprobable knots has a KS
distance ≤ 1/(N−1) from the sample it was built from. Measured on GEANT4
mat1 (2·10⁵ draws through the §4 contract vs. the raw survivors; split-half
noise floor ≈ 0.0035):

| N_θ / N_u | KB per material | KS_θ (23 keV … 4.6 MeV) | KS_u |
|---|---|---|---|
| 32 / 32   | 464  | 0.008 – 0.027 | 0.012 – 0.018 |
| 64 / 64   | 1856 | 0.005 – 0.013 | 0.003 – 0.005 |
| 128 / 32  | 1899 | 0.004 – 0.006 | 0.008 – 0.012 |
| 128 / 64  | 3741 | 0.002 – 0.006 | 0.003 – 0.005 |

The engine reads only 4 table values per step whatever N is (two knots of
each ICDF), so resolution costs memory, not time.

**Defaults: N_θ = 128, N_u = 64.** Size per material =
N_E·(N_θ + (N_θ−1)·N_u)·4 B = 116·(128 + 127·64)·4 B ≈ 3.7 MB; 30 materials
≈ 110 MB (fits comfortably in GPU global memory; CPU path is unaffected).
Both are header fields, so other choices need no code change.

#### 4. Sampling contract

Binding for compaction, engine and validator.

Inputs: material m, particle energy E, two independent uniforms r₁, r₂ ∈ [0,1).
Plus one uniform r₀ for the annihilation test.

```
1. Energy bracket.  Find i such that E_grid[i] <= E < E_grid[i+1]
   (clamp: E < E_grid[0] -> i = 0, t = 0;  E >= E_grid[N_E-1] -> i = N_E-2, t = 1).
   t = (ln E - ln E_grid[i]) / (ln E_grid[i+1] - ln E_grid[i]),  t in [0,1].

2. Annihilation.  p = (1-t)*p_ann[m,i] + t*p_ann[m,i+1].
   If r0 < p: annihilate (theta = 0, dE = 0, E -> 0). Stop.

3. Row weights.  Let v_i = valid[m,i], v_i1 = valid[m,i+1]  (valid != 0 means
   the row carries quantiles: FULL or POOLED, see §5).
     both valid     : w_i = 1-t, w_i1 = t
     only i valid   : w_i = 1,   w_i1 = 0
     only i+1 valid : w_i = 0,   w_i1 = 1
     neither        : treat as annihilated (this can only happen when
                      p_ann interpolates to ~1; it is a defensive branch).

4. Theta.  p1 = r1*(N_theta-1);  j = floor(p1), clamp j to [0, N_theta-2];
   w = p1 - j.
   th_lo = w_i*theta_icdf[m,i,j]   + w_i1*theta_icdf[m,i+1,j]
   th_hi = w_i*theta_icdf[m,i,j+1] + w_i1*theta_icdf[m,i+1,j+1]
   theta = (1-w)*th_lo + w*th_hi.
   Conditional bin b = j.

5. Fractional loss.  p2 = r2*(N_u-1);  k = floor(p2), clamp k to [0, N_u-2];
   w2 = p2 - k.
   u_lo = w_i*u_icdf[m,i,b,k]   + w_i1*u_icdf[m,i+1,b,k]
   u_hi = w_i*u_icdf[m,i,b,k+1] + w_i1*u_icdf[m,i+1,b,k+1]
   u = (1-w2)*u_lo + w2*u_hi;  clamp u to [0, 1).

6. dE = u * E;  E -> E - dE.  theta as sampled; azimuth uniform in [0, 2pi).
```

Step 2' (stopping position, when the LUT carries it; uniforms r₃, r₄ and an
azimuth):

```
2'. Row weights from svalid exactly as in step 3 (both / only i / only i+1);
    neither valid -> no table (the engine tallies at the hinge point).
    p3 = r3*(N_z-1);  j = floor(p3), clamp j to [0, N_z-2];  w = p3 - j.
    z = (1-w)*(w_i*z_icdf[m,i,j]   + w_i1*z_icdf[m,i+1,j])
      +    w *(w_i*z_icdf[m,i,j+1] + w_i1*z_icdf[m,i+1,j+1]);  bin b = j.
    p4 = r4*(N_r-1);  k = floor(p4), clamp k to [0, N_r-2];  w2 = p4 - k.
    r = (1-w2)*(w_i*r_icdf[m,i,b,k]   + w_i1*r_icdf[m,i+1,b,k])
      +    w2 *(w_i*r_icdf[m,i,b,k+1] + w_i1*r_icdf[m,i+1,b,k+1]).
    Clamp z to [-step, step] and r to [0, step]. The stopping point is
    P0 + z*d + r*(unit vector perpendicular to d at a uniform azimuth),
    P0 and d being the position and direction at the start of the step.
```

Interpolating quantile functions row-wise (steps 4–5) is a displacement
(Wasserstein-barycentre) interpolation between the two neighbouring
distributions, which is the appropriate way to blend distributions that
shift smoothly with energy. Interpolating densities would not be.

Determinism: given identical (m, E, r₀, r₁, r₂) and tables, every implementer
(compaction self-test, Fortran CPU, CUDA, Python validator) must return
bit-comparable θ and u to float32 rounding; likewise (z, r) for
(m, E, r₃, r₄).

#### 5. Row validity and gap handling

Per (m, i) the LUT stores `n_surv` (int32, number of surviving histories
*actually simulated* for that row) and `valid` (int32 code):

| code | name | meaning |
|---|---|---|
| 0 | DEAD | p_ann ≥ 1 − 1/n_hist (no survivors). No quantiles are stored (arrays hold 0). The engine never samples from it (step 3). |
| 1 | FULL | n_surv ≥ N_FULL. Quantiles from this row's own survivors. |
| 2 | POOLED | 0 < n_surv < N_FULL. Quantiles from survivors pooled over the same material's rows i−k … i+k (k grown from 1 until ≥ N_FULL pooled survivors or k = K_MAX). Pooled in (θ, u). Approximation, flagged. |
| 3 | SPARSE | 0 < n_surv < N_FULL and pooling to K_MAX still < N_FULL. Quantiles from whatever is pooled. Must be re-simulated. |

Defaults: N_FULL = 2000, K_MAX = 2. Raw files are named `mat1.dat` …
`mat30.dat` (unpadded). Material order is numeric, not lexicographic. Compaction never
writes zero padding into a row with code ≠ 0. The census of codes per
material is printed and written to `gap_census.csv`. The stopping-position
rows carry their own code (`svalid`, same table, threshold N_FULL_STOP) and
their DEAD rows mean "no stopped history", not a gap to re-simulate.

Adaptive simulation (GEANT4 driver, keywords `NSURV_MIN` and `NHIST_MAX`):
at each energy, histories are run until n_surv ≥ NSURV_MIN or
n_hist ≥ NHIST_MAX, so that POOLED and SPARSE rows become FULL on the next
build. The compaction step needs no change.

#### 6. Binary layout

Two files in `output/<engine>/`, little-endian, no padding. Every
array is row-major (C order) in the index order written.

##### `ann_LUT.raw`

| field | type | count |
|---|---|---|
| magic | char[4] | `"HMC2"` |
| version | int32 | 4 |
| kind | int32 | 1 (= annihilation table) |
| N_E | int32 | |
| N_M | int32 | |
| step_cm [cm] | float32 | path length per step the tables describe |
| N_z | int32 | stopping-position knots (0: no stopping tables) |
| N_r | int32 | stopping-position knots per z bin (0 with N_z = 0) |
| E_grid [eV] | float32 | N_E |
| rho_nom [g/cm³] | float32 | N_M (0 = unknown) |
| p_ann | float32 | N_M × N_E |
| n_hist | int32 | N_M × N_E |
| n_surv | int32 | N_M × N_E |
| valid | int32 | N_M × N_E |
| svalid | int32 | N_M × N_E |
| z_icdf [cm] | float32 | N_M × N_E × N_z |
| r_icdf [cm] | float32 | N_M × N_E × (N_z−1) × N_r |

Version 3 (still readable) ends after `valid` and has zeros where N_z and
N_r are.

##### `scatt-eloss_LUT.raw`

| field | type | count |
|---|---|---|
| magic | char[4] | `"HMC2"` |
| version | int32 | 4 |
| kind | int32 | 2 (= scattering/energy-loss table) |
| N_E | int32 | |
| N_M | int32 | |
| N_θ | int32 | |
| N_u | int32 | |
| step_cm [cm] | float32 | path length per step the tables describe |
| E_grid [eV] | float32 | N_E |
| theta_icdf [rad] | float32 | N_M × N_E × N_θ |
| u_icdf [–] | float32 | N_M × N_E × (N_θ−1) × N_u |
| valid | int32 | N_M × N_E |

Flat indices (0-based):

```
theta_icdf : ((m*N_E + i)*N_theta + j)
u_icdf     : (((m*N_E + i)*(N_theta-1) + b)*N_u + k)
z_icdf     : ((m*N_E + i)*N_z + j)
r_icdf     : (((m*N_E + i)*(N_z-1) + b)*N_r + k)
p_ann, n_surv, valid, svalid : (m*N_E + i)
```

A row's θ slice and its (N_θ−1)×N_u conditional block are contiguous, so
the engine reads two contiguous blocks (rows i and i+1) per step.

The two files must agree on `version`, `E_grid`, `valid` and `step_cm`; the
Fortran reader of `hmc.x` must abort on disagreement. Header
fields replace the compile-time constants `N_ENE`, `N_CDF`, `N_MATS` and
`STEP_CM`: the engine must move each particle by the header's `step_cm`, the
step the tables were built for. Version 3 added `step_cm` in place of a
reserved zero; version 4 uses the other reserved ints for N_z and N_r and
appends `rho_nom`, `svalid` and the stopping tables. Readers accept 3 and 4
and reject anything else.

#### 7. Reference implementation

The normative implementation of §3–§6 is `luts/src/lutgen/lut_contract.py`, and the
compaction in `build_tables.py` imports it rather than re-implementing it:

* `build_row(theta, u, n_theta, n_u, theta_range, u_range)` → (theta_icdf,
  u_icdf): §3, and with (z, r) and `stop_ranges(step_cm)` the §3bis tables
* `classify_and_pool(...)`: §5
* `write_ann_lut`, `write_scatt_lut` (`version=3` writes the old layout),
  `read_luts`, `combine` (materials in numeric order; checks grid and
  (N_θ, N_u, N_z, N_r) equality) and its inverse `select`: §6
* `sample(lut, m, E, r0, r1, r2)` and the vectorised `sample_many(...)`: §4,
  implemented literally so that the Fortran and CUDA ports can be checked
  sample-for-sample against it on a shared random stream; `sample_stop` and
  `sample_stop_many` likewise for step 2'.

#### 8. Design choices not adopted

* No energy-grid refinement near the annihilation threshold. The shared
  116-point log grid is kept, and the threshold is handled with adaptive
  NHIST. Revisit if the validator shows interpolation error concentrated
  there.
* No learned denoising of sparse rows. With 30 materials, re-simulation is
  cheaper and auditable.
* No change to the azimuthal sampling, rotation, or voxel traversal.

