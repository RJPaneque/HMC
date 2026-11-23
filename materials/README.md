# materials/ — the 30 tissue materials

`materials/` generates PENELOPE `.mat` material-definition files for a list of tissue densities, using the Schneider, Bortfeld and
Schlegel (2000, *Phys. Med. Biol.* 45, 459) calibration of CT numbers to tissue composition.

**How it works.** You give it a list of mass densities (`input/materials.in`, one per line, g/cm³). For each density it looks up the
Hounsfield unit and the tissue composition of the corresponding bin of Table 6 of the paper (`src/schneider_materials.py`), then calls
PENELOPE's `material` program to write one `.mat` file per density.

```
python materials/generate_materials.py          # materials/output/<N>mats/mat1.mat, mat2.mat, ...  (N = number of densities)
pytest materials/tests                          # -m "not slow" skips the runs of `material`
```

- `output/30mats/` holds the 30 files the validation and the tables use (lung 0.027 g/cm³ to cortical bone 1.84 g/cm³), so you do not
  need to run anything to use them. The densities are `input/materials.in`.
- Regenerating them needs `material` (compiled from the PENELOPE source, PenSuite 2023) and a `pdfiles/` PENELOPE database linked into
  `materials/input/`; neither is distributed here (they come from the NEA Data Bank). The `.mat` files are under the notice in
  part 4.1 of `LICENSE`.
- The 24 materials of the first HMC are not a subset of these 30 (same densities, different compositions for 13 of them): see
  `legacy/README.md`.
