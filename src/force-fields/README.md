# Force fields

Each subdirectory here is one force field that chirality-kit can build against.
A force-field directory contains:

| File | Origin | Purpose |
|------|--------|---------|
| `chirality_kit_fragments.xdata` | chirality-kit | Functional-group fragment definitions |
| `chirality_kit_solvents.xdata` | chirality-kit | Solvent and ion definitions |
| `toppar/` | upstream force-field authors and chirality-kit | Topology and parameter files copied into every output folder |

The `.xdata` files are part of chirality-kit and are covered by the MIT licence
in the repository root. The upstream files in the `toppar/` trees are **not**
chirality-kit's work. Their provenance and licences are recorded below, along
with any separate chirality-kit additions.

Select a force field with the `force_field` key in the input JSON. Point at a
different tree entirely with `--force-fields /path/to/force-fields`; the folder
must be named exactly `force-fields`.

---

## Redistributed third-party parameters

### `charmm36m/`: CHARMM36m additive force field

- Upstream release: **`toppar_c36_jul24.tgz`**
- Source: <https://github.com/mackerell-lab/charmm36-force-field>
- Licence: MIT; see [`LICENSE.charmm`](LICENSE.charmm)
- Modifications: **none.** All 21 files are byte-identical to the release.

A later release, `toppar_c36_feb26.tgz`, is available upstream and changes four
of these files: `par_all36_cgenff.prm`, `top_all36_cgenff.rtf`,
`toppar_water_ions.str`, and `toppar_all.history`. Consider updating before
publishing work that depends on CGenFF or on water and ion parameters.

### `charmm36m-ndp/`: CHARMM Drude polarizable force field

- Upstream release: **`drude_toppar_2023`**, distributed as
  `toppar/drude/drude_toppar_2023.tgz` inside `toppar_c36_feb26.tgz`
- Source: <https://github.com/mackerell-lab/charmm36-force-field>
- Licence: MIT; see [`LICENSE.charmm`](LICENSE.charmm)
- Modifications: **none.** All seven upstream files are byte-identical to the
  release.

`cooq_drude_fg.str` is a separate chirality-kit parameter file for attaching
COO- groups to a Drude nanotube sidewall. It contains the required bond, angle,
dihedral, and improper parameters without altering an upstream file. Load it in
addition to `toppar_defs_2023.str` when using the COO- fragment.

### Please cite the force-field authors

If you simulate with these parameters, cite the force-field papers as well as
chirality-kit. The relevant references are listed in the header comments of each
`toppar/` file. For CHARMM36m, Huang *et al.*, *Nat. Methods* **14** (2017)
71–73; for CGenFF, Vanommeslaeghe *et al.*, *J. Comput. Chem.* **31** (2010)
671–690.

### A note on vendored copies

The CHARMM developers ask that users prefer the current files from the upstream
distribution rather than copies bundled with other software, because bundled
copies go stale and attract bug reports for issues already fixed upstream. The
copies here are a convenience so that a fresh clone runs without extra downloads.
For production work, and especially before publishing, check the upstream
release and pass a current tree with `--force-fields`.

---

## Placeholder force fields

Directories named `placeholder-*` are **not validated production
parameterisations**. They record elements, masses, and bond distances so that
structure and connectivity are generated correctly, but carry zero charges and
zero Lennard-Jones terms. Each one has a `toppar/README.md` describing what it
covers and what to supply. Do not simulate with them without providing real
parameters.
