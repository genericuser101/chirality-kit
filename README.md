# chirality-kit

Build molecular structures and topology files for functionalised nanotubes,
solvated channels, electrolyte boxes, and molecular packing workflows. The
repository contains both the original standalone script and an equivalent
split package.

## Roadmap

These features are planned, but are not implemented yet. I simply have not had
the chance to get to them, email me if you have any other suggestions:

- Fullerene tubes (capped CNTs).
- Multi-walled carbon nanotubes (MWCNTs).
- Full LAMMPS `.data` support the current output does not yet cover the full format because I do not use LAMMPS in my own workflow, `.xdata` borrows a lot from it, but needs a finished formalised implementation.
- CHARMM-style `.rtf` output, so generated structures can be reused in other workflows with their parameter definitions.
- Other force field parameters and other polarisable force fields could be on the cards if there is a strong need.

---

## Quick start

chirality-kit is not published on PyPI; you clone the repository and run it in
place. There is nothing to install but the one dependency. It requires Python
3.10 or newer and NumPy 2.0 or newer:

```bash
git clone https://github.com/genericuser101/chirality-kit.git
```

```bash
python3 -m pip install "numpy>=2.0"
```

Run either implementation from the repository root. Passing a JSON file is
recommended because the exact input and random seed are archived with the
output:

```bash
python3 src/chirality_kit_monolith/chirality_kit.py \
  --json examples/chirality_kit_input_dummy_gaps_dry.json
```

```bash
PYTHONPATH=src python3 -m chirality_kit \
  --json examples/chirality_kit_input_dummy_gaps_dry.json
```

Use `--seed 12345` for a reproducible run, `--force-fields
path/to/force-fields` to select a different force-field tree, and `--out-name
path/to/folder` to choose the output folder and generated filename prefix.

The same seed with the same input JSON produces byte-identical `.pdb` and `.psf`
files. The seed may also be given as a top-level `"seed"` key in the input JSON.
Resolution order, highest priority first:

1. `--seed` on the command line
2. `"seed"` in the input JSON
3. `metadata.rng_seed_used` in the input JSON
4. the `SEED` constant in the source
5. a freshly drawn random seed

Because every run records its seed as `metadata.rng_seed_used` in the
`chirality_kit_input.json` it writes to the output folder, and because that key
is honoured on the way back in, **any output folder replays itself**:

```bash
python3 src/chirality_kit_monolith/chirality_kit.py \
  --json previous_run/chirality_kit_input.json
```

reproduces `previous_run` exactly, with no `--seed` needed.

---

## Standalone and Split Layouts

chirality-kit ships two equivalent code layouts:

| Layout | Path | Best for |
|--------|------|----------|
| Standalone monolith | `src/chirality_kit_monolith/chirality_kit.py` | One-file use and simple copying |
| Split package | `src/chirality_kit/` | Importing, extension, and navigation by class or module |

Both layouts expose the same main classes. From the package, with `src` on the
Python path:

```python
from chirality_kit import Config, Geometry, File_Manager, Force_Field_Manager, Structure_Generator, Functional_Group_Generator, Solution_Generator, IO
```

The split package keeps shared runtime defaults in
`src/chirality_kit/_runtime.py`, including `INPUT`, `SEED`, and logging state.
The major classes live one per file:

```
config.py
geometry.py
file_manager.py
force_field_manager.py
structure_generator.py
functional_group_generator.py
solution_generator.py
io.py
```

The split package is not a replacement for the monolith. New behaviour should
stay compatible with both layouts.

---

## What does *chirality-kit* do?

Given a nanotube chirality (n,m) and a set of parameters, chirality-kit:

1. Generates the nanotube atomic structure using the graphene rolling construction
2. Attaches force-field-defined functional groups at terminal, ring, or loose sidewall sites
3. Hydrogen-terminates all remaining under-coordinated carbons
4. Fills the simulation box with solvent using Poisson-disk sampling
5. Places counter-ions for charge neutrality
6. Writes `.pdb`, `.psf`, `.xyz`, and `.data` files plus logs, summaries, and the selected force-field data

---

## Supported Functionality

### Force Fields

The force field is selected by the name of a folder under
`src/force-fields/`. The current repository includes:

| Folder | Nanotube or workflow | Status |
|--------|-----------------------|--------|
| `charmm36m` | Non-polarisable `CNT` | Production |
| `charmm36m-ndp` | Drude-polarisable `CNT-Drude` | Production |
| `placeholder-bnnt-lu2023` | `BNNT` | Placeholder - geometry only |
| `placeholder-mos2-luo2025` | `MoS2NT` | Placeholder - geometry only |
| `placeholder-mosse-rough2026` | `MoSSeNT` | Placeholder - geometry only |

**The `placeholder-*` trees are not parameterisations.** They exist so the
multi-species rollers have something to resolve atom types and bonds against.
Each defines only element names, masses, **zero charges, zero Lennard-Jones
terms**, and a single nominal bond length and stiffness; there are no validated
non-bonded, angle, or dihedral parameters behind them.

What you get from one is therefore **structure and connectivity, not a runnable
force field**: the `.xyz` coordinates are real, and the `.pdb`/`.psf` carry the
correct topology (atoms, bonds, angles, and residues), so the tube can be
visualised, measured, and used as a starting geometry. Handing that `.psf`
straight to an MD engine gives you an uncharged, LJ-free system, which is not a
physical model of the material. Supply your own parameters before running
anything with it.

Additional force-field folders can be dropped into the same directory, or a
different tree selected with `--force-fields`. Nothing about the folder is
hard-coded: the requested `nanotube.type` must exactly match the
`Fragment Nanotube-Species ...` header in the selected force field. Valid
functional groups and solvents are also read from that folder's
`chirality_kit_fragments.xdata` and `chirality_kit_solvents.xdata` files and
are checked before a structure is built.

### Nanostructures

| Structure | `nanotube.type` | Status |
|-----------|-----------------|--------|
| Carbon nanotube | `CNT` | Full structure, topology, functionalisation, and solvation |
| Drude carbon nanotube | `CNT-Drude` | Full structure, topology, functionalisation, and solvation |
| Boron nitride nanotube | `BNNT` | Geometry and connectivity only, via `placeholder-bnnt-lu2023`; no parameters, no functionalisation |
| Molybdenum disulfide nanotube | `MoS2NT` | Geometry and connectivity only, via `placeholder-mos2-luo2025`; no parameters, no functionalisation |
| Janus MoSSe nanotube | `MoSSeNT` | Geometry and connectivity only, via `placeholder-mosse-rough2026`; no parameters, no functionalisation |

### Functionalisation

Functionalisation uses three independent strategies under the
`functionalisation` key: `term`, `rings`, and `loose`. They can be combined;
sites claimed by an earlier strategy are not reused by a later one. The
selected force field determines which fragment names are valid.

#### Terminal groups

`term.groups` targets the under-coordinated carbons at the tube rims:

```json
"term": {
    "term-hydrogenate": true,
    "term-start-highest-x": true,
    "groups": [
        {"type": "COO-", "count": 2, "side": "-"},
        {"type": "CONH2", "count": 2, "side": "+"}
    ]
}
```

`side` is `"both"` by default, `"-"` for the negative-z end, or `"+"` for
the positive-z end. `count` applies independently to every selected side.
After explicit terminal groups are placed, `term-hydrogenate: true` caps all
remaining rim carbons with the force field's `H-term` fragment.
`term-start-highest-x: true` makes the first requested group on each end start
at the eligible carbon with the greatest x coordinate; equal x coordinates are
resolved by the greatest y coordinate. It defaults to `false`, in which case the
starting carbon is drawn from the seeded RNG, so with `false` the start still
varies between seeds, but is reproducible for a given seed like everything else.

#### Rings

`functionalisation.rings` is a list of independent circumferential patterns.
Each ring specification chooses several axial bands, lays out the requested
slots at equal target angles, and snaps those targets to the nearest
unoccupied nanotube atoms:

```json
"rings": [
    {
        "ring-count": 12,
        "ring-placement": "external",
        "ring-padding": true,
        "ring-phase-start-offset": 0.0,
        "ring-phase-increment": 0.5235987756,
        "groups": [
            {"type": "OH-", "count": 2},
            {"type": "DUMMY", "count": 2}
        ]
    }
]
```

| Ring field | Meaning |
|------------|---------|
| `ring-count` | Number of axial bands. The `groups` pattern is repeated at every band. |
| `ring-placement` | `"external"` points fragments away from the tube; `"internal"` points them into the pore. |
| `ring-padding` | `true` insets the first and last bands from the rims; `false` spans the full tube length. |
| `ring-phase-start-offset` | Rotation of the first pattern around the tube, in radians. |
| `ring-phase-increment` | Additional rotation per successive band, in radians. `0` produces columns; a nonzero value produces a twist or helix. |
| `groups` | Ordered fragment types and slot counts **per band**. |

In the example, each band has four target slots but only two real `OH-`
attachments. Across 12 bands that gives 24 attached groups. The two `DUMMY`
entries preserve gaps in the angular pattern.

#### Ordered gaps with `DUMMY`

`DUMMY` is a placement-only slot supported in `term.groups` and ring
`groups`. It participates in the ordered spacing pattern but is removed
before topology and coordinate files are written; no force-field fragment
named `DUMMY` is required. A ring dummy leaves its sidewall atom unchanged.
A terminal dummy is later capped with `H-term` only when
`term-hydrogenate` is enabled.

#### Loose sidewall groups

`loose.groups` places fragments on otherwise free sidewall atoms:

```json
"loose": {
    "groups": [
        {
            "type": "NH2-",
            "count": 6,
            "loose-placement": "random-external",
            "z-from": 0.35,
            "z-to": 0.65
        },
        {
            "type": "OH-",
            "loose-placement": "all-internal",
            "z-from": 0.90,
            "z-to": 1.0
        }
    ]
}
```

The supported modes are `random-external`, `random-internal`,
`all-external`, and `all-internal`. `z-from` and `z-to` are normalised axial
coordinates from 0 to 1 and apply to both random and all-site modes. `count`
is required for random modes and ignored for all-site modes. Loose entries
are processed in list order.

See the [full functionalisation tutorial](tutorials/tutorial.md#5-functionalisation)
for columns, helices, internal/external bands, mixed chemistry, terminal
patterns, and `DUMMY` gaps.

### Solvents and Ions

`placement: "packed"` uses Bridson Poisson-disk centres for molecular
solvents; `placement: "ionic"` places the ions parsed from a salt formula such
as `KCl`, with optional charge neutralisation. Common bundled solvent names
include `TIP3`, `SPCE`, and `SWM4`, but availability is force-field-specific
and validated from `chirality_kit_solvents.xdata`.

Packed molecules are independently oriented with Shoemake/Marsaglia unit
quaternions. The resulting molecular rotations are Haar-uniform on `SO(3)`:
any molecule-fixed axis is uniform on `S²`, including a uniform roll about
that axis. The rigid proper rotations preserve bond geometry and chirality.

#### `min_distance`

`min_distance` is the minimum separation in Ångström, default `2.0`. It does two
jobs: it is the cutoff for the post-injection overlap cull, and in the packed
path it is the floor on the Bridson exclusion radius, so solvent centres are
generated at least this far apart rather than only being culled afterwards.

Raising it therefore constrains how many molecules fit. If a requested `count`
or `concentration` cannot be reached at that floor the run fails with a message
naming `min_distance` as one of the things to change.

It applies only to pairs the group being injected is part of. Atoms already in the
system keep whatever separation their own injection established, so a later group
with a larger `min_distance` cannot retroactively cull pairs an earlier one already
resolved.

#### The bulk phase

The packed solvation group with the **highest** `priority` number is the bulk
phase, and its molecules are the only ones never checked against each other during
overlap removal. There is no key for this; it is derived from the group priorities
at validation and reported in the log.

The exemption is needed because bulk spacing comes from Bridson's centre-to-centre
radius, not from `min_distance`. Bridson separates molecular *centres* while
`min_distance` separates *atoms*, so neighbouring bulk molecules routinely sit
closer than `min_distance` atom-to-atom without anything being wrong; policing them
there would delete a large share of the box.

Only one group can hold the exemption, and only a packed one. Two groups sharing
the highest priority would never be checked against each other and would
interpenetrate silently, so that is a validation error. Ionic groups can never hold
it; ionic placement relies on the cull to catch two ions landing on each other.

To add a second solvent, give it a lower `priority` number than the bulk; it is
then culled against the bulk. That is correct while it stays dilute enough for its
own Bridson radius to clear `min_distance`. A true 50/50 co-solvent mixture is not
supported because both phases would need self-exemption while still being checked against
each other, which needs cross-phase neighbour search.

#### Multiple salts

Solvation groups are injected in **priority order, highest number first**. The
bulk phase goes in first and each later group, being more important, carves its
space out of what is already there. Groups sharing a priority keep the order they
were written in. Ions accumulate: a later ionic group is given the positions of
every ion already placed via the `.salts` record, so this holds even when a
packed solvent group sits between two salts, and rejects any position within
`min_distance` of one. If no valid position is found within 10,000 attempts the
run raises rather than placing an overlapping ion.

**Charge neutrality is applied by the last ionic group only**, meaning last in injection
order, so the ionic group with the *lowest* priority number. A `charge-neutrality: true`
on any earlier group is deferred, with a warning naming the group. This is
deliberate: each correction is computed from the system charge at the moment it
runs, so applying it once per salt means later corrections react to charge the
earlier ones already cancelled, and the system ends up charged. Only the last
group sees the fully accumulated charge.

The rule is that the last group *may* neutralise, not that it must. Setting
`charge-neutrality: false` on the last ionic group leaves the system
deliberately charged, and nothing overrides that.

Note that ions are excluded from a single axis-aligned bounding box around the
structure, not from a per-atom envelope, so they are kept out of the whole
cuboid enclosing the nanotube, including its interior channel.

### Output formats

| Requested extension | Output |
|---------------------|--------|
| `.pdb` | Coordinates for molecular visualisation; also creates `restraints.ref` |
| `.psf` | CHARMM/NAMD-style topology |
| `.xyz` | Coordinates without topology |
| `.data` | LAMMPS data file |

Every run also writes the internal `.xdata`, a `.log`, a `.summary`, the
resolved `chirality_kit_input.json` including the RNG seed, and a copy of the
selected force-field directory.

Two sidecar records are written alongside them:

| File | Contents |
| --- | --- |
| `.nanotube` | The covalent structure, including tube atoms **and functional groups**, despite the name. Written once, after functionalisation and before solvation. Absent when `"nanotube": false`. |
| `.salts` | One record per placed ion, rewritten after each ionic group. Ion cores only; the coincident Drude particle of a polarisable ion is not recorded. |

Format is `index type residue x y z`, one atom per line. Ionic placement reads
these instead of re-parsing the main `.xdata`, which runs to tens of thousands
of lines once solvent is in, and `.salts` is what lets a later salt see an
earlier one's ions. Both are regenerated from scratch every run. A stale copy
would silently exclude regions of a fresh box, so do not hand-edit them or copy
them between runs.

---

## Usage

The recommended input is a JSON file. This complete dry example combines
terminal groups, helical external rings, and random internal sidewall groups:

```json
{
    "settings": {
        "system_name": "auto",
        "verbose": false,
        "output_types": [".pdb", ".psf", ".xyz", ".data"],
        "adjust-ion-partial-charge": false,
        "bonds": true,
        "angles": true,
        "dihedrals": true,
        "impropers": true,
        "lone-pairs": true,
        "anisotropy": true
    },
    "system": {
        "force-field": "CHARMM36m",
        "size": "auto",
        "size-padding-xy": 14.0,
        "size-padding-z": 30.0,
        "nanotube": true,
        "functionalisation": true,
        "solvation": false,
        "pbc-override": false
    },
    "nanotube": {
        "type": "CNT",
        "n": 8,
        "m": 4,
        "repeats": 3
    },
    "functionalisation": {
        "term": {
            "term-hydrogenate": true,
            "groups": [
                {"type": "COO-", "count": 2, "side": "-"}
            ]
        },
        "rings": [
            {
                "ring-count": 3,
                "ring-placement": "external",
                "ring-padding": true,
                "ring-phase-start-offset": 0.0,
                "ring-phase-increment": 0.5235987756,
                "groups": [
                    {"type": "OH-", "count": 2},
                    {"type": "DUMMY", "count": 2}
                ]
            }
        ],
        "loose": {
            "groups": [
                {
                    "type": "H+",
                    "count": 4,
                    "loose-placement": "random-internal",
                    "z-from": 0.35,
                    "z-to": 0.65
                }
            ]
        }
    },
    "solvation": {
        "groups": []
    }
}
```

Save it as `input.json`, then run either layout:

```bash
python3 src/chirality_kit_monolith/chirality_kit.py --json input.json
PYTHONPATH=src python3 -m chirality_kit --json input.json
```

Both entry points accept the same options:

```bash
--force-fields path/to/force-fields
--seed 12345
--out-name path/to/folder
```

The `--force-fields` path must point to a directory literally named
`force-fields`. Without `--json`, the standalone script uses the `INPUT`
dictionary in `src/chirality_kit_monolith/chirality_kit.py`; the split package
uses the copy in `src/chirality_kit/_runtime.py`.

The six topology-section settings are required booleans. Setting one to
`false` prevents that section from being created in `.xdata`; its template
records and force-field types are not looked up. If every bonded section is
disabled, bond-network discovery is skipped too. Subsequent formats inherit the
filtered `.xdata`; format-required PSF blocks remain present with a zero count.

For automatic boxes, `pbc-override: false` clamps x/y and z padding to the
recommended minimum of 14 Å. Setting it to `true` permits smaller requested
padding but does not suppress periodic-image safety warnings.

---

## Output

Each run creates a new folder in the current working directory. `--out-name`
accepts either a folder name (relative to the current working directory) or a
path and overrides the JSON `system_name`. Without it, the JSON name is used;
`"auto"` generates the system name as before. Generated files use the final
folder's basename. If the chosen name already exists, `_(1)`, `_(2)`, and so
on are tried until the path is unique:

```
<system-name>/
├── <system-name>.xdata
├── <system-name>.pdb       # when requested
├── <system-name>.psf       # when requested
├── <system-name>.xyz       # when requested
├── <system-name>.data      # when requested
├── <system-name>.log
├── <system-name>.summary
├── restraints.ref          # when .pdb is requested
├── chirality_kit_input.json
└── <force-field>/           # copied parameter and xdata files
```

The summary and log record:

- nanotube geometry and atom count;
- terminal, ring, and loose functionalisation settings;
- solvation counts and achieved concentration;
- box dimensions and periodic-image checks;
- nanotube, solution, and total charge.

---

## Directory layout

Both layouts and the shared force-field folder are kept under `src/`:

```
src/
├── chirality_kit_monolith/
│   └── chirality_kit.py
├── chirality_kit/
│   ├── __init__.py
│   ├── __main__.py
│   ├── _runtime.py
│   ├── config.py
│   ├── geometry.py
│   ├── file_manager.py
│   ├── force_field_manager.py
│   ├── structure_generator.py
│   ├── functional_group_generator.py
│   ├── solution_generator.py
│   └── io.py
└── force-fields/
    ├── README.md          provenance and licensing of the bundled parameters
    ├── LICENSE.charmm     upstream licence for the CHARMM toppar trees
    └── .../
```

If you copy only the standalone script elsewhere, keep a `force-fields/` folder next to it or pass `--force-fields`.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or newer |
| NumPy | 2.0 or newer |

---

## Documentation and examples

- [Full tutorial](tutorials/tutorial.md)
- [Force-field fragment and solvent conventions](tutorials/conventions.md)
- [Force-field provenance and licensing](src/force-fields/README.md)
- [Additional JSON inputs](examples/)

The `examples/` folder is the best source of small, runnable inputs for
terminal patterns, `DUMMY` gaps, loose placement, and solvated systems.

---

## Licence

chirality-kit is released under the [MIT licence](LICENSE).

The repository also redistributes CHARMM topology and parameter files, which are
the work of the MacKerell lab and not of chirality-kit. They are covered by their
own MIT licence; see [`src/force-fields/README.md`](src/force-fields/README.md)
for provenance, attribution, and the force-field references you should cite.

---

## Citing chirality-kit

Citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it as a
"Cite this repository" button. Please cite the version you used, and cite the
authors of any force field you simulated with as well.

---

## Contact & support

stefan.zhikharev@warwick.ac.uk

This project is maintained by one person. For urgent support please use the email above.
