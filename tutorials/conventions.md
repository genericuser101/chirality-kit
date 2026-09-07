# chirality-kit Adding Force Fields

**chirality-kit** builds complete MD input files for functionalised nanostructures from a single Python script. Edit the `INPUT` dictionary at the top, run the script, and your output folder is ready.

--- 

## Contents 

1. [Force Field Folder](#1-force-field-folder)
2. [Additional Force Field Files](#2-additional-force-field-files)
3. [Fragments File Conventions](#3-fragments-file-conventions)
4. [Solvents File Conventions](#4-solvents-file-conventions)
5. [Current Limitations](#5-current-limitations)

--- 

## 1. Force Field Folder

Each force field lives in its own subdirectory under `force-fields/`, named in lowercase after the force field identifier string used in the `INPUT` dictionary. The directory name must match exactly what you pass to `"force-field"` in the `system` block (case-insensitive).

```
chirality_kit.py
force-fields/
    charmm36m/
        chirality_kit_fragments.xdata
        chirality_kit_solvents.xdata
        toppar/
    my-new-ff/
        chirality_kit_fragments.xdata     <- required if building a nanotube
        chirality_kit_solvents.xdata      <- required if using solvation
        toppar/
```

chirality-kit resolves both data files at runtime via `reinitialise()`. If a nanotube is requested and the fragments file is absent, or solvation is requested and the solvents file is absent, the run will raise a `FileNotFoundError` immediately; nothing is written.

The `toppar/` folder (or equivalent) contains the raw force-field parameter files that chirality-kit copies verbatim into the output directory. All files in the force-field folder **except** `chirality_kit_fragments.xdata` and `chirality_kit_solvents.xdata` are copied over. Name and organise these however the target MD software requires.

--- 

## 2. Additional Force Field Files

Any file placed inside the force-field folder that is not `chirality_kit_fragments.xdata` or `chirality_kit_solvents.xdata` is treated as a parameter file and is copied into the output directory unchanged. This is handled by `copy_other_FF_files()` at the end of the build pipeline.

This means you can include stream files, parameter files, topology files, or any supporting input the MD software requires simply by placing them in the folder. No registration is needed; the copy is unconditional and recursive for subdirectories.

---

## 3. Fragments File Conventions

The fragments file (`chirality_kit_fragments.xdata`) defines the atom-type coefficients and bond topology for every species that can be attached to or grown from the nanotube structure, including the nanotube carbon itself. It is only required when `"nanotube": true` in the `system` block.

### When do you need a fragments file?

If your force field will never produce a nanotube, such as a purely amorphous carbon system using S8 rings with no rolled graphene structure, you do not need a fragments file at all. chirality-kit only reads it when `nanotube` is active.

If your force field does build a nanotube, the file is mandatory.

### The Nanotube-Species entry: always first

The very first nanotube `Fragment` definition in the file **must** be `Nanotube-Species`, and its third token is the force field's supported nanotube type. chirality-kit reads this header during `Config.reinitialise()` and compares it with `nanotube.type` in the JSON input before generation starts.

```
Fragment Nanotube-Species CNT
...
```

Other valid examples include:

```
Fragment Nanotube-Species CNT-Drude
Fragment Nanotube-Species BNNT
Fragment Nanotube-Species MoS2NT
Fragment Nanotube-Species MoSSeNT
```

All nanotube atom types live in the `Masses` section of that one fragment. chirality-kit parses atom type, element, mass, charge, and optional `DRUDE (...)` values from the existing `Masses` line format, then builds `config.default_species_chain` in file order. The `atom_structure` value is assigned from the fragment header, not from a Masses comment token.

If the force field says `Fragment Nanotube-Species CNT-Drude` but the JSON requests `"type": "CNT"`, the build aborts early and asks you to choose a matching force field or change `nanotube.type`.

### H-term

If you want hydrogenation of nanotube edge carbons (via `"term-hydrogenate": true` in the `functionalisation` block, or via explicit `H-term` / `H+` groups), you must define an `H-term` fragment. chirality-kit routes end-cap hydrogen placement through the `H-term` fragment label internally.

```
Fragment H-term
...
```

Without this fragment, any run requesting hydrogenation will raise a validation error.

### DUMMY spacing slots

`DUMMY` is a reserved configuration value for ordered gaps between terminal
or ring functional groups. It is not a physical fragment, so do not add a
`Fragment DUMMY` section to a force-field file. It participates in attachment
site spacing and is removed before molecular files are written. With
`term-hydrogenate: true`, a dummy terminal position is subsequently capped by
the ordinary `H-term` fragment. `DUMMY` itself never consumes an index or
creates a record in any atom or bonded section.

### The c-idx placeholder

When writing bond records inside a fragment definition, the special keyword `c-idx` is used to refer to the attachment carbon on the nanotube, meaning the specific carbon atom index to which the fragment is being grafted. You do not know this index at file-writing time; chirality-kit substitutes it at build time when injecting each functional group.

In a bond line, either or both atom columns can be `c-idx`:

```
Bonds
1   bond_type   c-idx   1   #   CG2R61   CG2D1O
```

This tells chirality-kit: "atom 1 of this bond is the attachment carbon; atom 2 is the first new atom of this fragment (index offset 1)." Any atom column that is not `c-idx` is treated as a local index within the fragment and is offset by the atom-index offset at injection time.

### Adding a new functional group fragment

1. Add a new `Fragment YourGroupName` section after `Nanotube-Species` in the file.
2. Define all atom types, masses, charges, and LJ parameters in the section body.
3. In the `Bonds` subsection, use `c-idx` wherever the bond connects back to the nanotube carbon.
4. Register the fragment name in `Config.supported_functional_groups` if you want chirality-kit to validate it at input-parsing time (optional but recommended).

---

## 4. Solvents File Conventions

The solvents file (`chirality_kit_solvents.xdata`) defines every molecule or ion species that can be placed in the simulation box. Each species is a `Solvent` block.

### Naming convention: molecular solvents

For non-ionic solvents, the section header follows the pattern:

```
Solvent ATOMTYPE-MOLECULENAME
```

The atom-type prefix ties the entry to the force-field atom naming scheme and makes it unambiguous when multiple parameterisations of the same molecule exist. For example:

```
Solvent SWM4-NDP       <- SWM4 water model for the NDP polarisable field
Solvent TIP3P-Water    <- TIP3P water
Solvent SPCE-Water     <- SPC/E water
```

The `type` string you supply in the JSON solvation block must match the header name exactly:

```json
{ "type": "SWM4-NDP", "placement": "packed", ... }
```

### Setting up a salt: ionic placement

For ionic species (salts), each individual ion is defined as its own `Solvent` entry named by its chemical symbol alone:

```
Solvent K
Solvent Cl
Solvent Na
Solvent Ca
```

You do **not** add a separate entry for the salt pair. Instead, you specify the salt formula as the `type` in the JSON input and set `"placement": "ionic"`:

```json
{ "type": "KCl", "placement": "ionic", "count": 10 }
```

chirality-kit parses the formula string at runtime via `_process_salt_formula()` to extract the cation symbol, anion symbol, and their stoichiometric counts (e.g. `KCl` -> `K x 1, Cl x 1`; `Ca2Cl2` -> `Ca x 2, Cl x 2`). It then looks up each ion independently in the solvents file. Both ion entries must exist or the run will raise a `ValueError`.

Because each ion is just a standard `Solvent` block, the same ion entries (e.g. `K`, `Cl`) are reused across all salt formulae that contain them; you only define each ion once.

### LJ 12-6 coefficient injection

At the end of every build, chirality-kit calls `inject_lj_coefficients()` for every species present in the system, including both fragments and solvents. This rebuilds the `PairIJ Coeffs` section using Lorentz-Berthelot combination rules and appends any NBFIX entries found in the relevant blocks. For this to work, every `Solvent` entry must contain the LJ epsilon and sigma parameters for each atom type in the standard format.

### Adding a new solvent

1. Add a new `Solvent ATOMTYPE-MOLECULENAME` block to the solvents file following the naming convention above.
2. Include `Masses`, `Atoms`, `Bonds` (if applicable), and LJ coefficient subsections in the block body.
3. Use the exact header name as the `type` string in the JSON input.
4. For a new salt, add individual ion entries and rely on the formula parser; no salt-level entry is needed.

---

## 5. Current Limitations

- **Single attachment point only.** The functional group builder currently routes all groups through `_build_single_point_fg()`. Multi-point attachment (e.g. bridging groups spanning two carbons) is not yet implemented. The dispatcher in `write_functional_groups()` is reserved for future routing logic.

- **Salt formula parsing is element-symbol based.** `_process_salt_formula()` expects standard chemical symbols separated by optional stoichiometric digits (e.g. `NaCl`, `Ca2Cl2`). Non-standard or polyatomic ion formulae are not supported and will parse incorrectly.

- **Nanotube-Species must match JSON.** The first nanotube fragment must be `Fragment Nanotube-Species <Nanotube Type>`, and `<Nanotube Type>` must exactly match `nanotube.type` in the JSON input. Reordering the file, omitting this entry, or pairing it with the wrong JSON nanotube type causes an early validation error.

- **Force-field folder name is case-insensitive at lookup but must be lowercase on disk.** The folder name is lowercased before path resolution. Keep folder names entirely lowercase to avoid platform-specific path mismatches on case-sensitive filesystems.
