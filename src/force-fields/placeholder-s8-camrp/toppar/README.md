# placeholder-s8-camrp

Rough placeholder parameters for packing cyclo-octasulfur (S8) rings into a
single-site carbon medium.

This folder is intentionally minimal. It records chemical elements, masses,
zero charges, zero Lennard-Jones placeholders, the crown-ring S8 geometry, and
an S-S bond distance of 1.2000 Angstrom.

It is not a validated production molecular mechanics parameterization:
structure and connectivity are generated correctly, but the system it produces
has no charges and no Lennard-Jones interactions. Supply real parameters before
simulating with it.

## Contents

| Solvent | Description |
|---------|-------------|
| `S8` | Eight-membered sulfur crown ring, eight bonds, no angles or dihedrals |
| `CAM` | Single-site carbon bead used as the surrounding medium |

`S_ring.xyz` is the bare S8 ring geometry the solvent definition was built from,
kept for reference.
