# Changelog

All notable changes to chirality-kit are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2026-09-14

Reference-structure validation release.

### Added

- A Structure Validation section documenting armchair, zigzag, and chiral
  carbon nanotubes against peer-reviewed structural values.
- Separate validation results for additive CHARMM36m and Drude-polarisable
  carbon nanotubes, with Drude comparisons restricted to carbon core sites.

### Internal validation

- Development-only tests generate unsolvated, unfunctionalised nanotubes with
  periodicity disabled and validate atom counts, bonding, coordination,
  diameters, chiral angles, axial lengths, and C-C spacing.
- The internal tests pass for both the standalone and split implementations;
  the manually maintained README tables record the corresponding results.

## [0.6.1] - 2026-09-07

First public release.

### Added

- A basic roadmap section in the README.
