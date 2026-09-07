from __future__ import annotations

import datetime
import json
import logging
import os
import re
import shutil
import string
import time
from collections import defaultdict
from itertools import combinations, permutations
from typing import TYPE_CHECKING

import numpy as np

from ._runtime import SEED, my_log_file


class Config:

    def __init__(self):

        '''
        Initialise the Config object with default force field settings, file paths,
        and supported parameter lists. The force-fields directory defaults to
        ../force-fields next to this package and can be overridden by assigning
        self.force_fields_root before calling reinitialise().

        Parameters:
        ----------
        None

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        self.verbose = False

        #======================================================================
        # INITIALISATION FUNCTIONALITY
        #======================================================================

        self.section_order = [
            "Extended Datafile - chirality_kit",
            "Masses", "PairIJ Coeffs", "Atoms",
            "Bond Coeffs", "Bonds",
            "Angle Coeffs", "Angles",
            "Dihedral Coeffs", "Dihedrals",
            "Improper Coeffs", "Impropers",
            "Lone Pairs", "Anisotropy",
            "END"
        ]
        self.enabled_xdata_sections = {
            "Bonds", "Angles", "Dihedrals", "Impropers",
            "Lone Pairs", "Anisotropy",
        }
        self.functionalisation_requested = False

        self.section_spacing = {
            "Masses"        : [12, 9, 3, 16, 5, 10, 30, 8, 16],
            "PairIJ Coeffs" : [12, 9, 10, 10, 10, 10, 4, 16, 16],
            "Atoms"         : [12, 9, 7, 16, 15, 15, 15, 4, 16],
            "Bond Coeffs"   : [12, 9, 9, 4, 16, 16],
            "Bonds"         : [12, 7, 10, 10, 4, 16, 16],
            "Angle Coeffs"  : [12, 13, 13, 13, 13, 4, 16, 16, 16],
            "Angles"        : [12, 13, 13, 13, 13, 4, 16, 16, 16],
            "Dihedral Coeffs": [12, 13, 10, 13, 4, 16, 16, 16, 16],
            "Dihedrals"     : [12, 13, 13, 13, 13, 13, 4, 16, 16, 16, 16],
            "Improper Coeffs": [12, 13, 10, 13, 4, 16, 16, 16, 16],
            "Impropers"     : [12, 13, 13, 13, 13, 13, 4, 16, 16, 16, 16],
            "Lone Pairs"    : [12, 9, 3, 15, 15, 15, 4, 16, 36],
            "Anisotropy"    : [12, 9, 15, 15, 4, 16, 36]
        }

        self.supported_file_inputs       = [".pdb", ".psf", ".xyz", ".data"]
        self.supported_file_outputs      = [".pdb", ".psf", ".xyz", ".data"]
        self.supported_functional_groups = ["COO-", "C2H5", "H+", "OH-"]
        self.drude_particle_FFs          = ["charmm36m-ndp", "charmm36m-pdp",
                                            "p-charmm36m-ndp-misra", "p-charmm36m-pdp-misra"]

        #==================================================================================
        # HARD-CODED DEFAULTS FOR NANOTUBE SPECIES, ION NAMES, AND DRUDE PARTICLE NAMES
        #==================================================================================

        self.default_species_chain = [
            {
                "atom_type": "CG2R61",
                "atom_element": "C",
                "atom_mass": 12.0110,
                "atom_charge": 0.0,
                "atom_structure": "CNT",
            }
        ]
        self.default_species_by_type = {}
        self.default_species_by_element = defaultdict(list)
        self.default_species_primary = None
        self.default_species_nanotube_type = None
        self._refresh_default_species_lookups()

        # bond settings by element; reinitialise updates these
        self.bond_params = {
            
            # single-element
            "C"  : {"bond_length": 1.44, "bond_cutoff": 1.60, "atom_type": "C",  "element": "C",  "mass": 12.0110, "charge": 0.0},  # sp2 graphene/CNT
            "Si" : {"bond_length": 2.35, "bond_cutoff": 2.55, "atom_type": "Si", "element": "Si", "mass": 28.0860, "charge": 0.0},  # silicene
            "Au" : {"bond_length": 2.88, "bond_cutoff": 3.10, "atom_type": "Au", "element": "Au", "mass": 196.967, "charge": 0.0},  # gold NT
            "Ag" : {"bond_length": 2.89, "bond_cutoff": 3.10, "atom_type": "Ag", "element": "Ag", "mass": 107.868, "charge": 0.0},  # silver NT

            # shared geometry, with one entry per element
            "BN"   : {"bond_length": 1.45, "bond_cutoff": 1.65, 
                      "B" : {"atom_type": "B",  "element": "B",  "mass": 10.8110, "charge": 0.0}, 
                      "N" : {"atom_type": "N",  "element": "N",  "mass": 14.0067, "charge": 0.0}},  # h-BN
                      
            "MoS2" : {"bond_length": 2.73, "bond_cutoff": 2.95, 
                      "Mo": {"atom_type": "Mo", "element": "Mo", "mass": 95.9600, "charge": 0.0}, 
                      "S" : {"atom_type": "S",  "element": "S",  "mass": 32.0600, "charge": 0.0}},  # MoS2
        }

        self.ions  = [
            "LIT", "SOD", "MG",  "POT", "CAL", "K", "Cl",              # monovalent / divalent cations
            "RUB", "CES", "BAR", "ZN", "CAD",               # alkali / heavy cations
            "CLA",                                          # halide anions
        ]

        self.dion_particle_by_core = {
            "LID": "DLID", "SODD": "DSOD", "MAGD": "DMG",
            "POTD": "DPOT", "CALD": "DCAL", "RBD": "DRUB",
            "CSD": "DCES", "BAD": "DBA", "ZND": "DZN",
            "FAD": "DF", "CLAD": "DCLA", "BRAD": "DBR",
            "IAD": "DI", "SRD": "DSR",
        }

        self.dion_pdb_names = {
            "LID": "LIT", "SODD": "SOD", "MAGD": "MG",
            "POTD": "POT", "CALD": "CAL", "RBD": "RUB",
            "CSD": "CES", "BAD": "BA", "ZND": "ZN",
            "FAD": "F", "CLAD": "CLA", "BRAD": "BR",
            "IAD": "I", "SRD": "SR",
        }

        self.dions = list(self.dion_particle_by_core)
        self.dion_particles = list(self.dion_particle_by_core.values())

        #======================================================================
        # BRIDSON / SOLVATION SAMPLING PARAMETERS
        #======================================================================

        self.default_min_distance = 2.0

        self.bridson_params = {
            # core sampler 
            "k"               : 30,    # candidate attempts per active point; higher -> denser packing, slower

            # r binary-search (count/concentration mode only)
            "r_floor"         : 1.0,   # Angstrom; hard lower bound on exclusion radius during r-search
            "repeats"         : 2,     # Bridson runs per r evaluation to reduce stochastic variance
            "bracket_steps"   : 10,    # iterations to bracket the feasibility boundary
            "bin_steps"       : 10,    # binary-search refinement steps after bracketing

            # final placement loop
            "final_repeats"   : 10,    # independent runs at the chosen r; best result is kept
            "tol_frac"        : 0.05,  # +/- fractional tolerance on target count before a warning is raised
            
            # ionic charge neutrality 
            "injection_limit" : 1024,  # max ion pairs tried during charge-neutrality correction
        }

        #======================================================================
        # SETUP AND VERIFICATION FUNCTIONALITY
        #======================================================================

        self.seed = None      # the seed actually used; set by apply_seed()
        self.cli_seed = None  # --seed from the command line, if given

        # Output sidecars; IO fills in the paths once it knows the output folder.
        self.structure_record_file = None  # structure plus functional groups
        self.salts_record_file = None      # one row per ion

        self.atom_type_priority = defaultdict(list) # priority levels for atom types
        for entry in self.default_species_chain:
            self.atom_type_priority[0].append(entry["atom_type"])
        # Validation sets the packed bulk priority here; None means check every pair.
        self.bulk_priority = None

        # Current release version; archived inputs don't store this yet.
        self.version = "0.6.1"

        self.force_field = "CHARMM36m"
        self.drude_polarisable = False  # whether to use Drude polarisable model

        self.script_location = os.path.dirname(os.path.abspath(__file__))
        self.script_dir = os.path.dirname(self.script_location)
        self.called_from = os.getcwd()

        # force-fields sits next to the code unless --force-fields points elsewhere
        self.force_fields_root = os.path.abspath(os.path.join(self.script_location, "..", "force-fields"))

        # reinitialise fills these in once it knows the force field
        self.fragments_file = os.path.join(self.force_fields_root, self.force_field.lower(), "chirality_kit_fragments.xdata")
        self.solvents_file  = os.path.join(self.force_fields_root, self.force_field.lower(), "chirality_kit_solvents.xdata")

        self.main_file = None # main file to be processed into .xdata
        self.system_name = None # name of the system, used for naming output files
        self.system_folder = None # folder to save output files in, named after system and timestamp to avoid overwriting
        self.out_name = None # optional CLI output folder; its basename names generated files


    def _default_species_entry_is_drude_particle(self, entry):

        atom_type = str(entry.get("atom_type", ""))
        element = str(entry.get("atom_element", ""))
        return (
            atom_type == "DRUD"
            or atom_type.endswith("-DRUD")
            or element in {"DP", "LP", "X"}
        )

    def _normalise_default_species_chain(self, species_chain, nanotube_type=None):

        normalised_chain = []
        for raw_entry in species_chain:
            entry = dict(raw_entry)
            if "atom_element" not in entry and "element" in entry:
                entry["atom_element"] = entry.pop("element")
            if "atom_mass" not in entry and "mass" in entry:
                entry["atom_mass"] = entry.pop("mass")
            if "atom_charge" not in entry and "charge" in entry:
                entry["atom_charge"] = entry.pop("charge")
            if "atom_structure" not in entry:
                if nanotube_type is None:
                    raise ValueError(
                        "Default species entries must define atom_structure "
                        "unless nanotube_type is provided."
                    )
                entry["atom_structure"] = nanotube_type

            entry["atom_type"] = str(entry["atom_type"])
            entry["atom_element"] = str(entry["atom_element"])
            entry["atom_mass"] = float(entry["atom_mass"])
            entry["atom_charge"] = float(entry["atom_charge"])
            entry["atom_structure"] = str(entry["atom_structure"])
            if "drude_alpha" in entry and entry["drude_alpha"] is not None:
                entry["drude_alpha"] = float(entry["drude_alpha"])
            if "drude_thole" in entry and entry["drude_thole"] is not None:
                entry["drude_thole"] = float(entry["drude_thole"])
            normalised_chain.append(entry)

        if not normalised_chain:
            raise ValueError("Default species chain cannot be empty.")

        return normalised_chain

    def _refresh_default_species_lookups(self):

        if not self.default_species_chain:
            raise ValueError("Default species chain cannot be empty.")

        structures = {entry.get("atom_structure") for entry in self.default_species_chain}
        if None in structures or "" in structures or len(structures) != 1:
            raise ValueError(
                "Default species chain must contain exactly one atom_structure value."
            )

        self.default_species_by_type = {
            entry["atom_type"]: entry for entry in self.default_species_chain
        }
        self.default_species_by_element = defaultdict(list)
        for entry in self.default_species_chain:
            self.default_species_by_element[entry["atom_element"]].append(entry)

        self.default_species_primary = next(
            (
                entry for entry in self.default_species_chain
                if not self._default_species_entry_is_drude_particle(entry)
            ),
            self.default_species_chain[0],
        )
        self.default_species_nanotube_type = next(iter(structures))

    def _set_default_species_chain(self, species_chain, nanotube_type=None):

        self.default_species_chain = self._normalise_default_species_chain(
            species_chain,
            nanotube_type=nanotube_type,
        )
        self._refresh_default_species_lookups()

    def _parse_default_species_mass_line(self, line, atom_structure):

        if "#" not in line:
            raise ValueError(f"Could not parse nanotube species Masses line: {line.strip()}")

        data, comment = line.split("#", 1)
        data_parts = data.split()
        comment_parts = comment.split()
        if len(data_parts) < 2 or len(comment_parts) < 2:
            raise ValueError(f"Could not parse nanotube species Masses line: {line.strip()}")

        try:
            charge_index = comment_parts.index("CHARGE") + 1
            charge = float(comment_parts[charge_index])
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not parse CHARGE from nanotube species Masses line: {line.strip()}"
            ) from exc

        entry = {
            "atom_type": comment_parts[0],
            "atom_element": comment_parts[1],
            "atom_mass": float(data_parts[1]),
            "atom_charge": charge,
            "atom_structure": atom_structure,
        }

        drude_match = re.search(r"DRUDE\s*\(([^)]*)\)", comment)
        if drude_match:
            drude_parts = [part.strip() for part in drude_match.group(1).split(",")]
            if len(drude_parts) >= 2:
                entry["drude_alpha"] = float(drude_parts[0])
                entry["drude_thole"] = float(drude_parts[1])

        return entry

    def get_default_species_for_element(self, element):

        entries = self.default_species_by_element.get(element, [])
        if not entries:
            return None
        return next(
            (
                entry for entry in entries
                if not self._default_species_entry_is_drude_particle(entry)
            ),
            entries[0],
        )

    def get_default_species_type_for_element(self, element):

        entry = self.get_default_species_for_element(element)
        return entry["atom_type"] if entry else None

    def get_default_species_bond_param(self, name, fallback):

        primary = self.default_species_primary or {}
        element = primary.get("atom_element", "C")
        params = self.bond_params.get(element, {})
        return params.get(name, fallback) if isinstance(params, dict) else fallback

    def apply_seed(self, json_input=None):

        '''
        Resolve the RNG seed for this run, seed numpy's global stream with it, and
        record it on the config. Must be called once, after the JSON input has been
        loaded and before anything consumes randomness.

        Resolution order, highest priority first:
            1. self.cli_seed: the --seed command-line flag
            2. json_input['seed']: explicit seed in the input file
            3. json_input['metadata']['rng_seed_used']: the seed an earlier run
               recorded into its archived input, so an output folder replays itself
            4. the SEED module constant
            5. a freshly drawn random seed

        Calling this from the IO entry path rather than only from the CLI is what
        makes programmatic (library) use reproducible; previously such runs never
        seeded at all and recorded seed=None.

        Parameters:
        ----------
        json_input : dict, optional
            The parsed input dictionary. Ignored if not a dict.

        Returns:
        -------
        seed : int
            The seed that was applied.

        Raises:
        ------
        None
        '''

        seed = self.cli_seed

        if seed is None and isinstance(json_input, dict):
            seed = json_input.get('seed')
            if seed is None:
                metadata = json_input.get('metadata')
                if isinstance(metadata, dict):
                    seed = metadata.get('rng_seed_used')

        if seed is None:
            seed = SEED

        if seed is None:
            seed = int(np.random.randint(0, 2**31))

        seed = int(seed)
        np.random.seed(seed)
        self.seed = seed

        print("\n SYS-INFO: RNG seed set to:", seed)
        my_log_file.info(f"RNG seed set to {seed}.")

        return seed

    def reinitialise(self, force_field=None, nanotube=None, functionalisation=None,
                     solvation=None, nanotube_type=None):

        '''
        Update file paths, Drude polarisability flag, and default nanotube species
        to match a new force field. Called after the JSON input is read so that the
        correct fragment and solvent files are located before any structure is built.
        Also re-reads the Nanotube-Species entry from the fragments file to keep
        atom types, elements, masses, and charges in sync with the chosen field.

        Parameters:
        ----------
        force_field : str, optional
            Force field identifier string (e.g. 'CHARMM36m', 'CHARMM36m-NDP').
            If None the currently stored value is used unchanged.

        Returns:
        -------
        None

        Raises:
        ------
        FileNotFoundError
            If the fragment or solvent data files for the chosen force field
            cannot be located in the expected directory.
        '''

        if force_field is not None: 
            self.force_field = force_field
        if self.force_field.lower() in self.drude_particle_FFs:
            self.drude_polarisable = True
            self.section_spacing["Masses"] = [12, 9, 3, 16, 5, 10, 30, 8, 16, 8, 30]
        else:
            self.drude_polarisable = False
            self.section_spacing["Masses"] = [12, 9, 3, 16, 5, 10, 30, 8, 16]
        
        # =================================
        # FILE PATHING AND VERIFICATION
        # =================================

        self.fragments_file = os.path.join(self.force_fields_root, self.force_field.lower(), "chirality_kit_fragments.xdata")
        self.solvents_file  = os.path.join(self.force_fields_root, self.force_field.lower(), "chirality_kit_solvents.xdata")
        if not os.path.exists(self.solvents_file) and solvation:
            raise FileNotFoundError("Could not find force field solvents file. Please ensure force field folder is in the same folder as the python script or specify the path with the --force-fields flag.")
        if not os.path.isfile(self.fragments_file) and nanotube:
            raise FileNotFoundError("Could not find force field fragments file. Please ensure force field folder is in the same folder as the python script or specify the path with the --force-fields flag.")

        # =================================
        # DEFAULT NANOTUBE SPECIES UPDATE
        # =================================

        if nanotube:
            ff_nt_type = None
            species_chain = []
            with open(self.fragments_file, "r") as f:
                in_section = False
                in_masses = False
                first_fragment_seen = False
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("Fragment "):
                        if in_section:
                            break
                        header_parts = stripped.split()
                        is_nanotube_species = (
                            len(header_parts) >= 3
                            and header_parts[0] == "Fragment"
                            and header_parts[1] == "Nanotube-Species"
                        )
                        if not first_fragment_seen:
                            first_fragment_seen = True
                            if not is_nanotube_species:
                                raise ValueError(
                                    "The first Fragment in a nanotube fragments file must be "
                                    "'Fragment Nanotube-Species <Nanotube Type>'."
                                )
                        in_section = is_nanotube_species
                        if in_section:
                            ff_nt_type = header_parts[2]
                        in_masses = False
                        continue
                    if in_section:
                        if stripped == "":
                            continue
                        if stripped == "END":
                            break
                        if stripped.startswith("Masses"):
                            in_masses = True
                            continue
                        if in_masses:
                            if not stripped[0].isdigit():
                                break
                            species_chain.append(
                                self._parse_default_species_mass_line(stripped, ff_nt_type)
                            )

            if ff_nt_type is None:
                raise ValueError(
                    "Could not find required 'Fragment Nanotube-Species <Nanotube Type>' "
                    "in fragments file."
                )

            if nanotube_type is not None and str(nanotube_type) != str(ff_nt_type):
                raise ValueError(
                    f"Force field '{self.force_field}' defines nanotube species '{ff_nt_type}',\n"
                    f"but input requested '{nanotube_type}'.\n"
                    "Choose a matching force field or change nanotube.type."
                )

            if not species_chain:
                raise ValueError(
                    f"Could not find Masses entries in Fragment Nanotube-Species {ff_nt_type}."
                )

            self._set_default_species_chain(species_chain, nanotube_type=ff_nt_type)
            if any(self._default_species_entry_is_drude_particle(entry)
                   for entry in self.default_species_chain):
                self.drude_polarisable = True
                self.section_spacing["Masses"] = [12, 9, 3, 16, 5, 10, 30, 8, 16, 8, 30]

            self.atom_type_priority = defaultdict(list)
            for entry in self.default_species_chain:
                atom_type = entry["atom_type"]
                if atom_type not in self.atom_type_priority[0]:
                    self.atom_type_priority[0].append(atom_type)
        
    
    def get_supported_info(self):

        '''
        Return a summary dictionary of the currently supported force fields
        and accepted file input formats.

        Parameters:
        ----------
        None

        Returns:
        -------
        info : dict
            A dictionary with supported configuration metadata.

        Raises:
        ------
        None
        '''

        info = {
            "supported_file_inputs": self.supported_file_inputs,
        }
        return info
