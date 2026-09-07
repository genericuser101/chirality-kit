from __future__ import annotations 
import argparse
import logging
import os
import re
import shutil
import string
import json
import time
import datetime 
from collections import defaultdict
from typing import TYPE_CHECKING
from itertools import permutations, combinations

import numpy as np # yeah thats it

SEED = None

DUMMY_FG_TYPE = "DUMMY"

TERM_SIDES = {
    "both": (True, True),
    "-": (True, False),
    "+": (False, True),
}

TERM_GROUP_FIELDS = {"type", "count", "side"}

XDATA_SECTION_SETTINGS = {
    "bonds": "Bonds",
    "angles": "Angles",
    "dihedrals": "Dihedrals",
    "impropers": "Impropers",
    "lone-pairs": "Lone Pairs",
    "anisotropy": "Anisotropy",
}

# system parameters here or with .json
INPUT = {
    # Handy default seed. --seed wins, then JSON/archive, then this.  None picks a fresh one.
    "seed": None,

    "metadata": {
        "made-by": "chirality-kit",
        "version": "1.0",
        "creator": "Stef @ Warwick University",
        "filename": "chirality_main.py"
    },

    "settings": {
        "system_name": "auto",
        "verbose": False,
        "output_types": [".pdb", ".psf", ".xyz", ".data"],
        "adjust-ion-partial-charge": False,
        "bonds": True,
        "angles": True,
        "dihedrals": True,
        "impropers": True,
        "lone-pairs": True,
        "anisotropy": True,
    },

    "system": {
        "force-field": "CHARMM36m",
        "size": "auto",
        "size-padding-xy": 14.0,
        "size-padding-z": 30.0,
        "nanotube": True,
        "functionalisation": True,
        "solvation": False,
        "pbc-override": False
    },

    "nanotube": {
        "type": "CNT",
        "n": 8,
        "m": 4,
        "repeats": 7,
        "periodic": False
    },

    "functionalisation": {
        "term": {
            "term-hydrogenate": True,
            "term-start-highest-x": True,
            "groups": [
                {
                    "type": "COO-",
                    "count": 2,
                },
                {
                    "type": "OH-",
                    "count": 4,
                    "side": "-"   # "-", "+", or "both"
                }
            ]
        },
        "rings": [
            {
                "ring-phase-increment": 0.78539816339,
                "ring-phase-start-offset": 0.0,
                "ring-padding": True,
                "ring-count": 7,
                "ring-placement": "external",
                "groups": [
                    {
                        "type": "C2H5",
                        "count": 1,
                    },
                    {
                        "type": "OH-",
                        "count": 1,
                    },
                    {
                        "type": "OH-",
                        "count": 5,
                    }
                ]
            },
            {
                "ring-phase-increment": 0.78539816339,
                "ring-phase-start-offset": 0.0,
                "ring-padding": True,
                "ring-count": 3,
                "ring-placement": "internal",
                "groups": [
                    {
                        "type": "H+",
                        "count": 3,
                    },
                    {
                        "type": "OH-",
                        "count": 1,
                    },
                    {
                        "type": "OH-",
                        "count": 1,
                    }
                ]
            },
        ],
    },

    "solvation": {
        "groups": [
            {
                "type": "SPCE",
                "placement": "packed",
                "priority": 100,
                "min_distance": 1.8,
                "concentration": 55.5, # molarity in mol/L
                "molecules_per_unit": 1
            },
            {
                "type": "KCl",
                "placement": "ionic",
                "charge-neutrality": True,
                "priority": 50,
                "count": 4,
                "min_distance": 2.5
            }
        ],
        "bridson_params": {
            "k"              : 30,    # candidate attempts per active point; higher -> denser packing, slower
            "r_floor"        : 1.0,   # Angstrom, hard lower bound on exclusion radius during r-search
            "repeats"        : 2,     # Bridson runs per r evaluation to reduce stochastic variance
            "bracket_steps"  : 10,    # iterations to bracket the feasibility boundary
            "bin_steps"      : 10,    # binary-search refinement steps after bracketing
            "final_repeats"  : 10,    # independent runs at the chosen r; best result is kept
            "tol_frac"       : 0.05,  # fractional tolerance on target count before a warning is raised
            "injection_limit": 1024   # max ion pairs tried during charge-neutrality correction
        }

    }
}


#========================================================================================
# CHIRALITY KIT CORE CLASSES AND FUNCTIONS ACTUALLY START HERE
#========================================================================================


class Config:

    def __init__(self):

        '''
        Initialise the Config object with default force field settings, file paths,
        and supported parameter lists. The force-fields directory defaults to
        ../force-fields next to this script's folder and can be overridden by assigning
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


class Geometry:

    def __init__(self,
                 config: Config) -> None:
        
        self.config : Config = config


    def gcd(self, a, b):

        '''
        Compute the greatest common divisor of two integers using Euclid's algorithm.
        Used internally to determine the translational period of a CNT unit cell.

        Parameters:
        ----------
        a : int
            First integer.
        b : int
            Second integer.

        Returns:
        -------
        a : int
            Greatest common divisor of the two inputs.

        Raises:
        ------
        None
        '''

        while b:
            a, b = b, a % b
        return a

    def random_unit_quaternion(self):

        '''
        Sample a uniformly distributed unit quaternion on S^3 using the
        Shoemake / Marsaglia method, which correctly draws from the Haar measure
        and avoids polar-cap bias.

        Parameters:
        ----------
        None

        Returns:
        -------
        q : (4,) np.ndarray of float64
            Unit quaternion [x, y, z, w] where w is the scalar part.

        Raises:
        ------
        None
        '''

        u1, u2, u3 = np.random.random(3)
        s1 = np.sqrt(1.0 - u1)
        s2 = np.sqrt(u1)
        theta1 = 2.0 * np.pi * u2
        theta2 = 2.0 * np.pi * u3
        x = s1 * np.sin(theta1)
        y = s1 * np.cos(theta1)
        z = s2 * np.sin(theta2)
        w = s2 * np.cos(theta2)

        return np.array([x, y, z, w], dtype=np.float64)

    def normalise_quat(self, q):

        '''
        Normalise a quaternion to unit length. Raises ValueError if the input
        quaternion has zero norm, as normalisation would be undefined.

        Parameters:
        ----------
        q : array-like of shape (4,)
            Quaternion [x, y, z, w] to normalise.

        Returns:
        -------
        q_norm : (4,) np.ndarray of float64
            Unit-length quaternion with the same orientation as the input.

        Raises:
        ------
        ValueError
            If the quaternion norm is zero.
        '''

        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n == 0.0:
            raise ValueError("Quaternion has zero norm.")
        return q / n

    def quat_to_rotmat(self, q):

        '''
        Convert a quaternion to a 3x3 rotation matrix using the standard
        quaternion-to-matrix formula. The quaternion is normalised internally
        before conversion.

        Parameters:
        ----------
        q : array-like of shape (4,)
            Quaternion [x, y, z, w]; need not be pre-normalised.

        Returns:
        -------
        M : (3,3) np.ndarray of float64
            Orthogonal rotation matrix corresponding to the quaternion.

        Raises:
        ------
        ValueError
            If the quaternion has zero norm (propagated from normalise_quat).
        '''

        x, y, z, w = self.normalise_quat(q)
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        return np.array([         # yeah i dont know what to comment here, its just q formula
            [1.0 - 2.0*(yy + zz),     2.0*(xy - wz),         2.0*(xz + wy)],
            [    2.0*(xy + wz),   1.0 - 2.0*(xx + zz),       2.0*(yz - wx)],
            [    2.0*(xz - wy),       2.0*(yz + wx),     1.0 - 2.0*(xx + yy)]
        ], dtype=np.float64)

    def rotate_points_quat(self, pos_matrix, quat, center):

        '''
        Rotate a set of 3-D points about a given centre using a quaternion.
        Accepts either an (N,3) array or a flat length-3N array; returns the
        same shape as the input.

        Parameters:
        ----------
        pos_matrix : (N,3) np.ndarray or flat array of length 3N
            Cartesian coordinates of the points to rotate.
        quat : array-like of shape (4,)
            Rotation quaternion [x, y, z, w].
        center : array-like of shape (3,)
            The point about which to rotate.

        Returns:
        -------
        Prot : (N,3) np.ndarray or flat array
            Rotated coordinates in the same shape as pos_matrix.

        Raises:
        ------
        ValueError
            If pos_matrix is a flat array whose length is not a multiple of 3,
            or if it is not a valid (N,3) array.
        '''

        P = np.asarray(pos_matrix, dtype=np.float64) # some typesetting
        flat = False
        if P.ndim == 1:
            if P.size % 3 != 0:
                raise ValueError("Flat points array length must be a multiple of 3.")
            P = P.reshape(-1, 3)
            flat = True
        elif P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points must be (N,3) or flat length-3N.")

        c = np.asarray(center, dtype=np.float64).reshape(1, 3)
        R = self.quat_to_rotmat(quat)                    # 3x3
        Prot = (P - c) @ R.T + c                        # rotate about center

        if flat:
            return Prot.reshape(-1)
        return Prot

    def rotate_a_to_b_matrix(self, a, b, eps=1e-12):

        '''
        Construct the 3x3 rotation matrix R such that R @ a_unit = b_unit,
        i.e. the smallest rotation that maps unit vector a onto unit vector b.
        Handles the collinear edge cases (same and opposite directions) exactly.

        Parameters:
        ----------
        a : array-like of shape (3,)
            Source unit vector (does not need to be pre-normalised).
        b : array-like of shape (3,)
            Target unit vector (does not need to be pre-normalised).
        eps : float, optional
            Threshold below which the cross-product squared norm is treated as
            collinear. Defaults to 1e-12.

        Returns:
        -------
        R : (3,3) np.ndarray of float64
            Rotation matrix satisfying R @ (a/|a|) ~= (b/|b|).

        Raises:
        ------
        None
        '''

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s2 = np.dot(v, v)

        if s2 > eps:  # general case
            V = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])
            return np.eye(3) + V + V @ V * ((1 - c) / s2)

        # collinear cases
        if c > 0:      # same direction
            return np.eye(3)
        else:          # opposite direction
            t = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(a, t)
            u /= np.linalg.norm(u)
            return 2.0 * np.outer(u, u) - np.eye(3) 

    def rotate_points_simple(self, pos_matrix, ref_vector):

        '''
        Rotate all atoms in pos_matrix so that the centroid of the group
        is aligned to ref_vector. Uses rotate_a_to_b_matrix internally.
        Falls back to returning the input unchanged for degenerate cases
        (centroid at origin, or single-atom groups) with a logged warning.

        Parameters:
        ----------
        pos_matrix : (N,3) np.ndarray
            Cartesian coordinates of the atoms to rotate.
        ref_vector : array-like of shape (3,)
            The direction the centroid should point toward after rotation.

        Returns:
        -------
        rotated_matrix : (N,3) np.ndarray
            Rotated atom coordinates. Same shape as pos_matrix.

        Raises:
        ------
        None
        '''

        centroid = np.mean(pos_matrix, axis=0)
        
        centroid_norm = np.linalg.norm(centroid) # normalise centroid vector
        if centroid_norm < 1e-12 and pos_matrix.shape[0] < 2: # no point doing it for single atom fgs 
            if centroid.shape[0] == 1:
                my_log_file.warning("Warning: Centroid is at origin, cannot determine rotation direction")
            return pos_matrix
        
        centroid_unit = centroid / centroid_norm
        
        ref_vector = np.asarray(ref_vector, dtype=float)
        ref_norm = np.linalg.norm(ref_vector)
        if ref_norm < 1e-12:
            my_log_file.warning("Warning: Reference vector is zero")
            return pos_matrix
        
        ref_unit = ref_vector / ref_norm
        
        R = self.rotate_a_to_b_matrix(centroid_unit, ref_unit) # get the rotation matrix
        
        rotated_matrix = (R @ pos_matrix.T).T  # rotate all points, shape (N, 3), transpose back
        
        rotated_centroid = np.mean(rotated_matrix, axis=0) # could verify centroid is aligned now
        
        return rotated_matrix


    def _get_circle_points_angle_greedy(self, x_atom_positions, y_atom_positions, n,
                                        start_index=None):

        '''
        Select n atoms from a ring of candidates such that their angular positions
        around the ring are as evenly spaced as possible. Starts at a requested
        atom, or a random atom when none is supplied, and greedily picks the next
        candidate closest to the ideal angular step, ensuring no atom is chosen
        twice.

        Parameters:
        ----------
        x_atom_positions : array-like of float
            X coordinates of the candidate atoms in the ring plane.
        y_atom_positions : array-like of float
            Y coordinates of the candidate atoms in the ring plane.
        n : int
            Number of atoms to select.
        start_index : int, optional
            Array index of the first atom. Random when omitted.

        Returns:
        -------
        chosen : (n,) np.ndarray of int
            Array indices into the input arrays of the selected atoms.

        Raises:
        ------
        None
        '''

        x_atom_positions = np.asarray(x_atom_positions)
        y_atom_positions = np.asarray(y_atom_positions)
        M = len(x_atom_positions)
        cx, cy = np.mean(x_atom_positions), np.mean(y_atom_positions) # compute radius and centre
        angles = np.degrees(np.arctan2(y_atom_positions - cy, x_atom_positions - cx)) % 360

        order = np.argsort(-angles) # clockwise wrap-around
        sorted_angles = angles[order]
        angles_ext = np.concatenate([sorted_angles, sorted_angles - 360])
        idx_ext = np.concatenate([order, order])

        if start_index is None:
            # Use the shared RNG; default_rng() would ignore the run seed.
            start_pos = np.random.randint(0, M)
        else:
            start_pos = int(np.flatnonzero(order == start_index)[0])
        chosen = [order[start_pos]]
        used = set(chosen) # so that a point cannot be chosen twice 
        prev_angle = angles_ext[start_pos]
        current_pos = start_pos

        angular_step = 360.0 / n

        for _ in range(1, n):
            target_angle = prev_angle - angular_step
            search_slice = angles_ext[current_pos + 1 : current_pos + 1 + M]
            candidates = idx_ext[current_pos + 1 : current_pos + 1 + M]

            diffs = np.abs(search_slice - target_angle) # compute angle, mask already used
            for j, c in enumerate(candidates):
                if c in used:
                    diffs[j] = np.inf

            sel_rel = np.argmin(diffs)
            sel_pos = current_pos + 1 + sel_rel
            chosen_idx = idx_ext[sel_pos]

            chosen.append(chosen_idx)
            used.add(chosen_idx)

            prev_angle = angles_ext[sel_pos]
            current_pos = sel_pos

        return np.array(chosen, dtype=int)


    def _find_discrete_cylinder_points(self, 
                                        structure_atoms, indexes, occupied,
                                        rings_count, 
                                        per_ring_count,
                                        per_ring_phase_step,
                                        ring_phase_start_offset: float = 0.0,
                                        in_or_ex: str = "internal",
                                        padding: bool = True):
            
        # container setup
        all_indexes = np.empty((0, 1))
        all_vectors = np.empty((0, 3))
        all_positions = np.empty((0, 3))

        # approximate the cylinder 
        diameter = (np.max(structure_atoms[:,0]) - np.min(structure_atoms[:,0]))
        length = np.max(structure_atoms[:,2]) - np.min(structure_atoms[:,2])

        if padding:
            lengths = np.linspace(np.min(structure_atoms[:,2]), np.max(structure_atoms[:,2]), rings_count + 2)[1:-1]
        else: 
            lengths = np.linspace(np.min(structure_atoms[:,2]), np.max(structure_atoms[:,2]), rings_count)
        
        angles = np.linspace(ring_phase_start_offset, 2*np.pi + ring_phase_start_offset, per_ring_count, endpoint=False)

        xyz_positions = []
        for i in range(len(lengths)):
            for j in range(len(angles)):
                z = lengths[i]
                angle = angles[j]
                angle += i * per_ring_phase_step
                xyz_positions.append(np.array([np.cos(angle)*diameter/2, np.sin(angle)*diameter/2, z]))
        xyz_positions = np.array(xyz_positions)
        
        # snap to nearest carbon
        for xyz_position in xyz_positions:
            dists = np.linalg.norm(structure_atoms - xyz_position, axis=1)

            for idx in occupied:
                program_idx = np.where(indexes == idx)[0]
                dists[program_idx] = np.inf  # mask already occupied atoms

            canditate_found = False
            while not canditate_found:

                if all(dists == np.inf):  # all used up
                    my_log_file.error("Not enough unique atoms to functionalise. Reduce the number of functional groups or set padding=False.")
                    raise ValueError("Not enough unique atoms to functionalise. Reduce the number of functional groups or set padding=False.")
                
                best_index_local = np.where(dists == np.min(dists))[0][0]

                atom_id = int(indexes[best_index_local])

                if atom_id not in occupied:
                    all_indexes = np.vstack([all_indexes, atom_id])

                    if in_or_ex == "internal":
                        carbon_vector = np.array([
                            -structure_atoms[best_index_local, 0],
                            -structure_atoms[best_index_local, 1],
                            0
                            ])
                    else:
                        carbon_vector = np.array([
                            structure_atoms[best_index_local, 0],
                            structure_atoms[best_index_local, 1],
                            0
                            ])
                        
                    carbon_vector = carbon_vector / np.linalg.norm(carbon_vector)
                    all_vectors = np.vstack([all_vectors, carbon_vector])
                    all_positions = np.vstack([all_positions, structure_atoms[best_index_local, :]])
                    canditate_found = True

                else:
                    dists[best_index_local] = np.inf  # mark as used
        
        for index in all_indexes:
            occupied.add(int(index[0]))  # atom IDs only

        return all_indexes, all_vectors, all_positions, occupied


class File_Manager:

    def __init__(self,
                 config: Config) -> None:
        
        self.config : Config = config


    def insert_line(self, lines, position, newline):

        '''
        Insert a single line string into a list of lines at the specified index,
        shifting all subsequent lines down by one position.

        Parameters:
        ----------
        lines : list of str
            The list of file lines to modify in-place.
        position : int
            Zero-based index at which to insert the new line.
        newline : str
            The line string to insert (should include a trailing newline if required).

        Returns:
        -------
        lines : list of str
            The modified list with the new line inserted.

        Raises:
        ------
        None
        '''

        lines.insert(position, newline)
        return lines

    def write_file(self, lines, file_path):

        '''
        Write a list of line strings to a file, overwriting any existing content.

        Parameters:
        ----------
        lines : list of str
            Lines to write. Each string is written as-is; callers are responsible
            for including newline characters where needed.
        file_path : str
            Absolute or relative path to the output file.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        with open(file_path, 'w') as f:
            f.writelines(lines)

    def read_file(self, filename_or_lines):

        '''
        Read a file and return its lines, or pass a list through unchanged.
        This dual behaviour lets callers pipe an already-loaded list of lines
        into any function that expects a filename, avoiding redundant disk reads.

        Parameters:
        ----------
        filename_or_lines : str or list of str
            If a string, treated as a file path and the file is opened and read.
            If a list, returned directly without modification.

        Returns:
        -------
        lines : list of str
            Lines read from the file, or the original list if one was passed in.

        Raises:
        ------
        None
        '''

        try:
            with open(filename_or_lines, 'r') as f:
                return f.readlines()
        except (TypeError, FileNotFoundError):
            return filename_or_lines  # this is done just so I can pass lists of lines directly

    def line_to_array(self, line):

        '''
        Convert a whitespace-delimited line string into a list of tokens.
        If the input is already a list it is returned unchanged, making this
        safe to call on data that has already been split.

        Parameters:
        ----------
        line : str or list of str
            A raw line from an xdata file, or an already-split token list.

        Returns:
        -------
        tokens : list of str
            Whitespace-split tokens from the line, or the original list.

        Raises:
        ------
        None
        '''

        if isinstance(line, list):
            return line
        else:
            return line.strip().split()


    def find_line(self, filename, section_name):

        '''
        Scan a file line-by-line and return the index of the first line that
        contains the given section name as a substring.

        Parameters:
        ----------
        filename : str
            Path to the file to search.
        section_name : str
            The section header string to locate (e.g. 'Atoms', 'Bonds').

        Returns:
        -------
        line_number : int or None
            Zero-based index of the first matching line, or None if not found.

        Raises:
        ------
        None
        '''

        lines = self.read_file(filename)
        
        for i, line in enumerate(lines):
            if section_name in line:
                return i
        return None 

    def find_last_section_idx(self, filename_or_lines, section_name):

        '''
        Find the integer index value of the last entry in a named section of an
        xdata file, as read from the first column of the last non-blank line
        before the following section header. Returns 0 if the section header is
        the only line present (i.e. the section is empty).

        Parameters:
        ----------
        filename_or_lines : str or list of str
            File path or pre-loaded list of lines.
        section_name : str
            Name of the section to query (must be in config.section_order).

        Returns:
        -------
        idx : int or None
            The LAMMPS index of the last entry in the section, 0 if the
            section is empty, or None if no content could be found.

        Raises:
        ------
        ValueError
            If section_name is not present in config.section_order.
        '''

        section_sequence = self.config.section_order

        terminating_idx = next((i for i, s in enumerate(section_sequence) if s == section_name), -1)
        if terminating_idx == -1:
            raise ValueError("Section name not found in the provided sequence.")
        
        lines = self.read_file(filename_or_lines)
        section_start = next(
            (i for i, line in enumerate(lines) if line.strip() == section_name),
            None,
        )
        if section_start is None:
            return None

        index = len(lines)
        later_sections = set(section_sequence[terminating_idx + 1:])
        for i, line in enumerate(lines[section_start + 1:], section_start + 1):
            if line.strip() in later_sections:
                index = i
                break

        for i in range(index - 1, section_start - 1, -1):
            line = lines[i].strip()
            if line == section_name:
                return 0  
            if line:  
                idx_line = lines[i].rstrip('\n').split()
                return int(idx_line[0])
        return None

    def find_injection_point(self, filename_or_lines, section_name):

        '''
        Determine the line index at which new entries should be inserted into
        a named section. Walks backward from the next section header to find
        the line immediately after the last existing entry, or the line after
        the section header itself if the section is currently empty.

        Parameters:
        ----------
        filename_or_lines : str or list of str
            File path or pre-loaded list of lines.
        section_name : str
            Name of the target section (must be in config.section_order).

        Returns:
        -------
        idx : int or None
            Line index to pass to insert_line for appending a new entry,
            or None if the injection point cannot be determined.

        Raises:
        ------
        ValueError
            If section_name is not present in config.section_order.
        '''

        section_sequence = self.config.section_order

        terminating_idx = next((i for i, s in enumerate(section_sequence) if s == section_name), -1)
        if terminating_idx == -1:
            raise ValueError("Section name not found in the provided sequence.")
        
        lines = self.read_file(filename_or_lines)
        section_start = next(
            (i for i, line in enumerate(lines) if line.strip() == section_name),
            None,
        )
        if section_start is None:
            return None

        index = len(lines)
        later_sections = set(section_sequence[terminating_idx + 1:])
        for i, line in enumerate(lines[section_start + 1:], section_start + 1):
            if line.strip() in later_sections:
                index = i
                break

        for i in range(index - 1, section_start - 1, -1):
            line = lines[i].strip()
            if line in self.config.section_order:
                return i + 2
            if line != "":
                return i+1
        return None


    def find_and_replace_lines(self, filename, start_marker, end_marker, new_lines):

        '''
        Locate an exact block of content between start_marker and end_marker
        in a file and replace it with new_lines, writing the result back to disk.
        Matching is done token-by-token after stripping newlines, so minor
        whitespace differences are tolerated.

        Parameters:
        ----------
        filename : str
            Path to the file to modify in-place.
        start_marker : str
            The exact line content that marks the start of the block to replace.
        end_marker : str
            The exact line content that marks the end of the block to replace.
        new_lines : list of str
            Replacement content. Lines without trailing newlines have one appended.

        Returns:
        -------
        success : bool
            True if both markers were found and the replacement was written.
            False if either marker could not be matched.

        Raises:
        ------
        None
        '''

        lines = self.read_file(filename)

        start_idx = None
        end_idx = None
        start_marker_ref = start_marker.strip("\n").split()
        end_marker_ref = end_marker.strip("\n").split()

        for i, line in enumerate(lines):
            line_stripped = line.strip("\n").split()
            if line_stripped == start_marker_ref:
                start_idx = i-1
            if line_stripped == end_marker_ref and start_idx is not None:
                end_idx = i+1
                break
        
        if start_idx is None or end_idx is None:
            print(f"Could not find exact matches for '{start_marker_ref}' and '{end_marker_ref}'")
            return False
        
        new_content = lines[:start_idx + 1]        # build new content
        new_content.extend([line + '\n' if not line.endswith('\n') else line 
                        for line in new_lines])
        new_content.extend(lines[end_idx:])
        
        with open(filename, 'w') as file:
            file.writelines(new_content)
        
        return True

    def find_and_extract_lines(self, filename_or_lines, section_name, header=True):

        '''
        Extract all non-blank, non-COMMENT lines belonging to a named section,
        stopping as soon as the next recognised section header is encountered.
        Logs an error if the section is not found at all.

        Parameters:
        ----------
        filename_or_lines : str or list of str
            File path or pre-loaded list of lines to search.
        section_name : str
            The name of the section to extract (e.g. 'Atoms', 'Bond Coeffs').
        header : bool, optional
            If True (default), the section header line itself is included as the
            first element of the returned list.

        Returns:
        -------
        section_lines : list of str
            Lines belonging to the section, optionally including the header.

        Raises:
        ------
        None
        '''

        lines = self.read_file(filename_or_lines)

        section_lines = []
        in_section = False

        for line in lines:
            if section_name in line:
                in_section = True
                if header:
                    section_lines.append(line)
                continue
            if in_section:
                if line.strip() in self.config.section_order:
                    break
                if line.strip() == "" or line.strip().startswith("COMMENT"):
                    continue
                else:
                    section_lines.append(line)

        if section_lines == [] and not in_section:
            my_log_file.error(f"Section '{section_name}' not found in the file.")
        return section_lines

    def find_and_extract_subsection(self, filename, structure_type="Fragment", structure="COO-"):

        '''
        Extract all lines belonging to a named sub-block (Fragment or Solvent entry)
        within an xdata file. Starts reading after the line that matches both
        structure_type and structure, and stops at the next occurrence of
        structure_type (indicating the start of the next sub-block).

        Parameters:
        ----------
        filename : str
            Path to the xdata file (fragments or solvents).
        structure_type : str, optional
            The block-type keyword used as a delimiter (e.g. 'Fragment', 'Solvent').
        structure : str, optional
            The specific entry name to extract (e.g. 'COO-', 'TIP3').

        Returns:
        -------
        section_lines : list of str
            Lines of the matching sub-block, excluding blank lines and COMMENT lines.

        Raises:
        ------
        None
        '''

        lines = self.read_file(filename)

        section_lines = []
        in_section = False
        target_header = f"{structure_type} {structure}"
        
        for line in lines:            
            stripped = line.strip()
            if stripped == target_header or stripped.startswith(f"{target_header} "):
                in_section = True
                continue
            if in_section:
                if stripped.startswith(f"{structure_type} "):
                    break
                elif stripped == "" or stripped.startswith("COMMENT"):
                    continue
                else:
                    section_lines.append(line)

        if section_lines == []:
            my_log_file.error(f"Structure '{structure}' not found in the file {filename}.")
        return section_lines

    def find_number_of_entries_in_section(self, section):

        '''
        Count the number of data entries in a named section of the main file
        by extracting the section lines and counting them. Note this is the
        actual line count and may differ from the last index if entries are
        non-contiguous.

        Parameters:
        ----------
        section : str
            Section name to count entries in (e.g. 'Atoms', 'Bonds').

        Returns:
        -------
        count : int
            Number of non-blank, non-header lines in the section.

        Raises:
        ------
        None
        '''

        section_lines = self.find_and_extract_lines(self.config.main_file, section, header=False)

        return len(section_lines)

    def find_last_residue(self):

        '''
        Scan the Atoms section of the main file and return the highest residue
        index encountered. Used to determine the next available residue index
        when inserting new molecules.

        Parameters:
        ----------
        None

        Returns:
        -------
        res_last : int
            The largest residue index found in the Atoms section.

        Raises:
        ------
        None
        '''

        lines = self.read_file(self.config.main_file)

        in_atoms_section = False
        
        res_last = 0
        for line in lines:
            if "Atoms" in line:
                in_atoms_section = True
                continue
            if in_atoms_section:
                if line.strip("\n") == "":  # Skip empty lines
                    continue
            
                if "Bond" in line:
                    break  # end of section

                parts = line.strip('\n').split()
                res = int(parts[1])  # residue index
                if res > res_last:
                    res_last = res

        return res_last

    def delete_line(self, filename, line_to_delete):

        '''
        Remove a specific line from a file by exact token-wise comparison,
        then write the remaining lines back to disk.

        Parameters:
        ----------
        filename : str
            Path to the file to modify.
        line_to_delete : str
            The line to remove. Matched by splitting on whitespace, so
            differences in spacing between tokens are ignored.

        Returns:
        -------
        success : bool
            Always True once the filtered file has been written.

        Raises:
        ------
        None
        '''

        lines = self.read_file(filename)
        line_to_delete_array = self.line_to_array(line_to_delete)

        new_lines = []
        for line in lines:
            if self.line_to_array(line) != line_to_delete_array:
                new_lines.append(line)

        self.write_file(new_lines, filename)
        return True

    def standardise_top_counts(self):

        '''
        Recompute and rewrite the header block of the main xdata file, updating
        atom, bond, angle, dihedral, and improper counts as well as the simulation
        box dimensions. Called after any structural modification to keep the file
        self-consistent.

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

        values = {
            "atoms": 0,
            "bonds": 0,
            "angles": 0,
            "dihedrals": 0,
            "impropers": 0,
            "crossterms": 0, 
            "atom types": 0,
            "bond types": 0,
            "angle types": 0,
            "dihedral types": 0,
            "improper types": 0,
            "xlo xhi": [0, 0],
            "ylo yhi": [0, 0],
            "zlo zhi": [0, 0]
        }

        values['atoms'] = self.find_number_of_entries_in_section("Atoms")
        enabled = self.config.enabled_xdata_sections
        for count_name, section in [
            ("bonds", "Bonds"),
            ("angles", "Angles"),
            ("dihedrals", "Dihedrals"),
            ("impropers", "Impropers"),
        ]:
            if section in enabled:
                values[count_name] = self.find_number_of_entries_in_section(section)

        values['atom types'] = self.find_number_of_entries_in_section("Masses")
        if "Bonds" in enabled or "Anisotropy" in enabled:
            values['bond types'] = self.find_number_of_entries_in_section("Bond Coeffs")
        for count_name, section, coefficient_section in [
            ("angle types", "Angles", "Angle Coeffs"),
            ("dihedral types", "Dihedrals", "Dihedral Coeffs"),
            ("improper types", "Impropers", "Improper Coeffs"),
        ]:
            if section in enabled:
                values[count_name] = self.find_number_of_entries_in_section(coefficient_section)

        box = self.update_box_size()

        values['xlo xhi'] = box[0]
        values['ylo yhi'] = box[1]
        values['zlo zhi'] = box[2]


        new_lines = ['Extended Datafile - chirality_kit']
        for key, value in values.items():

            if key == "xlo xhi" or key == "ylo yhi" or key == "zlo zhi":
                new_line = (
                    f"{str(value[0]):>16}"
                    f" "
                    f"{str(value[1]):>16}"
                    f" "
                    f"{str(key):<20} \n"
                )
                new_lines.append(new_line)
            else:
                if key == "atoms":
                    new_lines.append(f"\n")
                new_line = (
                    f"{str(value):>16}"
                    f" "
                    f"{str(key):<30} \n"
                )
                new_lines.append(new_line)
                if key == "crossterms" or key == "improper types": 
                    new_lines.append(f"\n")

        new_lines.append("")
        new_lines.append("Masses\n")
        self.find_and_replace_lines(self.config.main_file, "Extended Datafile - chirality_kit", "Masses", new_lines)

    def update_box_size(self):

        '''
        Scan the Atoms section of the main file and derive the simulation box
        extents by finding the minimum and maximum x, y, and z coordinates.

        Parameters:
        ----------
        None

        Returns:
        -------
        box : list of list of float
            Three pairs [[xlo, xhi], [ylo, yhi], [zlo, zhi]] representing the
            bounding box of all atoms currently in the file.

        Raises:
        ------
        None
        '''

        xlo_xhi = [0, 0]
        ylo_yhi = [0, 0]
        zlo_zhi = [0, 0]

        lines = self.read_file(self.config.main_file)

        in_atom_section = False

        for line in lines:
            if "Atoms" in line:
                in_atom_section = True
                continue
            if in_atom_section:
                if line.startswith("Bond") or "Bonds" in line:
                    break  # end of section
                if "#" not in line:
                    continue
                parts = line.split()

                x = float(parts[4])
                y = float(parts[5])
                z = float(parts[6])
                if x > xlo_xhi[1]:
                    xlo_xhi[1] = x
                elif x < xlo_xhi[0]:
                    xlo_xhi[0] = x
                if y > ylo_yhi[1]:
                    ylo_yhi[1] = y
                elif y < ylo_yhi[0]:
                    ylo_yhi[0] = y
                if z > zlo_zhi[1]:
                    zlo_zhi[1] = z
                elif z < zlo_zhi[0]:
                    zlo_zhi[0] = z

        return [xlo_xhi, ylo_yhi, zlo_zhi]

    def standardise_line(self, array, section):

        '''
        Format a list of field values into a fixed-width line string using the
        column widths defined in config.section_spacing for the given section.
        Each value is right-aligned within its allocated column width, and the
        line is terminated with a newline character.

        Parameters:
        ----------
        array : list of str
            Field values for the line, one element per column.
        section : str
            Section name used to look up the column-width specification
            (e.g. 'Atoms', 'Bonds', 'Masses').

        Returns:
        -------
        line : str
            Right-aligned, newline-terminated string ready to write to the file.

        Raises:
        ------
        ValueError
            If the length of array does not match the number of columns
            defined in config.section_spacing for the given section.
        '''

        line = ""
        spacing = self.config.section_spacing[section]

        if len(array) != len(spacing):
            my_log_file.error(
                f"Array length {len(array)} does not match section spacing length {len(spacing)} for section '{section}'.")
            my_log_file.error(f"Array: {array}")
            raise ValueError("Array length does not match section spacing.")

        for i, value in enumerate(array):
            value_str = str(value)
            value_str = value_str[:spacing[i]-2]

            # right-align within allocated space
            line += f"{value_str:>{spacing[i]}}"

        line += "\n"
        return line

    def standardise_file(self):

        '''
        Re-format every data line in the main xdata file using the fixed-width
        column layout defined in config.section_spacing. Iterates through all
        sections in config.section_order, leaving the header block and blank
        lines intact, and writes the result back to disk.

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

        lines = self.read_file(self.config.main_file)

        standardised_lines = []
        section = None  # guard against lines before the first section header
        for line in lines:
            if line.strip() in self.config.section_order:
                section = line.strip()
                standardised_lines.append(line)
                continue

            if section and section != "Extended Datafile - chirality_kit":
                if line.strip() == "":
                    standardised_lines.append("\n")
                else:
                    line_array = line.strip("\n").split()
                    new_line = self.standardise_line(line_array, section)
                    standardised_lines.append(new_line)

        self.write_file(standardised_lines, self.config.main_file)


class Force_Field_Manager:

    def __init__(self, 
                 file_manager: File_Manager, 
                 config: Config,
                 geometry: Geometry) -> None:
        
        self.file_manager : File_Manager = file_manager
        self.config : Config = config
        self.gt : Geometry = geometry
        self.generated_bonds_data = []


    def pdb_check_atom_in_residue(self, old_atom_idx, residue_name):

        '''
        Check whether a given atom index belongs to a specific residue name in
        the currently loaded PDB file, by scanning all ATOM records.

        Parameters:
        ----------
        old_atom_idx : int
            The atom index (column 2 of the ATOM record) to look up.
        residue_name : str
            The residue name (column 4 of the ATOM record) to match against.

        Returns:
        -------
        found : bool
            True if an ATOM line with the matching index and residue name exists.

        Raises:
        ------
        None
        '''

        lines = self.file_manager.read_file(self.config.pdb_file)

        for line in lines:
            if "ATOM" in line and residue_name in line:
                parts = line.strip().split()
                idx = parts[1]       # atom idx
                res_name = parts[3]  # residue name
                if str(idx) == str(old_atom_idx) and res_name == residue_name:
                    return True
                    
        return False

    def _normalise_allowed_bond_cutoffs(self, allowed_bonds):

        return {
            tuple(sorted((type1, type2))): float(cutoff)
            for (type1, type2), cutoff in allowed_bonds.items()
        }

    def _bond_key(self, atom1, atom2):

        return tuple(sorted((int(atom1), int(atom2))))

    def _angle_key(self, atom1, atom2, atom3):

        path = (int(atom1), int(atom2), int(atom3))
        return min(path, path[::-1])

    def _dihedral_key(self, atom1, atom2, atom3, atom4):

        path = (int(atom1), int(atom2), int(atom3), int(atom4))
        return min(path, path[::-1])

    def _existing_topology_keys(self, section, key_func, width):

        return {
            key_func(*self.file_manager.line_to_array(line)[2:2 + width])
            for line in self.file_manager.find_and_extract_lines(self.config.main_file, section, header=False)
        }

    def _inject_xyz_atoms_and_bonds(
        self,
        xyz_filename,
        element_to_atom_type,
        allowed_bonds,
        type_charges=None,
        new_residue_idx="42",
        atom_index_offset=0,
        bond_index_offset=None,
        include_drude=False,
        drude_suffix="-DRUD",
    ):

        '''
        Inject nanotube atoms and cutoff-derived bonds from an XYZ file. Element
        labels are mapped to force-field atom types, bonds are generated only
        for allowed atom-type pairs, and optional Drude particles are inserted
        immediately after each core atom.
        '''

        enabled = self.config.enabled_xdata_sections
        write_bonds = "Bonds" in enabled
        need_bond_network = bool(
            enabled & {"Bonds", "Angles", "Dihedrals", "Impropers"}
        ) or self.config.functionalisation_requested

        type_charges = type_charges or {}

        lines = self.file_manager.read_file(xyz_filename)
        main_lines = self.file_manager.read_file(self.config.main_file)

        if atom_index_offset is None:
            atom_index_offset = self.file_manager.find_last_section_idx(main_lines, "Atoms") or 0
        if write_bonds and bond_index_offset is None:
            bond_index_offset = self.file_manager.find_last_section_idx(main_lines, "Bonds") or 0

        atom_section_end = self.file_manager.find_injection_point(main_lines, "Atoms")
        core_atoms = []
        inject_idx = 1

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            element = parts[0]
            atom_type = element_to_atom_type.get(element)
            if atom_type is None:
                continue

            x, y, z = map(float, parts[1:4])
            new_atom_index = int(atom_index_offset) + inject_idx
            new_atom_type = self.get_new_atom_type(atom_type)
            if new_atom_type is None:
                raise ValueError(f"Could not get atom code for atom '{atom_type}'")

            charge = type_charges.get(atom_type, type_charges.get(element, 0.0))
            new_line_arr = [
                f"{new_atom_index}",
                f"{int(new_residue_idx)}",
                f"{int(new_atom_type)}",
                f"{float(charge):.3f}",
                f"{float(x):.10f}",
                f"{float(y):.10f}",
                f"{float(z):.10f}",
                f"#",
                f"{atom_type}",
            ]

            new_line = self.file_manager.standardise_line(new_line_arr, "Atoms")
            self.file_manager.insert_line(main_lines, atom_section_end + inject_idx - 1, new_line)
            core_atoms.append({
                "atom_index": new_atom_index,
                "atom_type": atom_type,
                "x": float(x),
                "y": float(y),
                "z": float(z),
            })
            inject_idx += 1

            if include_drude:
                drude_atom_index = int(atom_index_offset) + inject_idx
                drude_atom_type = f"{atom_type}{drude_suffix}"
                drude_type_idx = self.get_new_atom_type(drude_atom_type)
                if drude_type_idx is None:
                    raise ValueError(f"Could not get atom code for atom '{drude_atom_type}'")

                drude_charge = type_charges.get(drude_atom_type, -1 * float(charge))
                new_line_arr = [
                    f"{drude_atom_index}",
                    f"{int(new_residue_idx)}",
                    f"{int(drude_type_idx)}",
                    f"{float(drude_charge):.3f}",
                    f"{float(x):.10f}",
                    f"{float(y):.10f}",
                    f"{float(z):.10f}",
                    f"#",
                    f"{drude_atom_type}",
                ]

                new_line = self.file_manager.standardise_line(new_line_arr, "Atoms")
                self.file_manager.insert_line(main_lines, atom_section_end + inject_idx - 1, new_line)
                core_atoms[-1]["drude_atom_index"] = drude_atom_index
                core_atoms[-1]["drude_atom_type"] = drude_atom_type
                inject_idx += 1

        if not core_atoms:
            raise ValueError(f"No supported nanotube atoms found in {xyz_filename}")

        self.file_manager.write_file(main_lines, self.config.main_file)
        if not need_bond_network:
            self.generated_bonds_data = []
            return []

        allowed_bonds = self._normalise_allowed_bond_cutoffs(allowed_bonds)
        if write_bonds:
            main_lines = self.file_manager.read_file(self.config.main_file)
            bond_section_end = self.file_manager.find_injection_point(main_lines, "Bonds")
            last_bond_idx = self.file_manager.find_last_section_idx(main_lines, "Bonds") or 0
            bond_index_offset = last_bond_idx if bond_index_offset is None else max(int(bond_index_offset), last_bond_idx)
            existing_bonds = self._existing_topology_keys("Bonds", self._bond_key, 2)
        else:
            bond_index_offset = 0
            existing_bonds = set()
        bonds_data = []
        inject_idx = 1

        if include_drude and write_bonds:
            for atom in core_atoms:
                bond_key = self._bond_key(atom["atom_index"], atom["drude_atom_index"])
                if bond_key in existing_bonds:
                    continue

                bond_type = self.get_new_bond_type(atom["atom_type"], atom["drude_atom_type"])
                if bond_type is None:
                    raise ValueError(
                        f"Could not get bond type for atoms "
                        f"'{atom['atom_type']}' and '{atom['drude_atom_type']}'"
                    )

                new_bond_index = int(bond_index_offset) + inject_idx
                new_line_arr = [
                    f"{new_bond_index}",
                    f"{bond_type}",
                    f"{atom['atom_index']}",
                    f"{atom['drude_atom_index']}",
                    f"#",
                    f"{atom['atom_type']}",
                    f"{atom['drude_atom_type']}",
                ]

                new_line = self.file_manager.standardise_line(new_line_arr, "Bonds")
                self.file_manager.insert_line(main_lines, bond_section_end + inject_idx - 1, new_line)
                existing_bonds.add(bond_key)
                inject_idx += 1

        for i, atom_i in enumerate(core_atoms):
            pi = np.array([atom_i["x"], atom_i["y"], atom_i["z"]], dtype=float)
            for atom_j in core_atoms[i + 1:]:
                key = tuple(sorted((atom_i["atom_type"], atom_j["atom_type"])))
                cutoff = allowed_bonds.get(key)
                if cutoff is None:
                    continue

                pj = np.array([atom_j["x"], atom_j["y"], atom_j["z"]], dtype=float)
                if np.linalg.norm(pi - pj) > cutoff:
                    continue

                bond_key = self._bond_key(atom_i["atom_index"], atom_j["atom_index"])
                if bond_key in existing_bonds:
                    continue

                new_bond_index = int(bond_index_offset) + inject_idx
                bond_type = ""
                if write_bonds:
                    bond_type = self.get_new_bond_type(
                        atom_i["atom_type"], atom_j["atom_type"]
                    )
                    if bond_type is None:
                        raise ValueError(
                            f"Could not get bond type for atoms "
                            f"'{atom_i['atom_type']}' and '{atom_j['atom_type']}'"
                        )

                    new_line_arr = [
                        f"{new_bond_index}",
                        f"{bond_type}",
                        f"{atom_i['atom_index']}",
                        f"{atom_j['atom_index']}",
                        f"#",
                        f"{atom_i['atom_type']}",
                        f"{atom_j['atom_type']}",
                    ]

                    new_line = self.file_manager.standardise_line(new_line_arr, "Bonds")
                    self.file_manager.insert_line(main_lines, bond_section_end + inject_idx - 1, new_line)
                    existing_bonds.add(bond_key)
                bonds_data.append([
                    str(new_bond_index),
                    str(bond_type),
                    str(atom_i["atom_index"]),
                    str(atom_j["atom_index"]),
                    atom_i["atom_type"],
                    atom_j["atom_type"],
                ])
                inject_idx += 1

        if write_bonds:
            self.file_manager.write_file(main_lines, self.config.main_file)

        if not bonds_data:
            my_log_file.warning("No nanotube core bonds were generated.")

        self.generated_bonds_data = bonds_data
        return bonds_data

    def write_default_species(self):

        '''
        Extract the Nanotube-Species fragment definition from the force-field
        fragments file and inject its atom-type coefficients into the main file.
        This sets up the baseline nanotube species before any nanotube structure
        is written.

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

        lines = self.file_manager.find_and_extract_subsection(self.config.fragments_file, 
                                                              structure_type="Fragment", 
                                                              structure="Nanotube-Species")

        self.inject_atom_coefficients(lines)
        my_log_file.info("Injected default nanotube species coefficients.")

    def write_defaults(self):

        return self.write_default_species()


    # =============================================
    # GET INTEGER TYPES FROM ATOM TYPE NAMES
    # =============================================

    def get_new_atom_type(
        self,
        atom_type_1,
        source_atom_number_type=None,
        source_lines=None,
    ):

        '''
        Look up the integer type index for a named atom type by scanning the
        Masses section of the main file. Returns None if the type is not found
        (and logs an error in verbose mode).

        Parameters:
        ----------
        atom_type_1 : str
            Atom type name to search for (e.g. 'CG2R61', 'HGR62').
        source_atom_number_type : int, optional
            Local Masses index from a molecule template. When supplied with
            source_lines, its Drude coefficients are matched so duplicate
            named types with different per-site Drude values remain distinct.
        source_lines : list of str, optional
            Molecule template containing source_atom_number_type.

        Returns:
        -------
        mass_idx : int or None
            The LAMMPS type index for the atom type, or None if not registered.

        Raises:
        ------
        None
        '''

        source_drude_coeffs = None
        if source_atom_number_type is not None and source_lines is not None:
            for line in self.file_manager.find_and_extract_lines(
                source_lines, "Masses", header=False
            ):
                parts = self.file_manager.line_to_array(line)
                if str(parts[0]) == str(source_atom_number_type):
                    if len(parts) >= 11:
                        source_drude_coeffs = tuple(map(
                            float, parts[10].strip("()").split(",")
                        ))
                    break

        lines = self.file_manager.read_file(self.config.main_file)

        in_masses_section = False

        for line in lines:
            line = line.strip()

            if not line:
                continue  # Skip empty lines

            if line.startswith("Masses"):
                in_masses_section = True
                continue

            if in_masses_section:
                if line.startswith("def ") or line.startswith("#") or line.startswith("Pair"):
                    break  # End of Masses section
                
                if line == "":
                    continue

                parts = self.file_manager.line_to_array(line)

                same_drude = (
                    source_drude_coeffs is None
                    or (
                        len(parts) >= 11
                        and tuple(map(float, parts[10].strip("()").split(",")))
                        == source_drude_coeffs
                    )
                )
                if parts[3] == atom_type_1 and same_drude:
                    return int(parts[0])  # mass type index
        if self.config.verbose:
            my_log_file.error(f"Could not find atom type '{atom_type_1}' in Masses section.")
        return None

    def get_new_bond_type(self, atom_type_1, atom_type_2):

        '''
        Look up the integer bond type index for a pair of atom type names by
        scanning the Bond Coeffs section. The match is order-insensitive.
        Returns None if no matching bond type is found (logged in verbose mode).

        Parameters:
        ----------
        atom_type_1 : str
            First atom type name (e.g. 'CG2R61').
        atom_type_2 : str
            Second atom type name.

        Returns:
        -------
        bond_idx : str or None
            The LAMMPS bond type index string, or None if not found.

        Raises:
        ------
        None
        '''

        # Stay inside Bond Coeffs; Bonds rows have # comments too.
        for line in self.file_manager.find_and_extract_lines(self.config.main_file, "Bond Coeffs", header=False):
            if "#" not in line:
                continue
            data, bond_type = line.split("#", 1)
            atoms = bond_type.strip().split()
            bond_idx = data.split()[0]  # bond type index
            if sorted(atoms) == sorted([atom_type_1, atom_type_2]):
                return bond_idx
        if self.config.verbose:
            my_log_file.error(f"Could not find bond type for atoms '{atom_type_1}' and '{atom_type_2}'.")  
        return None

    def get_new_angle_type(self, atom_type_1, atom_type_2, atom_type_3):

        '''
        Look up the integer angle type index for an ordered triplet of atom type
        names from the Angle Coeffs section. Checks both the forward and reversed
        ordering and returns a flipped flag so that node indices can be reordered
        correctly. Returns None if no match is found.

        Parameters:
        ----------
        atom_type_1 : str
            First atom type in the angle (the terminal atom).
        atom_type_2 : str
            Central atom type.
        atom_type_3 : str
            Second terminal atom type.

        Returns:
        -------
        angle_idx : str or None
            LAMMPS angle type index string, or None if not found.
        flipped : bool
            True if the match required reversing the order, meaning node indices
            1 and 3 should be swapped before writing the angle line.

        Raises:
        ------
        None
        '''

        atom_list = [atom_type_1, atom_type_2, atom_type_3]
        flipped_atom_list = [atom_type_3, atom_type_2, atom_type_1]

        # Stay inside Angle Coeffs; Angles rows have # comments too.
        for line in self.file_manager.find_and_extract_lines(self.config.main_file, "Angle Coeffs", header=False):
            if "#" not in line:
                continue
            data, angle_type = line.split("#", 1)
            atoms = angle_type.strip().split()
            angle_idx = data.split()[0]  # angle type index

            # angles can match in either direction
            if atoms == atom_list:
                flipped = False
                return angle_idx, flipped

            elif atoms == flipped_atom_list:
                flipped = True
                return angle_idx, flipped

        if self.config.verbose:
            my_log_file.error(f"Could not find angle type for atoms '{atom_type_1}', '{atom_type_2}', '{atom_type_3}'.")  
        return None

    def get_new_dihedral_types(self, atom_type_1, atom_type_2, atom_type_3, atom_type_4):

        '''
        Look up all dihedral type indices for an ordered quadruplet of atom types
        from the Dihedral Coeffs section. Checks both forward and reversed orderings
        and returns flipped flags. Returns an empty list if no entries match.

        Parameters:
        ----------
        atom_type_1 : str
            First atom type in the dihedral chain.
        atom_type_2 : str
            Second atom type.
        atom_type_3 : str
            Third atom type.
        atom_type_4 : str
            Fourth atom type.

        Returns:
        -------
        matches : list of tuple
            Each tuple is (dihedral_idx, flipped). flipped is True if the reversed
            ordering was matched; the caller should reverse both the type names
            and the corresponding node indices.

        Raises:
        ------
        None
        '''

        matches = []
        atom_list = [atom_type_1, atom_type_2, atom_type_3, atom_type_4]
        flipped_atom_list = [atom_type_4, atom_type_3, atom_type_2, atom_type_1]

        # Stay inside Dihedral Coeffs; Dihedrals rows have # comments too.
        for line in self.file_manager.find_and_extract_lines(self.config.main_file, "Dihedral Coeffs", header=False):
            if "#" not in line:
                continue
            data, dihedral_type = line.split("#", 1)
            atoms = dihedral_type.strip().split()
            dihedral_idx = data.split()[0]  # dihedral type index

            # match either direction
            if atoms == atom_list:
                matches.append((dihedral_idx, False))

            elif atoms == flipped_atom_list:
                matches.append((dihedral_idx, True))
        if matches:
            return matches
        if self.config.verbose:
            my_log_file.error(f"Could not find dihedral type for atoms '{atom_type_1}', '{atom_type_2}', '{atom_type_3}', '{atom_type_4}'.")
        return []

    def get_new_dihedral_type(self, atom_type_1, atom_type_2, atom_type_3, atom_type_4):

        '''
        Look up the first dihedral type index for an ordered quadruplet of atom
        types. Kept for callers that expect a single type.
        '''

        matches = self.get_new_dihedral_types(atom_type_1, atom_type_2, atom_type_3, atom_type_4)
        if matches:
            return matches[0]
        return None
    
    def get_new_improper_type(self, atom_type_1, atom_type_2, atom_type_3, atom_type_4, forward_only=True):

        '''
        Look up the improper type index for an ordered quadruplet of atom types
        from the Improper Coeffs section. By default only checks the forward
        ordering (forward_only=True) since impropers are centre-first and a
        fully reversed match would displace the central atom to position 4.
        Set forward_only=False to also check the reversed ordering, e.g. when
        called from dihedral lookup where reversal is physically equivalent.

        Parameters:
        ----------
        atom_type_1 : str
            First atom type; must be the central sp2 atom for impropers.
        atom_type_2 : str
            Second atom type.
        atom_type_3 : str
            Third atom type.
        atom_type_4 : str
            Fourth atom type.
        forward_only : bool, optional
            If True, only the forward ordering is matched (default True).
            If False, the fully reversed ordering is also tried.

        Returns:
        -------
        improper_idx : str or None
            LAMMPS improper type index string, or None if not found.
        flipped : bool
            True if the reversed ordering was matched (always False when
            forward_only=True).

        Raises:
        ------
        None
        '''

        atom_list         = [atom_type_1, atom_type_2, atom_type_3, atom_type_4]
        flipped_atom_list = [atom_type_4, atom_type_3, atom_type_2, atom_type_1]

        # Stay inside Improper Coeffs; Impropers rows have # comments too.
        for line in self.file_manager.find_and_extract_lines(self.config.main_file, "Improper Coeffs", header=False):
            if "#" not in line:
                continue
            data, improper_type = line.split("#", 1)
            atoms = improper_type.strip().split()
            improper_idx = data.split()[0]

            if atoms == atom_list:
                return improper_idx, False

            if not forward_only and atoms == flipped_atom_list:
                return improper_idx, True

        if self.config.verbose:
            my_log_file.error(
                f"Could not find improper type for atoms "
                f"'{atom_type_1}', '{atom_type_2}', '{atom_type_3}', '{atom_type_4}'."
            )
        return None
    

    # =============================================
    # GET NAME TYPES FROM ATOM TYPE INTEGER
    # =============================================

    def get_atom_type_from_int_type(self, int_type):

        '''
        Look up the integer type index for a named atom type by scanning the
        Masses section of the main file. Returns None if the type is not found
        (and logs an error in verbose mode).

        Parameters:
        ----------
        int_type : int
            Integer type index to search for.

        Returns:
        -------
        atom_type : str or None
            The atom type name corresponding to the given integer index, or None
        Raises:
        ------
        None
        '''

        lines = self.file_manager.read_file(self.config.main_file)

        in_masses_section = False

        for line in lines:
            line = line.strip()

            if not line:
                continue  # Skip empty lines

            if line.startswith("Masses"):
                in_masses_section = True
                continue

            if in_masses_section:
                if line.startswith("def ") or line.startswith("#") or line.startswith("Pair"):
                    break  # End of Masses section
                
                if line == "":
                    continue

                parts = self.file_manager.line_to_array(line)

                if parts[0] == str(int_type):
                    return parts[3]  # atom type name
        if self.config.verbose:
            my_log_file.error(f"Could not find integer type '{int_type}' in Masses section.")
        return None


    # =============================================
    # OTHER GET FUNCTIONALITY 
    # =============================================

    def get_atom_type_from_index(self, filename=None,  new_atom_index=None):

        '''
        Find the atom type name (e.g. 'CG2R61') for a given LAMMPS atom index
        by reading the comment field after the '#' delimiter in the Atoms section.

        Parameters:
        ----------
        filename : str, optional
            Path to an xdata file to search. If None, the main file is used.
        new_atom_index : int
            The LAMMPS atom index to look up.

        Returns:
        -------
        atom_code : str or None
            The atom type name string, or None if the index is not found.

        Raises:
        ------
        None
        '''

        if filename is None:
            filename = self.config.main_file
            lines = self.file_manager.read_file(self.config.main_file)
        else: 
            lines = self.file_manager.read_file(filename)

        in_atoms_section = False

        for line in lines:
            if "Atoms" in line:
                in_atoms_section = True
                continue
            if in_atoms_section:
                if line.strip("\n") == "":  # Skip empty lines
                    continue
            
                if "Bond" in line:
                    break  # end of section

                parts = line.strip('\n').split('#')
                atom_code = parts[1].strip()
                index = parts[0].split()[0].strip()# atom index 
                if str(index) == str(new_atom_index):
                    return atom_code
        return None
    
    def get_atom_element(self, atom_type, filename_or_lines=None):

        '''
        Retrieve the element symbol for a named atom type by looking it up in
        the Masses section comment fields. Nanotube species are resolved from
        config.default_species_by_type first.

        Parameters:
        ----------
        atom_type : str
            Atom type name to resolve (e.g. 'CG2R61', 'HGR62').
        filename_or_lines : str or list of str, optional
            File path or pre-loaded lines to search. Defaults to the main file.

        Returns:
        -------
        element_symbol : str or None
            One- or two-character element symbol (e.g. 'C', 'H'), or None if
            the type cannot be found.

        Raises:
        ------
        None
        '''  

        if atom_type in self.config.default_species_by_type:
            return self.config.default_species_by_type[atom_type]["atom_element"]

        if filename_or_lines is None:
            lines = self.file_manager.read_file(self.config.main_file)
        else:
            lines = self.file_manager.read_file(filename_or_lines)

        in_masses_section = False

        for line in lines:
            line = line.strip()

            if not line:
                continue  # Skip empty lines

            if line.startswith("Masses"):
                in_masses_section = True
                continue

            if in_masses_section:
                if line.startswith("Atoms")  or line.startswith("Pair"):
                    break  # End of Masses section
                
                if line == "":
                    continue

                parts = self.file_manager.line_to_array(line)

                if str(parts[3]) == str(atom_type):
                    return str(parts[4]) # element symbol
                
        my_log_file.error(f"Could not find element for atom type '{atom_type}'.")
        return None

    def get_atom_mass(self, atom_number_type):

        '''
        Retrieve the atomic mass for a given atom type integer index from the
        Masses section of the main file.

        Parameters:
        ----------
        atom_number_type : int
            LAMMPS integer type index (column 1 of the Masses section).

        Returns:
        -------
        atom_mass : float or None
            Mass in g/mol, or None if the index is not found.

        Raises:
        ------
        None
        '''  

        lines = self.file_manager.read_file(self.config.main_file)

        in_masses_section = False

        for line in lines:
            line = line.strip()

            if not line:
                continue  # Skip empty lines

            if line.startswith("Masses"):
                in_masses_section = True
                continue

            if in_masses_section:
                if line.startswith("def ") or line.startswith("#") or line.startswith("Pair"):
                    break  # End of Masses section
                
                if line == "":
                    continue

                parts = self.file_manager.line_to_array(line)

                if str(parts[0]) == str(atom_number_type):
                    return float(parts[1])

        return None

    def get_atoms_of_residue(self, residue_index):

        '''
        Return the set of LAMMPS atom indices belonging to a given residue index,
        by scanning the Atoms section of the main file.

        Parameters:
        ----------
        residue_index : int
            Residue index to look up (column 2 of the Atoms section).

        Returns:
        -------
        atom_indexes : set of int
            All atom indices assigned to the specified residue.

        Raises:
        ------
        None
        '''  

        atom_indexes = set()

        main_lines = self.file_manager.read_file(self.config.main_file)
        section_lines = self.file_manager.find_and_extract_lines(self.config.main_file, "Atoms")

        for line in section_lines:
            if line.strip() == "" or line.startswith("Atoms"):
                continue
            parts = self.file_manager.line_to_array(line)
            if parts[1] == str(residue_index):
                atom_indexes.add(int(parts[0]))

        return atom_indexes

    def get_residue_of_atom(self, atom_index):

        '''
        Return the residue index for a single atom by scanning the Atoms section
        of the main file.

        Parameters:
        ----------
        atom_index : int
            LAMMPS atom index to look up.

        Returns:
        -------
        residue_index : int or None
            Residue index of the atom, or None if the atom is not found.

        Raises:
        ------
        None
        '''

        main_lines = self.file_manager.read_file(self.config.main_file)
        section_lines = self.file_manager.find_and_extract_lines(self.config.main_file, "Atoms")

        for line in section_lines:
            parts = self.file_manager.line_to_array(line)
            if parts[0] == str(atom_index):
                return int(parts[1])

        return None

    def get_other_atoms_in_same_residue(self, atom_index):

        '''
        Return all atom indices in the same residue as a given atom, by first
        resolving the residue index and then collecting all atoms for that residue.

        Parameters:
        ----------
        atom_index : int
            LAMMPS atom index whose residue-mates are to be found.

        Returns:
        -------
        atom_indexes : set of int
            All atom indices sharing the same residue, including the input atom.
            Returns an empty set if the atom's residue cannot be determined.

        Raises:
        ------
        None
        '''

        residue_index = self.get_residue_of_atom(atom_index)
        if residue_index is None:
            return []
        
        atom_indexes = self.get_atoms_of_residue(residue_index)

        return atom_indexes

    def get_system_charge(self, filename_or_lines=None, residue_index=None): 

        '''
        Sum the partial charges of all atoms (or a specific residue) from the
        Atoms section of the main file. Used to check overall charge neutrality
        and to determine how many counter-ions are needed.

        Parameters:
        ----------
        filename_or_lines : str or list of str, optional
            File path or pre-loaded lines. Defaults to the main file.
        residue_index : str or int, optional
            If provided, only atoms matching this residue index are summed.
            If None, all atoms contribute.

        Returns:
        -------
        total_charge : float
            Sum of partial charges across the selected atoms.

        Raises:
        ------
        None
        '''

        if filename_or_lines is not None:
            lines = self.file_manager.read_file(filename_or_lines)
        else:
            lines = self.file_manager.read_file(self.config.main_file)

        in_section = False
        total_charge = 0.0

        for line in lines:
            if line.startswith("Atoms"):
                in_section = True
                continue 
            if ('Bond' in line or 'Pair' in line) and in_section: 
                break
            if in_section and line.strip() == "":
                continue
            elif in_section and line.strip() != "":

                parts = self.file_manager.line_to_array(line)

                if residue_index is None:
                    total_charge += float(parts[3])
                else:
                    if parts[1] == str(residue_index):
                        total_charge += float(parts[3])

        return total_charge

    def get_atom_positions_all(self, filename_or_lines=None):

        '''
        Parse the Atoms section and return arrays of positions, indices, type
        names, and residue indices for every atom in the file.

        Parameters:
        ----------
        filename_or_lines : str or list of str, optional
            File path or pre-loaded lines. Defaults to the main file.

        Returns:
        -------
        atom_positions : (N,3) np.ndarray of float64
            Cartesian coordinates of all N atoms.
        indexes : list of int
            LAMMPS atom indices in file order.
        types : list of str
            Atom type name for each atom (from the comment column).
        residues : list of int
            Residue index for each atom.

        Raises:
        ------
        None
        '''

        if filename_or_lines is not None:
            lines = self.file_manager.read_file(filename_or_lines)
        else:
            lines = self.file_manager.read_file(self.config.main_file)

        in_section = False
        atom_positions, indexes, types, residues = [], [], [], []
        
        for line in lines:
            if line.startswith("Atoms"):
                in_section = True
                continue 
            if line.startswith("Bond"): 
                break
            if in_section and line.strip() == "":
                continue
            elif in_section and line.strip() != "":
                parts = line.split()

                types.append(parts[-1])
                residues.append(int(parts[1]))
                indexes.append(int(parts[0]))

                atom_positions.append([
                    float(parts[4]), # x
                    float(parts[5]), # y
                    float(parts[6]), # z
                ])

        atom_positions = np.array(atom_positions, dtype=np.float64)

        return atom_positions, indexes, types, residues

    def get_atom_positions_residue(self, filename_or_lines=None, residue="42"):

        '''
        Parse the Atoms section and return positions, indices, and types for
        all atoms belonging to a single residue.

        Parameters:
        ----------
        filename_or_lines : str or list of str, optional
            File path or pre-loaded lines. Defaults to the main file.
        residue : str, optional
            Residue index string to filter by (default '42', the nanotube residue).

        Returns:
        -------
        structure_atoms : list of tuple of float
            (x, y, z) tuples for each matched atom.
        indexes : list of int
            LAMMPS atom indices of the matched atoms.
        types : list of str
            Atom type names of the matched atoms.

        Raises:
        ------
        None
        '''

        if filename_or_lines is not None:
            lines = self.file_manager.read_file(filename_or_lines)
        else:
            lines = self.file_manager.read_file(self.config.main_file)

        in_section = False
        structure_atoms = []
        indexes = []
        types = []

        for line in lines:
            if line.startswith("Atoms"):
                in_section = True
                continue 
            if line.startswith("Bond"): 
                break
            if in_section and line.strip() == "":
                continue
            elif in_section and line.strip() != "":
                parts = line.split()
                if parts[1] == residue:


                    indexes.append(int(parts[0]))
                    types.append(parts[-1])

                    structure_atoms.append((
                        float(parts[4]), # x
                        float(parts[5]), # y
                        float(parts[6]), # z
                    ))

        return structure_atoms, indexes, types
    


    # ================================================================
    # INJECT ANGLES AND DIHEDRALS DERIVED FROM BOND NETWORK
    # ================================================================

    def inject_angles_from_bonds(self, bonds_data, angle_idx_offset=0):

        '''
        Derive all unique angle triplets from the bond adjacency graph, look up
        each angle type in the main file, and inject the resulting Angle lines.
        Drude particles and lone pairs are excluded from the graph traversal.
        Missing angle types are collected and logged as warnings.

        Parameters:
        ----------
        bonds_data : list of list
            Bond records as returned by _create_bonds_data; each entry contains
            [bond_idx, bond_code, atom1_idx, atom2_idx, type1, type2].
        angle_idx_offset : int, optional
            Added to every new angle index to avoid clashing with existing entries.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        if "Angles" not in self.config.enabled_xdata_sections:
            return

        main_lines = self.file_manager.read_file(self.config.main_file)

        angle_section_end = self.file_manager.find_injection_point(self.config.main_file, "Angles")
        adjacency = self._create_adjacency(bonds_data)
        angle_idx_offset = max(angle_idx_offset, self.file_manager.find_last_section_idx(main_lines, "Angles") or 0)
        existing_angles = self._existing_topology_keys("Angles", self._angle_key, 3)
        unique_angles = set()
        missing_angles = set()

        angle_values = []
        inject_idx = 1
        missed = 0

        for node1 in adjacency:

            if self.get_atom_element(self.get_atom_type_from_index(None, node1)) in ["DP", "LP"]: # dont check drude atoms or lone pairs
                continue

            for node2 in adjacency[node1]:

                if self.get_atom_element(self.get_atom_type_from_index(None, node2)) in ["DP", "LP"]: # dont check drude atoms or lone pairs
                    continue

                for node3 in adjacency[node2]:

                    if self.get_atom_element(self.get_atom_type_from_index(None, node3)) in ["DP", "LP"]: # dont check drude atoms or lone pairs
                        continue                    
                    if node3 == node1:                    # Skip if node3 is node 1
                        continue
                    
                    # Create the path
                    path = (node1, node2, node3)
                    
                    # reversed paths are the same angle
                    normalised_path = self._angle_key(*path)
                    if normalised_path in existing_angles:
                        continue
                    unique_angles.add(normalised_path)
        
        for normalised_path in unique_angles:
            node1, node2, node3 = normalised_path
            atom_types = [self.get_atom_type_from_index(self.config.main_file, new_atom_index=node) for node in normalised_path]
            angle_info = self.get_new_angle_type(atom_types[0], atom_types[1], atom_types[2])

            if angle_info is None:
                missing_angles.add((atom_types[0], atom_types[1], atom_types[2]))
                missed += 1
                continue

            angle_type, flipped = angle_info

            if flipped:
                # If the angle is flipped, we need to adjust the node indices
                node1, node2, node3 = node3, node2, node1

            new_angle_index = angle_idx_offset + inject_idx
            
            new_line_arr = [
                f"{new_angle_index}",
                f"{angle_type}",
                f"{node1}",
                f"{node2}",
                f"{node3}",
                f"#",
                f"{atom_types[0]}",
                f"{atom_types[1]}",
                f"{atom_types[2]}"
            ]
            
            new_line = self.file_manager.standardise_line(new_line_arr, "Angles")
            self.file_manager.insert_line(main_lines, angle_section_end + inject_idx - 1, new_line)
            existing_angles.add(normalised_path)
            inject_idx += 1
        
        # Write the modified file back
        self.file_manager.write_file(main_lines, self.config.main_file)

        my_log_file.warning("Missing angles:")
        for angle in missing_angles:
            my_log_file.warning(f" - {angle}")
        my_log_file.warning(f"No: {missed}")

    def inject_dihedrals_from_bonds(self, bonds_data, dihedral_idx_offset=0):

        '''
        Derive all unique dihedral quadruplets from the bond adjacency graph,
        applying three filters to exclude degenerate paths, and inject matching
        Dihedral lines into the main file. Drude particles and lone pairs are
        skipped. Missing dihedral types are logged as warnings.

        Parameters:
        ----------
        bonds_data : list of list
            Bond records as returned by _create_bonds_data.
        dihedral_idx_offset : int, optional
            Added to every new dihedral index.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        # Keep repeated CHARMM terms separate; merge only if the file format changes
        if "Dihedrals" not in self.config.enabled_xdata_sections:
            return

        main_lines = self.file_manager.read_file(self.config.main_file)

        dihedral_section_end = self.file_manager.find_injection_point(self.config.main_file, "Dihedrals")
        dihedral_idx_offset = max(dihedral_idx_offset, self.file_manager.find_last_section_idx(main_lines, "Dihedrals") or 0)

        adjacency = self._create_adjacency(bonds_data)

        seen_paths = set()
        existing_dihedral_terms = set()
        for line in self.file_manager.find_and_extract_lines(self.config.main_file, "Dihedrals", header=False):
            parts = self.file_manager.line_to_array(line)
            existing_dihedral_terms.add((self._dihedral_key(*parts[2:6]), parts[1]))
        missing_dihedrals = set()

        missed = 0
        inject_idx = 1

        for node1 in adjacency:

            if self.get_atom_element(self.get_atom_type_from_index(None, node1)) in ["DP", "LP"]: # dont check drude atoms or lone pairs
                continue

            for node2 in adjacency[node1]:

                if self.get_atom_element(self.get_atom_type_from_index(None, node2)) in ["DP", "LP"]: # dont check drude atoms or lone pairs
                    continue

                for node3 in adjacency[node2]:

                    if self.get_atom_element(self.get_atom_type_from_index(None, node3)) in ["DP", "LP"]: # dont check drude atoms or lone pairs
                        continue

                    # 1. 3-atom loops cannot be governed by a dihedral coefficient
                    if node3 == node1 or node3 in adjacency[node1]:
                        continue
                    
                    for node4 in adjacency[node3]:

                        if self.get_atom_element(self.get_atom_type_from_index(None, node4)) in ["DP", "LP"]: # dont check drude atoms or lone pairs
                            continue

                        # 2. Already in the path
                        if node4 == node1 or node4 == node2 or node4 == node3:
                            continue
                        
                        # 3. Second node cannot be bonded to both node3 and node4 
                        if node2 in adjacency[node4]:
                            continue
                        
                        # Create the path
                        path = (node1, node2, node3, node4)
                        
                        # reversed paths are the same dihedral
                        normalised_path = self._dihedral_key(*path)
                        if normalised_path in seen_paths:
                            continue

                        seen_paths.add(normalised_path)

                        n1, n2, n3, n4 = normalised_path

                        atom_types = [self.get_atom_type_from_index(self.config.main_file, new_atom_index=node) for node in normalised_path]
                        dihedral_infos = self.get_new_dihedral_types(atom_types[0], atom_types[1], atom_types[2], atom_types[3])

                        if not dihedral_infos:
                            missed +=1 
                            missing_dihedrals.add((atom_types[0], atom_types[1], atom_types[2], atom_types[3]))
                            continue

                        for dihedral_type, flipped in dihedral_infos:

                            if (normalised_path, dihedral_type) in existing_dihedral_terms:
                                continue

                            term_n1, term_n2, term_n3, term_n4 = n1, n2, n3, n4
                            term_atom_types = atom_types

                            if flipped:
                                # If the dihedral is flipped, we need to adjust the node indices
                                term_n1, term_n2, term_n3, term_n4 = term_n4, term_n3, term_n2, term_n1
                                term_atom_types = list(reversed(term_atom_types))

                            new_dihedral_index = dihedral_idx_offset + inject_idx

                            new_line_arr = [
                                f"{new_dihedral_index}",
                                f"{dihedral_type}",
                                f"{term_n1}",
                                f"{term_n2}",
                                f"{term_n3}",
                                f"{term_n4}",
                                f"#",
                                f"{term_atom_types[0]}",
                                f"{term_atom_types[1]}",
                                f"{term_atom_types[2]}",
                                f"{term_atom_types[3]}"
                            ]

                            new_line = self.file_manager.standardise_line(new_line_arr, "Dihedrals")
                            self.file_manager.insert_line(main_lines, dihedral_section_end + inject_idx - 1, new_line)
                            existing_dihedral_terms.add((normalised_path, dihedral_type))
                            inject_idx += 1

        # Write the modified file back
        self.file_manager.write_file(main_lines, self.config.main_file)

        my_log_file.warning("Missing dihedrals:")
        for dihedral in missing_dihedrals:
            my_log_file.warning(f" - {dihedral}")
        my_log_file.warning(f"No: {missed}")

    def inject_impropers_from_bonds(self, bonds_data, improper_idx_offset=0):

        '''
        Derive improper quadruplets by finding sp2 centres, which are atoms with
        exactly three heavy-atom neighbours, and generating one out-of-plane improper
        per centre. Convention: centre atom is written first (CHARMM RTF / CGenFF
        standard). All six permutations of the three peripheral atoms are tried
        against Improper Coeffs until a match is found. Drude particles and lone
        pairs are excluded throughout.

        Format of Improper Coeffs entries (must be followed in xdata):
            idx  K  phi0  # CENTRE  a  b  c
        The FIRST atom type is always the central sp2 atom bonded to all three
        others. Peripheral atom order is arbitrary but must match the coeff entry
        exactly (forward or any permutation of the three peripherals).

        Parameters:
        ----------
        bonds_data : list of list
            Bond records as returned by _create_bonds_data.
        improper_idx_offset : int, optional
            Added to every new improper index.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        if "Impropers" not in self.config.enabled_xdata_sections:
            return

        main_lines = self.file_manager.read_file(self.config.main_file)
        improper_section_end = self.file_manager.find_injection_point(self.config.main_file, "Impropers")
        adjacency = self._create_adjacency(bonds_data)

        excluded = {"DP", "LP"}

        def is_heavy(node):
            return self.get_atom_element(self.get_atom_type_from_index(None, node)) not in excluded

        seen_centres  = set()
        missing_impropers = set()
        missed    = 0
        inject_idx = 1

        for centre in adjacency:

            if not is_heavy(centre):
                continue

            heavy_neighbours = [nb for nb in adjacency[centre] if is_heavy(nb)]

            # an sp2 centre has three heavy neighbours
            if len(heavy_neighbours) != 3:
                continue

            # one improper per centre
            if centre in seen_centres:
                continue
            seen_centres.add(centre)

            # try every neighbour order; CHARMM keeps the centre first
            found = False
            for perm in permutations(heavy_neighbours):
                a, b, c = perm
                atom_types = [
                    self.get_atom_type_from_index(self.config.main_file, new_atom_index=node)
                    for node in [centre, a, b, c]
                ]

                improper_info = self.get_new_improper_type(*atom_types)
                if improper_info is None:
                    continue

                improper_type, _ = improper_info  # improper_type is the coeff index string

                new_line_arr = [
                    f"{improper_idx_offset + inject_idx}",
                    f"{improper_type}",
                    f"{centre}", f"{a}", f"{b}", f"{c}",
                    "#",
                    self.get_atom_type_from_index(self.config.main_file, new_atom_index=centre),
                    self.get_atom_type_from_index(self.config.main_file, new_atom_index=a),
                    self.get_atom_type_from_index(self.config.main_file, new_atom_index=b),
                    self.get_atom_type_from_index(self.config.main_file, new_atom_index=c),
                ]

                new_line = self.file_manager.standardise_line(new_line_arr, "Impropers")
                self.file_manager.insert_line(main_lines, improper_section_end + inject_idx - 1, new_line)
                inject_idx += 1
                found = True
                break

            if not found:
                missed += 1
                types = tuple(
                    self.get_atom_type_from_index(self.config.main_file, new_atom_index=node)
                    for node in [centre] + list(heavy_neighbours)
                )
                missing_impropers.add(types)

        self.file_manager.write_file(main_lines, self.config.main_file)

        if missing_impropers:
            my_log_file.warning("Missing impropers:")
            for imp in missing_impropers:
                my_log_file.warning(f"  - {imp}")
        my_log_file.info(f"Impropers: {inject_idx - 1} written, {missed} sp2 centres with no matching coeff.")


    def _get_index_bonds(self, bonds_data, index):

        '''
        Collect all bonds involving a specific atom index, by scanning the
        full bond list and checking both atom columns.

        Parameters:
        ----------
        bonds_data : list of list
            Bond records as returned by _create_bonds_data.
        index : int
            The LAMMPS atom index to find bonds for.

        Returns:
        -------
        count : int
            Number of bonds found involving this atom.
        relevant_bonds : list of list
            Subset of bonds_data where atom1 or atom2 equals index.

        Raises:
        ------
        None
        '''

        count = 0
        relevant_bonds = []
        for bond in bonds_data:
            if int(bond[2]) == index or int(bond[3]) == index:
                relevant_bonds.append(bond)
                count += 1
        return count, relevant_bonds
    
    def _create_bonds_data(self, atom_offset=0):

        '''
        Read all bond entries from the Bonds section of the main file and return
        them as a structured list. Only bonds whose both atom indices exceed
        atom_offset are included, allowing the caller to filter to a sub-structure.

        Parameters:
        ----------
        atom_offset : int, optional
            Minimum atom index threshold; bonds with both atoms above this value
            are included. Default 0 includes all bonds.

        Returns:
        -------
        those_bonds : list of list
            Each entry is [bond_idx, bond_code, atom1_idx, atom2_idx, type1, type2].

        Raises:
        ------
        None
        '''

        if "Bonds" not in self.config.enabled_xdata_sections:
            return []

        lines = self.file_manager.read_file(self.config.main_file)
        
        those_bonds = []

        in_bonds_section = False
        for line in lines:
            line = line.strip()
            
            if line == "Bonds": # Check for Bonds section
                in_bonds_section = True
                continue
                
            if not line:  # Skip empty lines
                continue
                
            if in_bonds_section and (line.startswith("Angle") or line.startswith("Dihedral") or line.startswith("Improper")):
                in_bonds_section = False # Exit if we hit another section
                break
                
            if in_bonds_section:
                if line.strip("\n") == "":
                    continue
                bond_info = line.split("#")
                data, atoms = bond_info
                bond_idx, bond_code, atom1_idx, atom2_idx = data.split()

                if int(atom1_idx) > atom_offset and int(atom2_idx) > atom_offset:

                    atom1_type, atoms2_type = atoms.strip("\n").split()
                    relevant_bond = [bond_idx, bond_code, atom1_idx, atom2_idx, atom1_type, atoms2_type]
                    those_bonds.append(relevant_bond)
                else: 
                    continue

        return those_bonds

    def _create_adjacency(self, bonds_data):

        '''
        Build an undirected adjacency map from a list of bond records, represented
        as a defaultdict of sets so that neighbours can be queried in O(1).

        Parameters:
        ----------
        bonds_data : list of list
            Bond records as returned by _create_bonds_data.

        Returns:
        -------
        adjacency : defaultdict of set
            Maps each atom index to the set of atom indices it is bonded to.

        Raises:
        ------
        None
        '''

        adjacency = defaultdict(set)

        # Parse bond data and build connectivity map
        for bond in bonds_data:
            bond_id, bond_type, atom1_idx, atom2_idx, type1, type2 = bond
            atom1_idx, atom2_idx = int(atom1_idx), int(atom2_idx)

            adjacency[atom1_idx].add((atom2_idx))
            adjacency[atom2_idx].add((atom1_idx))

        return adjacency


    def inject_lj_coefficients(self, molecule_lines_with_header):

        '''
        Rebuild the PairIJ Coeffs section of the main file using Lorentz-Berthelot
        combination rules for all existing atom-type pairs, then append any
        NBFIX (non-standard) interactions from both the current main file and the
        newly supplied molecule. Already-present normal interactions are filtered
        out to prevent duplicate NBFIX entries.

        Parameters:
        ----------
        molecule_lines_with_header : list of str
            Lines of a fragment or solvent xdata block, including section headers,
            from which new NBFIX interactions are read.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        main_lines = self.file_manager.read_file(self.config.main_file)

        # First we get LJ from atoms and combine them together
        main_atoms_lines = self.file_manager.find_and_extract_lines(main_lines, "Masses", header=False)

        #unique_atom_lines = self.get_unique_combination(main_atoms_lines, atoms_lines)
        unique_atom_lines = main_atoms_lines

        def lorentz_combination(sigma1, sigma2):

            '''
            Apply the Lorentz combination rule to compute the mixed sigma
            (sum rule) for a pair of atom types.

            Parameters:
            ----------
            sigma1 : float
                Sigma parameter of the first atom type (Angstrom).
            sigma2 : float
                Sigma parameter of the second atom type (Angstrom).

            Returns:
            -------
            sigma_mix : float
                Combined sigma value (sigma1 + sigma2).

            Raises:
            ------
            None
            '''

            return (sigma1 + sigma2)
        
        def berthelot_combination(epsilon1, epsilon2):

            '''
            Apply the Berthelot combination rule to compute the mixed epsilon
            (geometric mean) for a pair of atom types. Returns 0.0 if either
            epsilon is effectively zero to avoid zero-divide artefacts.

            Parameters:
            ----------
            epsilon1 : float
                Well depth of the first atom type (kcal mol^-1).
            epsilon2 : float
                Well depth of the second atom type (kcal mol^-1).

            Returns:
            -------
            epsilon_mix : float
                Geometric mean sqrt(epsilon1 * epsilon2), or 0.0 if either
                input is below the numerical threshold 1e-10.

            Raises:
            ------
            None
            '''

            eps = 1e-10
            if epsilon1 < eps or epsilon2 < eps:
                return 0.00000

            return (epsilon1 * epsilon2) ** 0.5
        
        atom_atom_arrays = []
        for i, line1 in enumerate(unique_atom_lines): # do the LJ combinations
            
            parts1 = self.file_manager.line_to_array(line1)
            atom_type1 = parts1[3]
            vanderwaals1_e, vanderwaals1_r = parts1[6].strip("()").split(",")

            for ii, line2 in enumerate(unique_atom_lines):

                parts2 = self.file_manager.line_to_array(line2)
                atom_type2 = parts2[3]
                vanderwaals2_e, vanderwaals2_r = parts2[6].strip("()").split(",")

                ij_line_arr = [
                    f"{i+1}",
                    f"{ii+1}",
                    f"{berthelot_combination(float(vanderwaals1_e), float(vanderwaals2_e)):.5f}",
                    f"{lorentz_combination(float(vanderwaals1_r), float(vanderwaals2_r)):.5f}",
                    f"{berthelot_combination(float(vanderwaals1_e), float(vanderwaals2_e)):.5f}",
                    f"{lorentz_combination(float(vanderwaals1_r), float(vanderwaals2_r)):.5f}",
                    f"#",
                    f"{atom_type1}",
                    f"{atom_type2}"
                ]
                atom_atom_arrays.append(ij_line_arr)


        main_nb_fix_array = []
        NBFIX_lines = self.file_manager.find_and_extract_lines(main_lines, "PairIJ Coeffs", header=False)

        for line in NBFIX_lines: # check already existing NBFIX interactions
            parts = self.file_manager.line_to_array(line)
            nb_line = True
            for normal_ij_line in atom_atom_arrays:
                if parts[2:] == normal_ij_line[2:]:
                    nb_line = False
                    if self.config.verbose:
                        my_log_file.warning(f"Skipping NBFIX interaction between {parts[-2]} and {parts[-1]} as it is a normal interaction.")
                    break
            if nb_line:
                nb1, nb2 = parts[-2], parts[-1] # get the two types of atoms.
                ij_line_arr = [
                        f"{self.get_new_atom_type(nb1)}", # so here we have to change typing from the injected structure to main file
                        f"{self.get_new_atom_type(nb2)}",
                        f"{parts[2]}",
                        f"{parts[3]}",
                        f"{parts[4]}",
                        f"{parts[5]}",
                        f"#",
                        f"{nb1}",
                        f"{nb2}"
                    ]
                main_nb_fix_array.append(ij_line_arr)

        new_nb_fix_array = []
        new_nb_fix_lines = self.file_manager.find_and_extract_lines(molecule_lines_with_header, "PairIJ Coeffs", header=False)

        for line in new_nb_fix_lines: # check new NBFIX interactions
            parts = self.file_manager.line_to_array(line)
            nb1, nb2 = parts[-2], parts[-1] # get the two types of atoms.

            already_nbfixed = False
            for existing_line in atom_atom_arrays:
                # this is a normal interactiopn, skip it.
                if self.file_manager.line_to_array(line)[2:] == existing_line[2:]:
                    if self.config.verbose:
                        my_log_file.warning(f"Skipping new NBFIX interaction between {nb1} and {nb2} as it is a normal interaction.")
                    already_nbfixed = True
                    break

            for existing_line in main_nb_fix_array:
                if self.file_manager.line_to_array(line)[2:] == existing_line[2:]:
                    # this nbfix already exists in main file, skip it.
                    if self.config.verbose:
                        my_log_file.warning(f"Skipping new NBFIX interaction between {nb1} and {nb2} as it already exists in main file.")
                    already_nbfixed = True
                    break

            if not already_nbfixed:
                ij_line_arr = [
                    f"{self.get_new_atom_type(nb1)}", # so here we have to change typing from the injected structure to main file
                    f"{self.get_new_atom_type(nb2)}",
                    f"{parts[2]}",
                    f"{parts[3]}",
                    f"{parts[4]}",
                    f"{parts[5]}",
                    f"#",
                    f"{nb1}",
                    f"{nb2}"
                ]
                new_nb_fix_array.append(ij_line_arr)  

                ij_line_arr = [ # we need to add the inverse too
                    f"{self.get_new_atom_type(nb2)}", # so here we have to change typing from the injected structure to main file
                    f"{self.get_new_atom_type(nb1)}",
                    f"{parts[2]}",
                    f"{parts[3]}",
                    f"{parts[4]}",
                    f"{parts[5]}",
                    f"#",
                    f"{nb2}",
                    f"{nb1}"
                ]

                new_nb_fix_array.append(ij_line_arr)  

        collected_pairwise = atom_atom_arrays + main_nb_fix_array + new_nb_fix_array
        
        write_lines = ["PairIJ Coeffs\n", "\n"] # rebuild the interaction data
        for line in collected_pairwise:
            write_lines.append(self.file_manager.standardise_line(line, "PairIJ Coeffs"))
        write_lines.append("\n")
        write_lines.append("Atoms\n")

        self.file_manager.find_and_replace_lines(self.config.main_file, "PairIJ Coeffs", "Atoms", write_lines)
        
    def inject_atom_priorities(self, molecule_lines_with_header, priority_level):

        '''
        Register all atom types found in the Masses section of a molecule block
        at a given priority level in config.atom_type_priority. Priority is used
        during overlap removal to determine which atoms are kept.

        Parameters:
        ----------
        molecule_lines_with_header : list of str
            Lines from a fragment or solvent xdata block.
        priority_level : int
            Priority level to assign (lower numbers = higher priority; 0 is reserved
            for the nanotube carbons).

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        molecule_masses_lines = self.file_manager.find_and_extract_lines(molecule_lines_with_header, "Masses", header=False)

        for line in molecule_masses_lines:
            parts = self.file_manager.line_to_array(line)
            atom_type = parts[3]
            self.config.atom_type_priority[priority_level].append(atom_type)

    def inject_atom_coefficients(self, molecule_lines_with_header):

        '''
        Inject atom-type coefficient lines (Masses, Bond Coeffs, Angle Coeffs,
        Dihedral Coeffs, Improper Coeffs) from a molecule block into the main file,
        skipping any entry that already exists. If an entry with the same type name
        but different numeric parameters is found, a warning is logged and the
        existing entry is overridden.

        Parameters:
        ----------
        molecule_lines_with_header : list of str
            Lines from a fragment or solvent xdata block including section headers.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        enabled = self.config.enabled_xdata_sections
        injection_sections = ["Masses"]
        if "Bonds" in enabled or "Anisotropy" in enabled:
            injection_sections.append("Bond Coeffs")
        injection_sections.extend(
            coefficient
            for section, coefficient in [
                ("Angles", "Angle Coeffs"),
                ("Dihedrals", "Dihedral Coeffs"),
                ("Impropers", "Improper Coeffs"),
            ]
            if section in enabled
        )

        main_lines = self.file_manager.read_file(self.config.main_file)

        for section in injection_sections:
            
            main_section_lines = self.file_manager.find_and_extract_lines(main_lines, section, header=False)
            molecule_section_lines = self.file_manager.find_and_extract_lines(molecule_lines_with_header, section, header=False)

            if molecule_section_lines == []:
                continue 

            for line in molecule_section_lines:
                
                exists = False

                mol_parts = self.file_manager.line_to_array(line)

                for main_line in main_section_lines:
                    main_parts = self.file_manager.line_to_array(main_line)

                    idx = main_parts.index("#")
                    
                    if mol_parts[idx:] == main_parts[idx:] and mol_parts[:idx] != main_parts[:idx]:
                        my_log_file.warning(f"Overriding parameters for {mol_parts[idx:]} as new ones provided.")
                    
                    if mol_parts[1:] == main_parts[1:]:
                        exists = True
                        break
                
                if not exists:
                    section_end = self.file_manager.find_injection_point(main_lines, section)
                    last_section_idx = self.file_manager.find_last_section_idx(main_lines, section)

                    new_line_arr = mol_parts.copy()
                    new_line_arr[0] = str(int(last_section_idx) + 1)

                    new_line = self.file_manager.standardise_line(new_line_arr, section) 

                    self.file_manager.insert_line(main_lines, section_end, new_line)

        self.file_manager.write_file(main_lines, self.config.main_file)


    def obliterate_atoms(self, atom_indexes):

        '''
        Remove multiple atoms and all their associated topology entries in a single
        read/write pass per section. Handles Atoms, Bonds, Angles, Dihedrals,
        Impropers, Lone Pairs, and Anisotropy sections.

        Parameters:
        ----------
        atom_indexes : set or list of int
            LAMMPS indices of all atoms to remove.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''  

        delete_section = [
            "Atoms",
            *(
                section
                for section in self.config.section_order
                if section in self.config.enabled_xdata_sections
            ),
        ]
        
        main_lines = self.file_manager.read_file(self.config.main_file)

        for section in delete_section:
            section_lines = self.file_manager.find_and_extract_lines(self.config.main_file, section, header = False)

            for line in section_lines:
                
                if line  != "":
                    parts = self.file_manager.line_to_array(line)

                    if section == "Atoms":
                        if int(parts[0]) in atom_indexes:
                            main_lines.remove(line)

                    if section == "Bonds":
                        if (int(parts[2]) in atom_indexes or int(parts[3]) in atom_indexes):
                            main_lines.remove(line)
                    
                    if section == "Angles":
                        if (int(parts[2]) in atom_indexes or int(parts[3]) in atom_indexes or int(parts[4]) in atom_indexes):
                            main_lines.remove(line)

                    if section == "Dihedrals":
                        if (int(parts[2]) in atom_indexes or int(parts[3]) in atom_indexes or int(parts[4]) in atom_indexes or int(parts[5]) in atom_indexes):
                            main_lines.remove(line)
                        
                    if section == "Impropers":
                        parts = self.file_manager.line_to_array(line)
                        if (int(parts[2]) in atom_indexes or int(parts[3]) in atom_indexes or int(parts[4]) in atom_indexes or int(parts[5]) in atom_indexes):
                            main_lines.remove(line)

                    if section == "Lone Pairs":
                        parts = self.file_manager.line_to_array(line)
                        if (int(parts[1])) in atom_indexes:
                            main_lines.remove(line)

                    if section == "Anisotropy":
                        parts = self.file_manager.line_to_array(line)
                        if (int(parts[1])) in atom_indexes:
                            main_lines.remove(line)

                else:
                    continue

        self.file_manager.write_file(main_lines, self.config.main_file)


    def _round_charge(self, x):

        '''
        Round a floating-point charge value to 6 decimal places, clamping values
        smaller than 1e-6 in magnitude to exactly zero to avoid machine-precision
        artefacts in charge-neutrality checks.

        Parameters:
        ----------
        x : float
            Raw charge value to process.

        Returns:
        -------
        charge : float
            Rounded charge, or 0.0 if |x| < 1e-6.

        Raises:
        ------
        None
        '''

        if np.abs(x) < 1e-6: # machine precision
            return 0.0
        else:
            return round(float(x),6) # rounds to 6 decimal places


    # -------------------------------------------------------------------------
    # DRUDE SUPPORT
    # -------------------------------------------------------------------------

    def _get_drude_coeffs(self, drude_type, atom_number_type=None):

        '''
        Retrieve the polarisability (alpha) and Thole damping (thole) parameters
        for a Drude atom type from the Masses section, where they are stored as
        a parenthesised pair in the last column.

        Parameters:
        ----------
        drude_type : str
            Atom type name of the Drude-polarisable atom (e.g. 'CG2R61').
        atom_number_type : int, optional
            Masses index for an individual atom. This disambiguates duplicate
            named types carrying different per-site Drude coefficients.

        Returns:
        -------
        alpha : float or None
            Polarisability parameter, or None if the type is not found or has
            no Drude parameters.
        thole : float or None
            Thole damping parameter, or None for the same reasons.

        Raises:
        ------
        None
        '''   

        species_entry = self.config.default_species_by_type.get(drude_type)
        if atom_number_type is None and species_entry is not None and "drude_alpha" in species_entry:
            return (
                species_entry.get("drude_alpha"),
                species_entry.get("drude_thole"),
            )

        lines = self.file_manager.read_file(self.config.main_file)

        in_masses_section = False

        for line in lines:
            line = line.strip()

            if not line:
                continue  # Skip empty lines

            if line.startswith("Masses"):
                in_masses_section = True
                continue

            if in_masses_section:
                if line.startswith("Atoms")  or line.startswith("Pair"):
                    break  # End of Masses section
                
                if line == "":
                    continue

                parts = self.file_manager.line_to_array(line)

                matches = (
                    str(parts[0]) == str(atom_number_type)
                    if atom_number_type is not None
                    else str(parts[3]) == str(drude_type)
                )
                if matches:
                    if len(parts) < 11:
                        #my_log_file.error(f"Drude parameters missing for atom type '{drude_type}'.")
                        return None, None
                    
                    alpha, thole = parts[10].strip("()").split(",")
                    return float(alpha), float(thole) # alpha, thole
                
        my_log_file.error(f"Could not find drude coeffs for atom type '{drude_type}'.")
        return None, None

    def _get_anisotropic_coeffs_CHARMM36_NDP(self, A11, A22, Kdrude):

        '''
        Compute the three diagonal force-constant elements (K11, K22, K33) of
        the anisotropic Drude polarisability tensor for the CHARMM36m-NDP field.
        Follows the derivation in the CHARMM source (genpsf.F90, drude.F90).

        Parameters:
        ----------
        A11 : float
            CHARMM36m-NDP A11 anisotropy parameter.
        A22 : float
            CHARMM36m-NDP A22 anisotropy parameter.
        Kdrude : float
            Bond spring constant (kcal mol^-1 Angstrom^-2) between the Drude core
            and its Drude particle.

        Returns:
        -------
        K11 : float
            First diagonal element of the polarisability force-constant tensor.
        K22 : float
            Second diagonal element.
        K33 : float
            Third diagonal element.

        Raises:
        ------
        None
        '''    

        A11, A22, Kdrude = float(A11), float(A22), float(Kdrude)

        # ~/charmm/build/cmake/genpsf.F90
        # ~/charmm/build/cmake/drude.F90
        # ~/charmm/build/cmake/consta_ltm.F90

        A33 = 3 - A11 - A22 # https://hpc.nih.gov/apps/charmm/c42b2html/drude.html

        K11_0, K22_0, K33_0 = 1/A11, 1/A22, 1/A33 # intial anisotropic polarisability values

        K11_1, K22_1, K33_1 = Kdrude * K11_0, Kdrude * K22_0, Kdrude * K33_0 # scaled by Drude spring constant

        K33_2 = K33_1 - Kdrude

        K11_2, K22_2 = K11_1 - Kdrude - K33_2, K22_1 - Kdrude - K33_2

        return K11_2, K22_2, K33_2


class Structure_Generator:

    def __init__(self,
                 force_field_manager: Force_Field_Manager,
                 file_manager: File_Manager,
                 config: Config,
                 geometry: Geometry) -> None:
        
        self.force_field_manager : Force_Field_Manager = force_field_manager
        self.file_manager : File_Manager = file_manager
        self.config : Config = config
        self.gt : Geometry = geometry

    def _nanotube_frame(self, n, m, a1, a2):

        '''
        Return the 2-D chiral frame used to roll a lattice into a nanotube.
        The caller supplies the material-specific lattice vectors.
        '''

        Ch  = n*a1 + m*a2
        Ch2 = Ch @ Ch
        Ch_len = np.sqrt(Ch2)
        R = Ch_len / (2*np.pi)

        dR = self.gt.gcd(2*n + m, 2*m + n)
        t1 = (2*m + n) // dR
        t2 = -(2*n + m) // dR
        T = t1*a1 + t2*a2
        T2 = T @ T
        T_len = np.sqrt(T2)

        return {
            "Ch": Ch,
            "Ch2": Ch2,
            "R": R,
            "T": T,
            "T2": T2,
            "T_len": T_len,
        }

    def _center_typed_records(self, records):

        if not records:
            return records

        positions = np.array([[r["x"], r["y"], r["z"]] for r in records], dtype=float)
        centre = np.mean(positions, axis=0)
        centred = []
        for record in records:
            new_record = record.copy()
            new_record["x"] = float(record["x"] - centre[0])
            new_record["y"] = float(record["y"] - centre[1])
            new_record["z"] = float(record["z"] - centre[2])
            centred.append(new_record)
        return centred

    def _renumber_typed_records(self, records):

        renumbered = []
        for i, record in enumerate(records, start=1):
            new_record = record.copy()
            new_record["atom_index"] = i
            renumbered.append(new_record)
        return renumbered

    def _deduplicate_typed_records(self, records, tol=1e-4):

        '''
        Order-preserving deduplication for typed atom records. Atom type is part
        of the key so different species are never collapsed into one atom.
        '''

        seen = set()
        out = []
        for record in records:
            key = (
                record["atom_type"],
                round(float(record["x"]) / tol),
                round(float(record["y"]) / tol),
                round(float(record["z"]) / tol),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(record)
        return self._renumber_typed_records(out)

    def _normalise_allowed_bonds(self, allowed_bonds):

        out = {}
        for pair, cutoff in allowed_bonds.items():
            type1, type2 = pair
            out[tuple(sorted((type1, type2)))] = float(cutoff)
        return out

    def _typed_degrees_open(self, records, allowed_bonds):

        allowed_bonds = self._normalise_allowed_bonds(allowed_bonds)
        deg = np.zeros(len(records), dtype=int)

        for i in range(len(records)):
            rec_i = records[i]
            pi = np.array([rec_i["x"], rec_i["y"], rec_i["z"]], dtype=float)
            for j in range(i + 1, len(records)):
                rec_j = records[j]
                key = tuple(sorted((rec_i["atom_type"], rec_j["atom_type"])))
                cutoff = allowed_bonds.get(key)
                if cutoff is None:
                    continue
                pj = np.array([rec_j["x"], rec_j["y"], rec_j["z"]], dtype=float)
                if np.linalg.norm(pi - pj) <= cutoff:
                    deg[i] += 1
                    deg[j] += 1

        return deg

    def _translate_single_bond_top_typed_records_down(self, records, Lz, allowed_bonds):

        if not records:
            return records

        out = [record.copy() for record in records]
        deg = self._typed_degrees_open(out, allowed_bonds)

        moved = 0
        for i, record in enumerate(out):
            if deg[i] == 1 and float(record["z"]) > 0:
                record["z"] = float(record["z"]) - Lz
                moved += 1

        if moved:
            my_log_file.info(f"Moved {moved} single-bond top multi-species atoms down by Lz.")
        else:
            my_log_file.info("No top-side single-bond multi-species atoms found.")

        remaining = np.count_nonzero(self._typed_degrees_open(out, allowed_bonds) == 1)
        my_log_file.info(f"Single-bond nanotube atoms after correction: {remaining}.")

        return out

    def _roll_nanotube_basis_to_records(self, n, m, Ncells, a1, a2, basis,
                                        allowed_bonds=None, deduplicate=None,
                                        PBC=False, material_name="nanotube"):

        '''
        Roll a material-specific 2-D basis into typed nanotube atom records.
        The basis entries define element, atom type, charge, 2-D position, and
        optional radial shell offset; this helper owns the common chirality
        frame, strip selection, axial repetition, centering, and open-end logic.
        '''

        PATCH_SCAN = 100 # this is a safety margin to ensure we capture all basis atoms that could possibly fall within the rolled unit cell after wrapping

        frame = self._nanotube_frame(n, m, a1, a2)

        N1, N2 = n + m, 2*n + m
        records_2d = []
        for i in range(-N1 - PATCH_SCAN, N1 + PATCH_SCAN):
            for j in range(-N2 - PATCH_SCAN, N2 + PATCH_SCAN):
                origin = i*a1 + j*a2
                for basis_record in basis:
                    position = np.array(basis_record["position"], dtype=float)
                    if basis_record.get("position_type", "cartesian") == "fractional":
                        r = origin + position[0]*a1 + position[1]*a2
                    else:
                        r = origin + position

                    u = (r @ frame["Ch"]) / frame["Ch2"]
                    v = (r @ frame["T"]) / frame["T2"]
                    eps = 1e-10
                    if not (-eps <= u < 1.0 - eps and -eps <= v < 1.0 - eps):
                        continue

                    records_2d.append({
                        "u": u - np.floor(u + eps),
                        "v": v - np.floor(v + eps),
                        "element": basis_record["element"],
                        "atom_type": basis_record["atom_type"],
                        "charge": float(basis_record.get("charge", 0.0)),
                        "shell_offset": float(basis_record.get("shell_offset", 0.0)),
                    })

        records = []
        Lz = Ncells * frame["T_len"]
        for cell in range(Ncells):
            for record in records_2d:
                theta = 2*np.pi * record["u"]
                radius = frame["R"] + record["shell_offset"]
                z = record["v"] * frame["T_len"] + cell*frame["T_len"] - Lz/2.0
                records.append({
                    "atom_index": len(records) + 1,
                    "element": record["element"],
                    "atom_type": record["atom_type"],
                    "charge": record["charge"],
                    "x": float(radius * np.cos(theta)),
                    "y": float(radius * np.sin(theta)),
                    "z": float(z),
                })

        if deduplicate:
            records = self._deduplicate_typed_records(records, tol=1e-4)

        if PBC:
            my_log_file.warning(f"Periodic boundary conditions are applied, {material_name} atoms may have 1 bond.")
        elif allowed_bonds:
            records = self._translate_single_bond_top_typed_records_down(records, Lz, allowed_bonds)

        records = self._renumber_typed_records(self._center_typed_records(records))
        return records, frame, Lz

    def _nanotube_xyz_filename(self, filename):

        folder = self.config.system_folder
        if folder is None and self.config.main_file:
            folder = os.path.dirname(self.config.main_file)
        if folder is None:
            folder = self.config.called_from
        return os.path.join(folder, filename)

    def _write_nanotube_xyz_from_records(self, records, filename, title):

        with open(filename, "w") as f:
            f.write(f"{len(records)}\n")
            f.write(f"{title}\n")
            for record in records:
                f.write(
                    f"{record['element']:<3s} "
                    f"{float(record['x']):>12.5f} "
                    f"{float(record['y']):>12.5f} "
                    f"{float(record['z']):>12.5f}\n"
                )

        return filename

    def _validate_multi_species_coefficients(self, atom_types, allowed_bonds, material_name):

        '''
        Ensure the active xdata force field has the atom and bond types required
        by a multi-species roller before geometry is injected.
        '''

        missing_atoms = [
            atom_type for atom_type in atom_types
            if self.force_field_manager.get_new_atom_type(atom_type) is None
        ]
        if missing_atoms:
            msg = (
                f"{material_name} generation requires atom type(s) "
                f"{', '.join(missing_atoms)} in the active force field Masses section."
            )
            my_log_file.error(msg)
            raise ValueError(msg)

        if "Bonds" in self.config.enabled_xdata_sections:
            missing_bonds = []
            for type1, type2 in allowed_bonds:
                if self.force_field_manager.get_new_bond_type(type1, type2) is None:
                    missing_bonds.append(f"{type1}-{type2}")
            if missing_bonds:
                msg = (
                    f"{material_name} generation requires bond coefficient(s) "
                    f"{', '.join(missing_bonds)} in the active force field Bond Coeffs section."
                )
                my_log_file.error(msg)
                raise ValueError(msg)

    def _inject_topology_from_bonds(self, bonds_data):

        self.force_field_manager.inject_angles_from_bonds(bonds_data=bonds_data)
        self.force_field_manager.inject_dihedrals_from_bonds(bonds_data=bonds_data)
        self.force_field_manager.inject_impropers_from_bonds(bonds_data=bonds_data)


    # ============================================================
    # ROUTING METHODS FOR MULTI-SPECIES NANOTUBE BUILDERS
    # ============================================================

    def write_nanotube(self, n, m, Ncells=7, a=1.44, deduplicate=None, PBC=False):

        '''
        Dispatcher for nanotube rolling. Routes to the appropriate builder
        based on the number of species in the material.

        Parameters:
        ----------
        n : int
            Chiral index n.
        m : int
            Chiral index m.
        Ncells : int, optional
            Number of unit-cell repeats along the tube axis.
        a : float, optional
            Bond length in Angstroms (default 1.44).
        deduplicate : bool, optional
            Passed through to the builder.
        PBC : bool, optional
            If True, apply periodic boundary conditions along the tube axis.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        nt_type = self.config.default_species_nanotube_type
        if nt_type == "BNNT":
            self._write_bnnt_nanotube(n, m, Ncells, deduplicate, PBC)
        elif nt_type in {"MoSSeNT", "MoSSe"}:
            self._write_mosse_nanotube(n, m, Ncells, deduplicate=deduplicate, PBC=PBC)
        elif nt_type in {"MoS2NT", "MoS2"}:
            self._write_mos2_nanotube(n, m, Ncells, deduplicate, PBC)
        else:
            self._write_single_species_nanotube(n, m, Ncells, a, deduplicate, PBC)

    def _write_single_species_nanotube(self, n, m, Ncells=7, a=1.44, deduplicate=None, PBC=False):

        '''
        Single-species nanotube builder. Calls _build_single_species_nanotube to
        generate an XYZ file, then injects atom and bond data into the main xdata
        file and derives angles, dihedrals, and impropers from the bond network.
        Only supports lattices with one element type (e.g. C, Cu, Si).

        Parameters:
        ----------
        n : int
            Chiral index n.
        m : int
            Chiral index m.
        Ncells : int, optional
            Number of unit-cell repeats along the tube axis.
        a : float, optional
            Bond length in Angstroms (default 1.44).
        deduplicate : bool, optional
            Passed through to _build_single_species_nanotube.
        PBC : bool, optional
            If True, apply periodic boundary conditions along the tube axis.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        file = self._build_single_species_nanotube(n, m, Ncells, a, deduplicate, PBC)
        primary = self.config.default_species_primary
        atom_type = primary["atom_type"]
        element = primary["atom_element"]
        bond_cutoff = self.config.get_default_species_bond_param("bond_cutoff", 1.60)
        allowed_bonds = {(atom_type, atom_type): bond_cutoff}
        type_charges = {
            entry["atom_type"]: entry["atom_charge"]
            for entry in self.config.default_species_chain
        }

        bonds_data = self.force_field_manager._inject_xyz_atoms_and_bonds(
            xyz_filename=file,
            element_to_atom_type={element: atom_type},
            allowed_bonds=allowed_bonds,
            type_charges=type_charges,
            include_drude=self.config.drude_polarisable,
        )
        self._inject_topology_from_bonds(bonds_data)

    def _write_multi_species_nanotube(self, n, m, Ncells=7, a=1.44, deduplicate=None, PBC=False):

        '''
        Compatibility dispatcher for implemented multi-species nanotube
        rollers (BNNT, MoS2NT/MoS2, and MoSSeNT/MoSSe). The explicit material
        writers are used by JSON dispatch; this method keeps the older internal
        entrypoint alive.

        Parameters:
        ----------
        n : int
            Chiral index n.
        m : int
            Chiral index m.
        Ncells : int, optional
            Number of unit-cell repeats along the tube axis.
        a : float, optional
            Bond length in Angstroms (default 1.44).
        deduplicate : bool, optional
            Passed through to the builder.
        PBC : bool, optional
            If True, apply periodic boundary conditions along the tube axis.

        Returns:
        -------
        None

        Raises:
        ------
        NotImplementedError
            If the active material is not one of the implemented rough
            multi-species rollers.
        '''

        nt_type = self.config.default_species_nanotube_type
        if nt_type == "BNNT":
            return self._write_bnnt_nanotube(n, m, Ncells, deduplicate, PBC)
        if nt_type in {"MoSSeNT", "MoSSe"}:
            return self._write_mosse_nanotube(n, m, Ncells, deduplicate=deduplicate, PBC=PBC)
        if nt_type in {"MoS2NT", "MoS2"}:
            return self._write_mos2_nanotube(n, m, Ncells, deduplicate, PBC)

        raise NotImplementedError(
            f"Multi-species nanotube rolling ('{nt_type}') "
            f"is not yet implemented."
        )


    # ============================================================
    # BUILD FUNCTIONALITY
    # ============================================================
    
    def _build_bnnt_nanotube(self, n, m, Ncells=7, a=1.4460, deduplicate=None,
                             PBC=False, boron_atom_type="BNTB",
                             nitrogen_atom_type="BNTN"):

        '''
        Build a rough BN nanotube using the graphene honeycomb rolling
        construction. Basis site 0 is boron (BNTB), basis site 1 is nitrogen
        (BNTN), and only B-N bonds are expected downstream.
        '''

        allowed_bonds = {(boron_atom_type, nitrogen_atom_type): 1.65}
        a1 = np.array([np.sqrt(3)*a, 0.0])
        a2 = np.array([np.sqrt(3)/2*a, 1.5*a])
        basis = [
            {"position": [0.0, 0.0], "element": "B", "atom_type": boron_atom_type, "charge": 0.0},
            {"position": [np.sqrt(3)/2*a, 0.5*a], "element": "N", "atom_type": nitrogen_atom_type, "charge": 0.0},
        ]

        records, _, _ = self._roll_nanotube_basis_to_records(
            n=n, m=m, Ncells=Ncells,
            a1=a1, a2=a2, basis=basis,
            allowed_bonds=allowed_bonds,
            deduplicate=deduplicate, PBC=PBC,
            material_name="BNNT",
        )
        my_log_file.info(f"Generated {len(records)} atoms for BNNT({n},{m}) with {Ncells} repeats.")
        return records

    def _build_mos2_nanotube(self, n, m, Ncells=7, bond_length=2.4100,
                             lattice_constant=3.1600, deduplicate=None,
                             PBC=False, mo_atom_type="MOS2MO",
                             s_atom_type="MOS2S"):

        '''
        Build a rough MoS2 nanotube as a rolled TMD sandwich: one central Mo
        layer and paired inner/outer sulfur layers. Only Mo-S bonds are expected
        downstream.
        '''

        a1 = np.array([lattice_constant, 0.0])
        a2 = np.array([0.5*lattice_constant, np.sqrt(3)/2*lattice_constant])
        frame = self._nanotube_frame(n, m, a1, a2)

        in_plane_mo_s = lattice_constant / np.sqrt(3)
        radial_offset_sq = bond_length*bond_length - in_plane_mo_s*in_plane_mo_s
        if radial_offset_sq <= 0:
            raise ValueError("MoS2 bond length must exceed the in-plane Mo-S projection.")
        radial_offset = np.sqrt(radial_offset_sq)

        if frame["R"] <= radial_offset:
            raise ValueError(
                f"MoS2 nanotube radius {frame['R']:.3f} Ang is too small for "
                f"the rough sulfur radial offset {radial_offset:.3f} Ang."
            )

        allowed_bonds = {(mo_atom_type, s_atom_type): 2.70}
        basis = [
            {"position": [0.0, 0.0], "position_type": "fractional", "element": "Mo", "atom_type": mo_atom_type, "charge": 0.0},
            {"position": [1.0/3.0, 1.0/3.0], "position_type": "fractional", "element": "S", "atom_type": s_atom_type, "charge": 0.0, "shell_offset": radial_offset},
            {"position": [1.0/3.0, 1.0/3.0], "position_type": "fractional", "element": "S", "atom_type": s_atom_type, "charge": 0.0, "shell_offset": -radial_offset},
        ]

        records, _, _ = self._roll_nanotube_basis_to_records(
            n=n, m=m, Ncells=Ncells,
            a1=a1, a2=a2, basis=basis,
            allowed_bonds=allowed_bonds,
            deduplicate=deduplicate, PBC=PBC,
            material_name="MoS2",
        )
        my_log_file.info(f"Generated {len(records)} atoms for MoS2NT({n},{m}) with {Ncells} repeats.")
        return records

    def _build_mosse_nanotube(self, n, m, Ncells=7, lattice_constant=3.2300,
                              mo_s_bond_length=2.4100,
                              mo_se_bond_length=2.5200,
                              outer_chalcogen="Se",
                              deduplicate=None, PBC=False,
                              mo_atom_type="MSSMo",
                              s_atom_type="MSSS",
                              se_atom_type="MSSSe"):

        '''
        Build a rough Janus MoSSe nanotube as an XYZ file. The middle shell is
        Mo, while S and Se occupy opposite radial shells so their surface
        identities are preserved before xdata topology injection.
        '''

        if outer_chalcogen not in {"S", "Se"}:
            raise ValueError("MoSSe outer-chalcogen must be 'S' or 'Se'.")

        a1 = np.array([lattice_constant, 0.0])
        a2 = np.array([0.5*lattice_constant, np.sqrt(3)/2*lattice_constant])
        frame = self._nanotube_frame(n, m, a1, a2)

        in_plane_mo_chalcogen = lattice_constant / np.sqrt(3)

        def radial_offset(bond_length):
            radial_offset_sq = bond_length*bond_length - in_plane_mo_chalcogen*in_plane_mo_chalcogen
            if radial_offset_sq <= 0:
                raise ValueError(
                    "MoSSe bond lengths must exceed the in-plane Mo-chalcogen projection."
                )
            return np.sqrt(radial_offset_sq)

        s_offset = radial_offset(mo_s_bond_length)
        se_offset = radial_offset(mo_se_bond_length)

        if frame["R"] <= max(s_offset, se_offset):
            raise ValueError(
                f"MoSSe nanotube radius {frame['R']:.3f} Ang is too small for "
                f"the rough chalcogen radial offsets."
            )

        if outer_chalcogen == "Se":
            s_shell_offset = -s_offset
            se_shell_offset = se_offset
        else:
            s_shell_offset = s_offset
            se_shell_offset = -se_offset

        allowed_bonds = {
            (mo_atom_type, s_atom_type): 2.70,
            (mo_atom_type, se_atom_type): 2.82,
        }
        basis = [
            {"position": [0.0, 0.0], "position_type": "fractional", "element": "Mo", "atom_type": mo_atom_type, "charge": 0.0},
            {"position": [1.0/3.0, 1.0/3.0], "position_type": "fractional", "element": "S", "atom_type": s_atom_type, "charge": 0.0, "shell_offset": s_shell_offset},
            {"position": [1.0/3.0, 1.0/3.0], "position_type": "fractional", "element": "Se", "atom_type": se_atom_type, "charge": 0.0, "shell_offset": se_shell_offset},
        ]

        records, _, _ = self._roll_nanotube_basis_to_records(
            n=n, m=m, Ncells=Ncells,
            a1=a1, a2=a2, basis=basis,
            allowed_bonds=allowed_bonds,
            deduplicate=deduplicate, PBC=PBC,
            material_name="MoSSe",
        )

        filename = self._nanotube_xyz_filename(
            f"SW-MoSSeNT_{n}_{m}_{Ncells}_outer-{outer_chalcogen}.xyz"
        )
        self._write_nanotube_xyz_from_records(
            records,
            filename,
            f"MoSSeNT generated with chirality_kit; outer chalcogen {outer_chalcogen}",
        )

        my_log_file.info(
            f"Generated {len(records)} atoms for MoSSeNT({n},{m}) with "
            f"{Ncells} repeats and outer {outer_chalcogen} shell."
        )
        return filename

    def _build_single_species_nanotube(self, n, m, Ncells=7, a=1.44, deduplicate=None, PBC=False):

        '''
        Generate the 3-D Cartesian coordinates of a single-wall carbon nanotube
        with chiral indices (n,m) and a given number of unit-cell repeats, using
        the graphene rolling construction. Writes the result as an XYZ file and
        returns the file path.

        Parameters:
        ----------
        n : int
            Chiral index n of the nanotube.
        m : int
            Chiral index m of the nanotube.
        Ncells : int, optional
            Number of translational unit cells to stack along the tube axis.
        a : float, optional
            Carbon-carbon bond length of the graphene lattice in Angstroms (default 1.44).
        deduplicate : bool, optional
            If True, remove atoms that are closer than 1e-4 Angstrom after rolling.
        PBC : bool, optional
            If True, keep the raw rolled structure (for periodic simulations).
            If False, corrects top-end stray atoms with the typed-record open-end path.

        Returns:
        -------
        filename : str
            Path to the generated XYZ file in the system output folder.

        Raises:
        ------
        None
        '''

        primary = self.config.default_species_primary
        element = primary["atom_element"]
        atom_type = primary["atom_type"]
        charge = primary["atom_charge"]
        bond_cutoff = self.config.get_default_species_bond_param("bond_cutoff", 1.60)
        allowed_bonds = {(atom_type, atom_type): bond_cutoff}

        a1 = np.array([np.sqrt(3)*a, 0.0])
        a2 = np.array([np.sqrt(3)/2*a, 1.5*a])
        basis = [
            {"position": [0.0, 0.0], "element": element, "atom_type": atom_type, "charge": charge},
            {"position": [np.sqrt(3)/2*a, 0.5*a], "element": element, "atom_type": atom_type, "charge": charge},
        ]

        records, frame, Lz = self._roll_nanotube_basis_to_records(
            n=n, m=m, Ncells=Ncells,
            a1=a1, a2=a2, basis=basis,
            allowed_bonds=allowed_bonds,
            deduplicate=deduplicate, PBC=PBC,
            material_name=f"{element}NT",
        )

        my_log_file.info(f"Generated {len(records)} atoms for {element}({n},{m}) with {Ncells} repeats.")

        filename = self._nanotube_xyz_filename(
            f"SW-{element}NT_{n}_{m}_{Ncells}.xyz"
        )
        self._write_nanotube_xyz_from_records(
            records,
            filename,
            f"{element}NT generated with chirality_kit",
        )

        my_log_file.info(f"Structure generation ({n},{m}) x {Ncells} complete. Radius = {frame['R']:.3f} Angs, Length = {Lz:.3f} Angs.")

        return filename
    

    # ============================================================
    # WRITE FUNCTIONALITY
    # ============================================================

    def _write_bnnt_nanotube(self, n, m, Ncells=7, deduplicate=None, PBC=False):

        b_type = self.config.get_default_species_type_for_element("B")
        n_type = self.config.get_default_species_type_for_element("N")
        if b_type is None or n_type is None:
            raise ValueError("BNNT force field must define B and N nanotube species.")
        allowed_bonds = {(b_type, n_type): 1.65}
        atom_types = [b_type, n_type]
        self._validate_multi_species_coefficients(atom_types, allowed_bonds, "BNNT")

        records = self._build_bnnt_nanotube(
            n=n, m=m, Ncells=Ncells, a=1.4460,
            deduplicate=deduplicate, PBC=PBC,
            boron_atom_type=b_type,
            nitrogen_atom_type=n_type,
        )
        xyz_filename = self._nanotube_xyz_filename(f"SW-BNNT_{n}_{m}_{Ncells}.xyz")
        self._write_nanotube_xyz_from_records(
            records,
            xyz_filename,
            "BNNT generated with chirality_kit",
        )
        bonds_data = self.force_field_manager._inject_xyz_atoms_and_bonds(
            xyz_filename=xyz_filename,
            element_to_atom_type={"B": b_type, "N": n_type},
            allowed_bonds=allowed_bonds,
            type_charges={
                entry["atom_type"]: entry["atom_charge"]
                for entry in self.config.default_species_chain
            },
        )
        self._inject_topology_from_bonds(bonds_data)
        return records, bonds_data

    def _write_mos2_nanotube(self, n, m, Ncells=7, deduplicate=None, PBC=False):

        mo_type = self.config.get_default_species_type_for_element("Mo")
        s_type = self.config.get_default_species_type_for_element("S")
        if mo_type is None or s_type is None:
            raise ValueError("MoS2NT force field must define Mo and S nanotube species.")
        allowed_bonds = {(mo_type, s_type): 2.70}
        atom_types = [mo_type, s_type]
        self._validate_multi_species_coefficients(atom_types, allowed_bonds, "MoS2NT")

        records = self._build_mos2_nanotube(
            n=n, m=m, Ncells=Ncells, bond_length=2.4100,
            lattice_constant=3.1600, deduplicate=deduplicate, PBC=PBC,
            mo_atom_type=mo_type,
            s_atom_type=s_type,
        )
        xyz_filename = self._nanotube_xyz_filename(f"SW-MoS2NT_{n}_{m}_{Ncells}.xyz")
        self._write_nanotube_xyz_from_records(
            records,
            xyz_filename,
            "MoS2NT generated with chirality_kit",
        )
        bonds_data = self.force_field_manager._inject_xyz_atoms_and_bonds(
            xyz_filename=xyz_filename,
            element_to_atom_type={"Mo": mo_type, "S": s_type},
            allowed_bonds=allowed_bonds,
            type_charges={
                entry["atom_type"]: entry["atom_charge"]
                for entry in self.config.default_species_chain
            },
        )
        self._inject_topology_from_bonds(bonds_data)
        return records, bonds_data

    def _write_mosse_nanotube(self, n, m, Ncells=7, outer_chalcogen="Se", deduplicate=None, PBC=False):

        mo_type = self.config.get_default_species_type_for_element("Mo")
        s_type = self.config.get_default_species_type_for_element("S")
        se_type = self.config.get_default_species_type_for_element("Se")
        if mo_type is None or s_type is None or se_type is None:
            raise ValueError("MoSSeNT force field must define Mo, S, and Se nanotube species.")
        allowed_bonds = {
            (mo_type, s_type): 2.70,
            (mo_type, se_type): 2.82,
        }
        atom_types = [mo_type, s_type, se_type]
        self._validate_multi_species_coefficients(atom_types, allowed_bonds, "MoSSeNT")

        xyz_filename = self._build_mosse_nanotube(
            n=n, m=m, Ncells=Ncells,
            lattice_constant=3.2300,
            mo_s_bond_length=2.4100,
            mo_se_bond_length=2.5200,
            outer_chalcogen=outer_chalcogen,
            deduplicate=deduplicate,
            PBC=PBC,
            mo_atom_type=mo_type,
            s_atom_type=s_type,
            se_atom_type=se_type,
        )
        bonds_data = self.force_field_manager._inject_xyz_atoms_and_bonds(
            xyz_filename=xyz_filename,
            element_to_atom_type={
                "Mo": mo_type,
                "S": s_type,
                "Se": se_type,
            },
            allowed_bonds=allowed_bonds,
            type_charges={
                entry["atom_type"]: entry["atom_charge"]
                for entry in self.config.default_species_chain
            },
        )
        self._inject_topology_from_bonds(bonds_data)
        return xyz_filename, bonds_data


class Functional_Group_Generator:

    def __init__(self,
                 force_field_manager: Force_Field_Manager,
                 file_manager: File_Manager,
                 config: Config,
                 geometry: Geometry) -> None:

        self.force_field_manager : Force_Field_Manager = force_field_manager
        self.file_manager : File_Manager = file_manager
        self.config : Config = config
        self.gt : Geometry = geometry

    def _get_points_of_attachment(self, structure_atoms, indexes, bonds, fg_config):

        '''
        Given the coordinates, indices, and bond network of the nanotube structure, as well as the functionalisation configuration, determine the optimal points of attachment for
        functional groups according to the specified rules for terminal groups, rings, and loose groups.
        Returns arrays of the selected atom indices, the corresponding normal vectors for group placement, the atom positions, and the assigned functional group type labels
        
        Parameters:
        ----------
        structure_atoms : (N,3) np.ndarray
            Cartesian coordinates of the nanotube atoms.
        indexes : list of int
            True atom IDs corresponding to the rows of structure_atoms.
        bonds : dict
            Mapping from atom ID to list of bonded atom IDs, derived from the bond network.
        fg_config : dict
            Functionalisation configuration dictionary specifying the types and counts of groups to place, 
            as well as any placement rules (e.g. ring phase or terminal side) for each group type.

        Returns:
        -------
        all_indexes : (M,1) np.ndarray of int
            Atom IDs of the selected attachment points for functional groups.
        all_vectors : (M,3) np.ndarray of float
            Normal vectors at the attachment points, indicating the direction for group placement.
        all_positions : (M,3) np.ndarray of float
            Cartesian coordinates of the selected attachment points.
        all_types : list of str
            Functional group type labels assigned to each attachment point, corresponding to the group types specified in fg_config.
        
        Raises:
        ------
        ValueError
            If the functionalisation configuration cannot be satisfied with the available attachment points (e.g. not enough end carbons for terminal groups).
        
        '''

        structure_atoms = np.array(structure_atoms)
        indexes         = np.array(indexes)

        all_indexes   = np.empty((0, 1))
        all_vectors   = np.empty((0, 3))
        all_positions = np.empty((0, 3))
        all_types     = []

        occupied = set()  # ALWAYS atom IDs 

        # ------------------------------------------------------------------
        # helpers
        # ------------------------------------------------------------------

        def _needed_min_bonds():
            if self.config.drude_polarisable:
                return 4, 2
            return 3, 1

        def _end_carbons():
            """Return ARRAY INDICES of under-coordinated end carbons."""
            needed, min_b = _needed_min_bonds()
            entry, exit_ = [], []

            for i, atom_id in enumerate(indexes):

                if atom_id in occupied:
                    continue

                count, _ = self.force_field_manager._get_index_bonds(bonds, atom_id)
                if min_b <= count < needed:
                    if structure_atoms[i, 2] < 0:
                        entry.append(i)   # array index
                    else:
                        exit_.append(i)

            return entry, exit_

        def _commit(array_i, vector, type_label):
            """Store using array index, track occupancy using atom ID."""
            nonlocal all_indexes, all_vectors, all_positions

            atom_id = int(indexes[array_i])

            all_indexes   = np.vstack([all_indexes, atom_id])
            all_vectors   = np.vstack([all_vectors, vector])
            all_positions = np.vstack([all_positions, structure_atoms[array_i]])
            all_types.append(type_label)

            occupied.add(atom_id)

        # ==================================================================
        # 1. TERM
        # ==================================================================

        term_cfg = fg_config.get('term', {}) or {}
        term_hydrogenate = bool(term_cfg.get('term-hydrogenate', False))
        term_start_highest_x = term_cfg.get('term-start-highest-x', False)
        if not isinstance(term_start_highest_x, bool):
            raise ValueError("term-start-highest-x must be a boolean.")
        dummy_term_ids = set()

        entry_seq, exit_seq = [], []
        for g in term_cfg.get('groups', []):
            count = int(g.get('count', 0))
            if count <= 0:
                continue

            unsupported = set(g) - TERM_GROUP_FIELDS
            if unsupported:
                raise ValueError(
                    f"Unsupported term group field(s): {', '.join(sorted(unsupported))}."
                )

            side = g.get('side', 'both')
            if side not in TERM_SIDES:
                raise ValueError(
                    f"Invalid term side {side!r}; expected 'both', '-', or '+'."
                )
            on_entry, on_exit = TERM_SIDES[side]

            if on_entry:
                entry_seq.extend([g['type']] * count)
            if on_exit:
                exit_seq.extend([g['type']] * count)

        for side, seq, z_vec in (('entry', entry_seq, [0, 0, -1]),
                                 ('exit',   exit_seq, [0, 0,  1])):
            if not seq: continue

            entry_c, exit_c = _end_carbons()
            pool = entry_c if side == 'entry' else exit_c  # ARRAY INDICES

            if len(seq) > len(pool):
                my_log_file.error(f"Not enough {side}-end carbons: need {len(seq)}, have {len(pool)}")
                raise ValueError("Not enough end carbons.")

            xs, ys = structure_atoms[pool, 0], structure_atoms[pool, 1]
            if term_start_highest_x:
                max_x = np.max(xs)
                start_index = max(
                    np.flatnonzero(xs == max_x),
                    key=lambda i: ys[i],
                )
                picks = self.gt._get_circle_points_angle_greedy(
                    xs, ys, len(seq), start_index=start_index
                )
            else:
                picks = self.gt._get_circle_points_angle_greedy(xs, ys, len(seq))
            vec = np.array(z_vec, dtype=float)
            for pick_local, type_label in zip(picks, seq):
                array_i = pool[pick_local]
                _commit(array_i, vec, type_label)
                if type_label == DUMMY_FG_TYPE:
                    dummy_term_ids.add(int(indexes[array_i]))

        # ==================================================================
        # 2. RINGS
        # ==================================================================

        for ring in fg_config.get('rings', []) or []:

            axial_rings = int(ring.get('ring-count', 0))
            if axial_rings <= 0:
                continue

            per_ring_seq = []
            for g in ring.get('groups', []):
                per_ring_seq.extend([g['type']] * int(g.get('count', 0)))

            sites_per_ring = len(per_ring_seq)
            if sites_per_ring == 0:
                continue

            ring_idx, ring_vec, ring_pos, occupied = self.gt._find_discrete_cylinder_points(
                structure_atoms, indexes, occupied,
                rings_count             = axial_rings,
                per_ring_count          = sites_per_ring,
                per_ring_phase_step     = float(ring.get('ring-phase-increment', 0.0)),
                ring_phase_start_offset = float(ring.get('ring-phase-start-offset', 0.0)),
                in_or_ex                = ring.get('ring-placement', 'external'),
                padding                 = bool(ring.get('ring-padding', True)),
            )

            all_indexes   = np.vstack([all_indexes, ring_idx])
            all_vectors   = np.vstack([all_vectors, ring_vec])
            all_positions = np.vstack([all_positions, ring_pos])
            all_types.extend(per_ring_seq * axial_rings)

        # ==================================================================
        # 3. LOOSE
        # ==================================================================

        # t runs from 0 at the entry end to 1 at the exit end
        z_vals = structure_atoms[:, 2]
        z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
        z_span = z_max - z_min

        def _normalised_z(array_i):
            if z_span < 1e-12:
                return 0.5
            return (float(structure_atoms[array_i, 2]) - z_min) / z_span

        def _in_window(array_i, z_from, z_to):
            t = _normalised_z(array_i)
            if z_to >= 1.0:  # last band is inclusive of the tube end
                return z_from <= t <= 1.0
            return z_from <= t < z_to

        def _radial_vec(pos, inward=False):
            vec = np.array([pos[0], pos[1], 0.0])
            if inward:
                vec = -vec
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])

        def _is_rim_carbon(array_i):
            """Under-coordinated end carbon, reserved for term/hydrogenation."""
            needed, min_b = _needed_min_bonds()
            count, _ = self.force_field_manager._get_index_bonds(bonds, int(indexes[array_i]))
            return min_b <= count < needed

        for g in (fg_config.get('loose', {}) or {}).get('groups', []) or []:

            mode   = g.get('loose-placement', 'random')
            n      = int(g.get('count', 0))
            label  = g['type']
            z_from = float(g.get('z-from', 0.0))
            z_to   = float(g.get('z-to', 1.0))
            if not (0.0 <= z_from < z_to <= 1.0):
                raise ValueError(
                    f"Loose group '{label}': z-from/z-to must satisfy "
                    f"0 <= z-from < z-to <= 1 (got {z_from}, {z_to})."
                )

            match mode:

                case 'random-external' | 'random-internal':
                    inward = mode == 'random-internal'
                    candidates = [i for i in range(len(indexes))
                                  if _in_window(i, z_from, z_to)]
                    if n > 0 and not candidates:
                        raise ValueError("No free atom found.")
                    for _ in range(n):
                        attempts = 0
                        while True:
                            attempts += 1
                            if attempts > 10_000:
                                raise ValueError("No free atom found.")
                            array_i = candidates[np.random.randint(0, len(candidates))]
                            if int(indexes[array_i]) in occupied:
                                continue
                            break
                        vec = _radial_vec(structure_atoms[array_i], inward=inward)
                        _commit(array_i, vec, label)

                case 'all-external' | 'all-internal':
                    if n > 0:
                        my_log_file.info(
                            f"Loose group '{label}': 'count' is ignored for "
                            f"'{mode}' placement (all free atoms in the z-window are used)."
                        )
                    inward = mode == 'all-internal'
                    for array_i in range(len(indexes)):
                        if int(indexes[array_i]) in occupied:
                            continue
                        if _is_rim_carbon(array_i):
                            continue
                        if not _in_window(array_i, z_from, z_to):
                            continue
                        vec = _radial_vec(structure_atoms[array_i], inward=inward)
                        _commit(array_i, vec, label)

                case _:
                    my_log_file.warning(f"Unsupported loose mode: {mode}")

        # ==================================================================
        # 4. HYDROGENATION
        # ==================================================================

        if term_hydrogenate:
            occupied.difference_update(dummy_term_ids)
            entry_c, exit_c = _end_carbons()
            if entry_c or exit_c:
                for i in entry_c:
                    _commit(i, np.array([0, 0, -1], dtype=float), 'H-term')
                for i in exit_c:
                    _commit(i, np.array([0, 0, 1], dtype=float), 'H-term')
            else:
                my_log_file.info("No additional hydrogens can be added.")

        all_types = np.array(all_types, dtype='<U32')
        visible = all_types != DUMMY_FG_TYPE
        return (
            all_indexes[visible],
            all_vectors[visible],
            all_positions[visible],
            all_types[visible]
        )

    def write_functional_groups(self, active_group_file, all_indexes, all_vectors, all_positions, all_types, residue_idx="42"):

        '''
        Dispatcher for functional group attachment. Routes to the appropriate
        builder based on attachment topology. Currently always routes to single-point.

        Parameters:
        ----------
        active_group_file : str
            Path to the fragments xdata file.
        all_indexes : (M,1) np.ndarray
            LAMMPS atom indices of the M attachment carbons.
        all_vectors : (M,3) np.ndarray
            Build direction unit vectors for each attachment site.
        all_positions : (M,3) np.ndarray
            Cartesian positions of the attachment carbons.
        all_types : list of str
            Functional group type string for each site (e.g. 'COO-', 'H+').
        residue_idx : str, optional
            Residue index to assign all injected atoms (default '42').

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        if np.any(np.asarray(all_types) == DUMMY_FG_TYPE):
            raise ValueError(
                "DUMMY is placement-only and must be removed before topology writing."
            )

        # future: inspect group topology here to route to multi-point if needed
        self._build_single_point_fg(active_group_file, all_indexes, all_vectors, all_positions, all_types, residue_idx)


    def _build_single_point_fg(self, active_group_file, all_indexes, all_vectors, all_positions, all_types, residue_idx="42"):

        '''
        Attach single-point functional groups to the nanotube one by one. For
        each attachment site, injects the group's atom-type coefficients if not
        already present, then writes its atom and bond lines. Only supports groups
        with a single carbon attachment point.

        Parameters:
        ----------
        active_group_file : str
            Path to the fragments xdata file containing functional group definitions.
        all_indexes : (M,1) np.ndarray
            LAMMPS atom indices of the M attachment carbons.
        all_vectors : (M,3) np.ndarray
            Build direction unit vectors for each attachment site.
        all_positions : (M,3) np.ndarray
            Cartesian positions of the attachment carbons.
        all_types : list of str
            Functional group type string for each site (e.g. 'COO-', 'H+').
        residue_idx : str, optional
            Residue index to assign all injected atoms (default '42').

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        ready_fg_types = []

        for i in range(len(all_indexes)):
            index    = all_indexes[i]
            vector   = all_vectors[i,:]
            position = all_positions[i,:]
            fg_type  = all_types[i]

            if fg_type not in ready_fg_types:
                ready_fg_types.append(fg_type)
                molecule = self.file_manager.find_and_extract_subsection(filename=active_group_file, structure_type='Fragment', structure=fg_type)
                self.force_field_manager.inject_atom_priorities(molecule, priority_level=10)
                self.force_field_manager.inject_atom_coefficients(molecule)

            self._inject_single_point_fg(active_group_file, fg_type, index, vector, position, residue_idx=residue_idx)

        e = self.force_field_manager.get_system_charge(residue_index=residue_idx)
        my_log_file.info(f"Functional group generation complete: {len(all_indexes)} groups attached. Nanostructure charge is now {self.force_field_manager._round_charge(e)} e.")


    def _build_multi_point_fg(self, active_group_file, all_indexes, all_vectors, all_positions, all_types, residue_idx="42"):

        '''
        Placeholder for multi-point functional group attachment (e.g. crosslinkers
        bridging two or more carbons). Requires anchor pair selection, shared
        build-vector resolution, and multi-bond injection.

        Parameters:
        ----------
        active_group_file : str
            Path to the fragments xdata file.
        all_indexes : (M,1) np.ndarray
            LAMMPS atom indices of the M attachment carbons.
        all_vectors : (M,3) np.ndarray
            Build direction unit vectors for each attachment site.
        all_positions : (M,3) np.ndarray
            Cartesian positions of the attachment carbons.
        all_types : list of str
            Functional group type string for each site.
        residue_idx : str, optional
            Residue index to assign all injected atoms (default '42').

        Returns:
        -------
        None

        Raises:
        ------
        NotImplementedError
            Always; multi-point attachment is not yet implemented.
        '''

        raise NotImplementedError(
            "Multi-point functional group attachment is not yet implemented."
        )


    def _inject_single_point_fg(self, active_group_file, functional_group_type, 
                                carbon_index, carbon_vector, carbon_position, residue_idx="42"):


        '''
        Place a single functional group on the nanotube: read its template from
        the fragments file, rotate and translate all atoms to the attachment site,
        add a small random perturbation to avoid atom coplanarity, and inject the
        resulting Atom, Bond, Lone Pair, and Anisotropy lines into the main file.

        Parameters:
        ----------
        active_group_file : str
            Path to the fragments xdata file.
        functional_group_type : str
            Name of the functional group to place (e.g. 'COO-', 'OH-').
        carbon_index : int
            LAMMPS index of the attachment carbon on the nanotube.
        carbon_vector : (3,) np.ndarray
            Direction in which to build the group (outward normal for external,
            axial unit vector for end groups).
        carbon_position : (3,) np.ndarray
            Cartesian position of the attachment carbon.
        residue_idx : str, optional
            Residue index for all injected atoms (default '42').

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        fg_lines = self.file_manager.find_and_extract_subsection(filename=active_group_file, 
                                                                 structure_type='Fragment',
                                                                 structure=functional_group_type)
        if self.config.verbose:
            my_log_file.debug(f"Functional group lines for {functional_group_type}: {fg_lines}")

        position_lines = self.file_manager.find_and_extract_lines(fg_lines, "Atoms", header=False) # find the correct fragments
        position_matrix = np.ones((len(position_lines), 3)) 

        all_parts = []
        for position_idx, line in enumerate(position_lines):  
            parts = line.strip("\n").split()
            all_parts.append(parts)
            position_matrix[position_idx, 0] = float(parts[4]) 
            position_matrix[position_idx, 1] = float(parts[5]) 
            position_matrix[position_idx, 2] = float(parts[6]) 

        prerotated_position_matrix = position_matrix # normalised
        rotated_position_matrix = self.gt.rotate_points_simple(prerotated_position_matrix, carbon_vector) # rotate

        translated_position_matrix = np.zeros_like(rotated_position_matrix) # translate
        for i in range(rotated_position_matrix.shape[0]):
            translated_position_matrix[i,:] = rotated_position_matrix[i,:] + carbon_position

        rotated_forwarded_matrix = np.zeros_like(rotated_position_matrix) # push off so there is space for bonds
        for ii in range(rotated_position_matrix.shape[0]):
            rotated_forwarded_matrix[ii,:] = translated_position_matrix[ii,:] + carbon_vector * 1.5

        for iii in range(rotated_forwarded_matrix.shape[0]): # add small random perturbation to avoid coplanarity of atoms for dihedrals and impropers
            perturbation = np.random.normal(0, 0.005, 3)  # small random perturbation
            rotated_forwarded_matrix[iii,:] += perturbation

        # atoms
        main_lines = self.file_manager.read_file(self.config.main_file) # build ATOM LINES
        atom_idx_offset = self.file_manager.find_last_section_idx(main_lines, "Atoms")
        atom_injection = self.file_manager.find_injection_point(main_lines, "Atoms")
        for iv in range(len(position_lines)):
            line_array = []
            line_array.append(str(int(atom_idx_offset) + int(all_parts[iv][0])))     # new atom index
            line_array.append(str(int(residue_idx)))                                # residue index
            line_array.append(str(self.force_field_manager.get_new_atom_type(
                all_parts[iv][8],
                source_atom_number_type=all_parts[iv][2],
                source_lines=fg_lines,
            )))                                                                       # main file atom type
            line_array.append(str(float(all_parts[iv][3])))                          # charge
            line_array.append(f"{rotated_forwarded_matrix[iv, 0]:.10f}")             # x
            line_array.append(f"{rotated_forwarded_matrix[iv, 1]:.10f}")             # y
            line_array.append(f"{rotated_forwarded_matrix[iv, 2]:.10f}")             # z     
            line_array.append(f"#")                                                 # cant foget the # bless up
            line_array.append(all_parts[iv][8])                                      # old atom type
            new_line = self.file_manager.standardise_line(line_array, "Atoms")
            main_lines.insert(atom_injection + iv, new_line)

        enabled = self.config.enabled_xdata_sections

        # Only load bonds if we write them or need them for higher-order topology.
        if enabled & {"Bonds", "Angles", "Dihedrals", "Impropers"}:
            bond_lines = self.file_manager.find_and_extract_lines(fg_lines, "Bonds", header=False)
            all_parts = [line.strip("\n").split() for line in bond_lines]
            write_bonds = "Bonds" in enabled
            bond_idx_offset = (
                self.file_manager.find_last_section_idx(main_lines, "Bonds")
                if write_bonds else len(self.force_field_manager.generated_bonds_data)
            )
            if write_bonds:
                bond_injection = self.file_manager.find_injection_point(main_lines, "Bonds")

            for v, bond in enumerate(all_parts):
                atom1 = int(carbon_index[0]) if bond[2] == "c-idx" else int(bond[2]) + int(atom_idx_offset)
                atom2 = int(carbon_index[0]) if bond[3] == "c-idx" else int(bond[3]) + int(atom_idx_offset)
                bond_type = ""
                new_bond_index = int(bond_idx_offset) + int(bond[0])

                if write_bonds:
                    bond_type = self.force_field_manager.get_new_bond_type(bond[5], bond[6])
                    line_array = [
                        str(new_bond_index), str(bond_type), str(atom1), str(atom2),
                        "#", bond[5], bond[6],
                    ]
                    new_line = self.file_manager.standardise_line(line_array, "Bonds")
                    main_lines.insert(bond_injection + v, new_line)

                self.force_field_manager.generated_bonds_data.append([
                    str(new_bond_index), str(bond_type), str(atom1), str(atom2),
                    bond[5], bond[6],
                ])

        if "Lone Pairs" in enabled:
            lp_mol_lines = self.file_manager.find_and_extract_lines(fg_lines, "Lone Pairs", header=False)
            lp_idx_offset = self.file_manager.find_last_section_idx(main_lines, "Lone Pairs")
            lp_injection = self.file_manager.find_injection_point(main_lines, "Lone Pairs")

            for vi, lone_pair_line in enumerate(lp_mol_lines):
                lone_pair = self.file_manager.line_to_array(lone_pair_line)
                dependencies = [
                    int(carbon_index[0]) if item == "c-idx" else int(item) + atom_idx_offset
                    for item in lone_pair[-1].strip("()").split(",")
                ]
                line_array = [
                    lp_idx_offset + vi + 1,
                    atom_idx_offset + int(lone_pair[1]),
                    *lone_pair[2:8],
                    f"({','.join(map(str, dependencies))})",
                ]
                new_line = self.file_manager.standardise_line(line_array, "Lone Pairs")
                main_lines.insert(lp_injection + vi, new_line)

        if "Anisotropy" in enabled:
            aniso_mol_lines = self.file_manager.find_and_extract_lines(fg_lines, "Anisotropy", header=False)
            aniso_idx_offset = self.file_manager.find_last_section_idx(main_lines, "Anisotropy")
            aniso_injection = self.file_manager.find_injection_point(main_lines, "Anisotropy")

            for vii, aniso_line in enumerate(aniso_mol_lines):
                aniso = self.file_manager.line_to_array(aniso_line)
                dependencies = [
                    int(item) + atom_idx_offset
                    for item in aniso[-1].strip("()").split(",")
                ]
                line_array = [
                    aniso_idx_offset + vii + 1,
                    atom_idx_offset + int(aniso[1]),
                    *aniso[2:6],
                    f"({','.join(map(str, dependencies))})",
                ]
                new_line = self.file_manager.standardise_line(line_array, "Anisotropy")
                main_lines.insert(aniso_injection + vii, new_line)

        self.file_manager.write_file(main_lines, self.config.main_file)


class Solution_Generator:

    def __init__(self, 
                 force_field_manager: Force_Field_Manager,
                 file_manager: File_Manager, 
                 config: Config,
                 geometry: Geometry) -> None:

        self.force_field_manager : Force_Field_Manager = force_field_manager
        self.file_manager : File_Manager = file_manager
        self.config : Config = config
        self.gt : Geometry = geometry


    # -------------------------------------------------------------------------
    # REMOVAL OF OVERLAPS
    # -------------------------------------------------------------------------

    def _remove_overlaps_IN_MEMORY(self, atom_types, atom_indices, atom_positions, atom_residues, 
                                   min_distance=1.4, new_residues_from=None, use_float32=True):


        '''
        Vectorised overlap removal using a broadcast distance-squared matrix.

        Rows of the matrix are every atom outside the bulk phase, meaning priority not
        equal to config.bulk_priority, which validation derived as the priority of
        the single packed group forming the bulk. Columns are every atom in the
        system, so a pair is examined whenever either side is a row, and the only
        pairs that escape are bulk-vs-bulk. That exemption is the whole point:
        bulk spacing comes from Bridson's centre-to-centre radius, not from
        min_distance, so policing it here would cull a large share of the box.

        Matching on the bulk priority rather than testing against a threshold also
        keeps a species numbered ABOVE the bulk in the check. A "priority < cutoff"
        test silently dropped those; an ion group at 150 against bulk water at 100
        was compared with nothing at all.

        min_distance belongs to the group currently being injected, so it may only
        be applied to pairs that group is part of. new_residues_from carries the
        residue high-water mark from before the injection and restricts the matrix
        to pairs with at least one atom above it. Without that restriction a later
        group's larger min_distance is applied retroactively to pairs an earlier
        injection already settled. A nanotube and the water around it, for example, get
        re-adjudicated at a threshold that was never theirs, and water that was
        correctly placed at 1.8 Angstrom is deleted by a 2.5 Angstrom salt group it never touched.

        Equal-priority overlaps are broken deterministically by program index: the
        newcomer yields to what is already placed. All atoms of an overlapping
        residue are removed together.

        Parameters:
        ----------
        atom_types : list of str
            Atom type names for all atoms.
        atom_indices : list of int
            LAMMPS atom indices for all atoms.
        atom_positions : (N,3) np.ndarray
            Cartesian coordinates of all atoms.
        atom_residues : list of int or str
            Residue index for each atom.
        min_distance : float, optional
            Overlap distance threshold in Angstroms (default 1.4).
        new_residues_from : int, optional
            Highest residue index present before the current injection. Only pairs
            with at least one atom in a residue above this are examined. None
            examines every pair, which re-adjudicates already-settled ones at this
            call's min_distance and is almost never what a caller wants
            (default None).
        use_float32 : bool, optional
            If True, cast positions to float32 before computing the distance
            matrix to reduce RAM usage (default True).

        Returns:
        -------
        None

        Raises:
        ------
        None
        ''' 

        N = len(atom_types)
        if N == 0:
            return

        priorities = np.array([self._get_atom_priority(t) for t in atom_types], dtype=int) #create priority array

        res_all = np.asarray(atom_residues)              # (N,)

        # Check any pair touching a non-bulk atom; None means there is no bulk exemption.
        bulk_priority = getattr(self.config, 'bulk_priority', None)
        if bulk_priority is None:
            cand_rows = np.arange(N)
        else:
            cand_rows = np.nonzero(priorities != bulk_priority)[0]

        if cand_rows.size == 0:
            my_log_file.info(
                f"Every atom belongs to the bulk phase (priority {bulk_priority}); "
                f"nothing to check.")
            return

        pos_all = atom_positions.astype(np.float32 if use_float32 else np.float64, copy=False)   # (N,3)
        pos_c   = pos_all[cand_rows]                                                             # (M,3)

        res_c   = res_all[cand_rows]                     # (M,)

        p_all   = priorities                             # (N,)
        p_c     = priorities[cand_rows]                  # (M,)

        prog_all = np.asarray(atom_indices)     # (N,) program indices
        prog_c   = prog_all[cand_rows]          # (M,)

        # Distance squared matrix (M,N) via ||a||^2 + ||b||^2 - 2 a dot b
        a2 = np.einsum('ij,ij->i', pos_c, pos_c)         # (M,)
        b2 = np.einsum('ij,ij->i', pos_all, pos_all)     # (N,)
        G  = pos_c @ pos_all.T                           # (M,N)
        d2 = a2[:, None] + b2[None, :] - 2.0 * G         # (M,N)
        cutoff2 = float(min_distance) * float(min_distance)

        # Masks: different atom (by PROGRAM index) and different residue (by array rows)
        not_same_atom = (prog_c[:, None] != prog_all[None, :])          # (M,N) bool
        not_same_res  = (res_c[:, None]  != res_all[None, :])           # (M,N) bool
        overlaps = (d2 < cutoff2) & not_same_atom & not_same_res

        # Only recheck pairs from this injection; older high-priority rows can still evict newcomers.
        if new_residues_from is not None:
            is_new_all = (res_all > new_residues_from)                  # (N,)
            is_new_c   = is_new_all[cand_rows]                          # (M,)
            overlaps &= (is_new_c[:, None] | is_new_all[None, :])

        if not overlaps.any():
            my_log_file.info("No overlaps found for the injected group vs all.")
            return

        # Decisions (broadcast on (M,N))
        p1 = p_c[:, None]        # candidate priorities
        p2 = p_all[None, :]      # all priorities

        remove_j = overlaps & (p1 < p2)                 # candidate beats j -> remove j (columns)
        remove_i = overlaps & (p2 < p1)      # j beats candidate -> remove i (rows)
        ties     = overlaps & (p1 == p2)                # tie -> resolved by index

        # On equal priority, the later atom loses. Random coin flips broke reproducibility here.
        later_is_j = (prog_c[:, None] < prog_all[None, :])   # j came later -> remove j
        tie_remove_j = ties & later_is_j
        tie_remove_i = ties & (~later_is_j)

        to_remove_rows = set() # collect rows to remove (in array-row space)

        # Columns j to remove (already in array-row space)
        if remove_j.any():
            to_remove_rows.update(np.nonzero(remove_j)[1].tolist())
        if tie_remove_j.any():
            to_remove_rows.update(np.nonzero(tie_remove_j)[1].tolist())

        # Rows i to remove: map from candidate row indices to array rows
        if remove_i.any():
            i_rows = np.nonzero(remove_i)[0]
            to_remove_rows.update(cand_rows[i_rows].tolist())
        if tie_remove_i.any():
            i_rows = np.nonzero(tie_remove_i)[0]
            to_remove_rows.update(cand_rows[i_rows].tolist())

        if not to_remove_rows:
            my_log_file.info("No atoms selected for removal after priority resolution.")
            return

        # Expand to whole residues (FFM expects PROGRAM indices)
        chopping_block = set()
        for row in to_remove_rows:
            residue_id = res_all[row]
            # get_atoms_of_residue should return PROGRAM indices
            for culprit in self.force_field_manager.get_atoms_of_residue(residue_id):
                chopping_block.add(culprit)

        to_remove_prog = sorted(prog_all[list(to_remove_rows)].tolist())

        n_removed = len(chopping_block)

        my_log_file.warning(f"Removing {n_removed} atoms due to overlap.")

        if self.config.verbose:
            my_log_file.info(f"Overlapping atoms (pre-residue expansion, program indices): {to_remove_prog}")
            my_log_file.info(f"Atoms to obliterate (expanded by residue): {sorted(chopping_block)}")

        self.force_field_manager.obliterate_atoms(chopping_block)

    # -------------------------------------------------------------------------
    # SIDECAR RECORDS (.nanotube / .salts)
    # -------------------------------------------------------------------------
    # .nanotube holds the structure and groups before solvation.
    # .salts holds ions after each cull so later salts can see them.
    # Both are rebuilt each run. Format: index type residue x y z

    def _record_line(self, index, atom_type, residue, position):

        '''
        Format a single sidecar record line.

        Parameters:
        ----------
        index : int
            Atom index.
        atom_type : str
            Atom type name.
        residue : int
            Residue index.
        position : array-like of shape (3,)
            Cartesian coordinates.

        Returns:
        -------
        line : str
            Formatted record line, newline terminated.

        Raises:
        ------
        None
        '''

        return (f"{int(index)} {atom_type} {int(residue)} "
                f"{float(position[0]):.6f} {float(position[1]):.6f} {float(position[2]):.6f}\n")

    def _read_record_file(self, path):

        '''
        Read a sidecar record file into parallel lists. A missing file is not an
        error; it means the run has no structure, or no ions have been placed yet.

        Parameters:
        ----------
        path : str or None
            Path to the .nanotube or .salts file.

        Returns:
        -------
        positions : (N,3) np.ndarray of float64
            Recorded coordinates, in file order. Empty (0,3) if the file is absent.
        types : list of str
            Atom type name for each record.
        residues : list of int
            Residue index for each record.

        Raises:
        ------
        None
        '''

        if not path or not os.path.isfile(path):
            return np.empty((0, 3)), [], []

        positions, types, residues = [], [], []
        for line in self.file_manager.read_file(path):
            parts = line.split()
            if len(parts) < 6:
                continue
            types.append(parts[1])
            residues.append(int(parts[2]))
            positions.append([float(parts[3]), float(parts[4]), float(parts[5])])

        if not positions:
            return np.empty((0, 3)), [], []

        return np.array(positions, dtype=np.float64), types, residues

    def write_structure_record(self):

        '''
        Write the .nanotube sidecar from the current contents of the main file.
        Call once, after structure generation and functionalisation have completed
        and before any solvent is injected. At that point the main file holds the
        covalent structure and nothing else.

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

        path = getattr(self.config, 'structure_record_file', None)
        if not path:
            return

        positions, indexes, types, residues = self.force_field_manager.get_atom_positions_all(self.config.main_file)

        if len(indexes) == 0:
            # Bare electrolyte box: leave the structure sidecar absent.
            my_log_file.info("Structure record: no structure atoms present, no .nanotube written.")
            return

        lines = [self._record_line(indexes[i], types[i], residues[i], positions[i, :])
                 for i in range(len(indexes))]

        self.file_manager.write_file(lines, path)
        my_log_file.info(f"Structure record: wrote {len(lines)} atoms to {os.path.basename(path)}.")

    def update_salt_record(self, new_residues_from):

        '''
        Rebuild the .salts sidecar after an ionic group has been injected and the
        overlap cull has run.

        Rebuilt rather than appended so the record self-heals: if a later group's
        cull removed an ion recorded by an earlier group, that ion drops out here
        instead of leaving a phantom exclusion zone behind.

        One record per distinct ion position. Deduplicating on position rather than
        on residue is deliberate and load-bearing:

          - For a Drude force field an ion residue carries a coincident Drude
            particle at the identical coordinate. Position dedup drops it, so the
            record count equals the ion count and no redundant exclusion is stored.

        Parameters:
        ----------
        new_residues_from : int
            Highest residue index present before this ionic group was injected.
            Residues above this belong to the group just placed.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        path = getattr(self.config, 'salts_record_file', None)
        if not path:
            return

        _, _, previous_residues = self._read_record_file(path)
        ion_residues = set(previous_residues)

        positions, indexes, types, residues = self.force_field_manager.get_atom_positions_all(self.config.main_file)

        for res in residues:
            if res > new_residues_from:
                ion_residues.add(res)

        lines = []
        seen = set()
        for i, res in enumerate(residues):
            if res not in ion_residues:
                continue
            key = (round(float(positions[i, 0]), 6),
                   round(float(positions[i, 1]), 6),
                   round(float(positions[i, 2]), 6))
            if key in seen:
                continue
            seen.add(key)
            lines.append(self._record_line(indexes[i], types[i], res, positions[i, :]))

        self.file_manager.write_file(lines, path)
        my_log_file.info(f"Salt record: {len(lines)} ions recorded in {os.path.basename(path)}.")

    def _get_atom_priority(self, atom_type1):

        '''
        Return the priority level assigned to an atom type in
        config.atom_type_priority. Lower integer values mean higher priority
        (priority 0 is the nanotube carbon). Raises ValueError if the atom type
        is not registered in the priority map.

        Parameters:
        ----------
        atom_type1 : str
            Atom type name to look up.

        Returns:
        -------
        priority : int
            Priority level of the atom type.

        Raises:
        ------
        ValueError
            If atom_type1 is not found in config.atom_type_priority.
        '''

        p1 = -1

        for priority_list in self.config.atom_type_priority:

            if atom_type1 in self.config.atom_type_priority[priority_list]:
                p1 = int(priority_list)

        if p1 == -1:
            my_log_file.error(f"Atom type {atom_type1} not found in priority list.")
            raise ValueError(f"Atom type {atom_type1}  not found in priority list.")

        return p1

    def _process_salt_formula(self, formula):

        '''
        Parse a chemical salt formula string into its cation and anion components
        and their stoichiometric counts (e.g. 'KCl' -> Kx1, Clx1;
        'Ca2Cl2' -> Cax2, Clx2).

        Parameters:
        ----------
        formula : str
            Salt formula string (e.g. 'NaCl', 'KCl', 'Ca2Cl2').

        Returns:
        -------
        cathode_type : str
            Chemical symbol of the cation.
        cathode_count : int
            Stoichiometric count of the cation per formula unit.
        anode_type : str
            Chemical symbol of the anion.
        anode_count : int
            Stoichiometric count of the anion per formula unit.

        Raises:
        ------
        None
        '''

        def split_formula(s):

            '''

            split formula.

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

            return re.findall(r'[A-Z][a-z]?|\d+', s)
        
        salt = split_formula(formula)

        cathode_type = salt[0]

        if len(salt) == 2: 
            cathode_count = 1
            anode_type = salt[1]
            anode_count = 1

        if len(salt) == 3:
            try:
                cathode_count = int(salt[1])
                anode_type = salt[2]      # e.g. 'K2Cl' -> salt=['K','2','Cl']
                anode_count = 1
            except ValueError:
                # salt[1] is a letter (e.g. 'KCl2' -> salt=['K','Cl','2'])
                cathode_count = 1
                anode_type = salt[1]
                anode_count = int(salt[2])

        if len(salt) == 4:
            cathode_count = int(salt[1])
            anode_type = salt[2]
            anode_count = int(salt[3])

        return cathode_type, cathode_count, anode_type, anode_count

    def _charge_neutrality(self, sys_charge, cathode_count, anode_count, cathode_charge, anode_charge, injection_limit=1024):

        '''
        Find the minimum number of cation and anion pairs to add that drives the
        total system charge as close to zero as possible. Uses a greedy scan
        bounded by injection_limit to avoid infinite loops in edge cases where
        the system is heavily unbalanced.

        Parameters:
        ----------
        sys_charge : float
            Current total charge of the system.
        cathode_count : int
            Stoichiometric ratio of cations per formula unit.
        anode_count : int
            Stoichiometric ratio of anions per formula unit.
        cathode_charge : float
            Partial charge of each cation (positive).
        anode_charge : float
            Partial charge of each anion (negative).
        injection_limit : int, optional
            Maximum number of ion pairs to try before raising an error (default 1024).

        Returns:
        -------
        nC : int
            Number of cation ions to add.
        nA : int
            Number of anion ions to add.
        final : float
            Resulting system charge after adding nC cations and nA anions.

        Raises:
        ------
        ValueError
            If the system charge cannot be balanced within the injection limit.
        '''          

        # injection_limit sourced from bridson_params via caller

        best = None  # (abs_err, total_ions, nC, nA, final)
        for nC in range(int(cathode_count)):
            for nA in range(int(anode_count)):
                final = sys_charge + nC * cathode_charge + nA * anode_charge
                cand = (abs(final), nC + nA, nC, nA, final)
                if best is None or cand < best:
                    best = cand


        if sys_charge > 0:

            for nC in range(int(cathode_count)):
                tend_wrong = False
                previous_best = None

                for nA in range(int(anode_count), injection_limit):

                    if nA == injection_limit - 2:
                        my_log_file.error("System charge heavily off balance.")
                        raise ValueError("System charge heavily off balance.")
                    
                    if tend_wrong:
                        break

                    final = sys_charge + nC * cathode_charge + nA * anode_charge
                    cand = (abs(final), nC + nA, nC, nA, final)

                    if previous_best is None:
                        previous_best = cand
                        continue
                    else:
                        if cand < previous_best:
                            previous_best = cand
                        else:
                            tend_wrong = True
                            if best is None or previous_best < best:
                                best = previous_best
                                continue
        else:

            for nA in range(int(anode_count)):
                tend_wrong = False
                previous_best = None

                for nC in range(int(cathode_count), injection_limit):

                    if nC == injection_limit - 2:
                        my_log_file.error("System charge heavily off balance.")
                        raise ValueError("System charge heavily off balance.")
                    
                    if tend_wrong:
                        break

                    final = sys_charge + nC * cathode_charge + nA * anode_charge
                    cand = (abs(final), nC + nA,nC, nA, final)

                    if previous_best is None:
                        previous_best = cand
                        continue
                    else:
                        if cand < previous_best:
                            previous_best = cand
                        else:
                            tend_wrong = True
                            if best is None or previous_best < best:
                                best = previous_best
                                continue

        abs_err, _, nC, nA, final = best

        return nC, nA, final


    # -------------------------------------------------------------------------
    # BRIDSON'S POISSON-DISK SAMPLING FOR FUNCTIONAL GROUP PLACEMENT
    # -------------------------------------------------------------------------

    def _Bridson_poisson_3d(self, box, r, k=30):

        '''
        Generate a Poisson-disk sample of 3-D points inside a rectangular box
        using Bridson's dart-throwing algorithm. No two sample points are closer
        than r, and the point cloud covers the box as densely as possible.

        Parameters:
        ----------
        box : list of list of float
            Box extents [[xlo, xhi], [ylo, yhi], [zlo, zhi]].
        r : float
            Minimum allowed distance between any two sample points in Angstroms.
        k : int, optional
            Number of candidate points attempted per active sample before it is
            retired. Higher values produce denser packings at the cost of runtime
            (default 30).

        Returns:
        -------
        samples : (N,3) np.ndarray
            Cartesian coordinates of the N sampled points inside the box.

        Raises:
        ------
        None
        '''

        W, H, D = box[0][1] - box[0][0], box[1][1] - box[1][0], box[2][1] - box[2][0]

        cell = r / np.sqrt(3)  # grid cell size
        gw, gh, gd = int(np.ceil(W / cell)), int(np.ceil(H / cell)), int(np.ceil(D / cell))
        grid = -np.ones((gw, gh, gd), dtype=int)

        def grid_index(p): 

            '''
            Map a 3-D point to an integer grid cell index tuple for the
            Bridson acceleration grid.

            Parameters:
            ----------
            p : array-like of shape (3,)
                Cartesian coordinates of the point.

            Returns:
            -------
            cell : tuple of int
                (ix, iy, iz) grid cell indices.

            Raises:
            ------
            None
            '''

            return (int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))

        def in_bounds(p):

            '''
            Check whether a point lies within the sampling box extents [0, W) x [0, H) x [0, D).

            Parameters:
            ----------
            p : array-like of shape (3,)
                Cartesian coordinates of the point in box-local space.

            Returns:
            -------
            inside : bool
                True if the point is within the box, False otherwise.

            Raises:
            ------
            None
            '''

            return (0 <= p[0] < W) and (0 <= p[1] < H) and (0 <= p[2] < D)

        def dist2(a, b): 

            '''
            Compute the squared Euclidean distance between two points, used
            for the Bridson proximity check without incurring a square root.

            Parameters:
            ----------
            a : np.ndarray
                First point.
            b : np.ndarray
                Second point.

            Returns:
            -------
            d2 : float
                Squared Euclidean distance between a and b.

            Raises:
            ------
            None
            '''

            return np.sum((a - b) ** 2)

        samples = []
        active = []

        # first point
        p0 = np.array([np.random.rand()*W, np.random.rand()*H, np.random.rand()*D])
        samples.append(p0); active.append(0)
        grid[grid_index(p0)] = 0
        r2 = r * r

        # neighbor offsets (within 2 cells in each direction)
        neigh = range(-2, 3)
        offsets = [(dx,dy,dz) for dx in neigh for dy in neigh for dz in neigh]

        k = k  # attempts per active point, sourced from bridson_params
        while active:
            idx = np.random.choice(active)
            base = samples[idx]
            found = False
            for _ in range(k):
                # random point in spherical shell [r,2r) open bracket is important
                rho = r * (1 + np.random.rand())
                v = np.random.normal(size=3)
                v /= np.linalg.norm(v)
                cand = base + rho * v
                if not in_bounds(cand): continue
                gi = grid_index(cand)

                ok = True
                for dx,dy,dz in offsets:
                    nx, ny, nz = gi[0]+dx, gi[1]+dy, gi[2]+dz
                    if 0 <= nx < gw and 0 <= ny < gh and 0 <= nz < gd:
                        j = grid[nx, ny, nz]
                        if j != -1 and dist2(cand, samples[j]) < r2:
                            ok = False; break
                if ok:
                    samples.append(cand)
                    new_idx = len(samples)-1
                    grid[gi] = new_idx
                    active.append(new_idx)
                    found = True
                    break
            if not found:
                active.remove(idx)
        
        samples = np.array(samples)
        samples[:,0] += box[0][0]
        samples[:,1] += box[1][0]
        samples[:,2] += box[2][0]

        return samples

    def _bridson_best_count(self, box, r, repeats=2, k=30):

        '''
        Run _Bridson_poisson_3d multiple times with the same r and return the
        maximum sample count achieved, since Bridson's algorithm is stochastic
        and a single run may underperform.

        Parameters:
        ----------
        box : list of list of float
            Box extents [[xlo, xhi], [ylo, yhi], [zlo, zhi]].
        r : float
            Minimum distance between sample points.
        repeats : int, optional
            Number of independent runs (default 2).
        k : int, optional
            Candidate attempts per active point passed to _Bridson_poisson_3d (default 30).

        Returns:
        -------
        best : int
            Highest sample count achieved across all runs.

        Raises:
        ------
        None
        '''

        best = 0
        for _ in range(max(1, int(repeats))):
            best = max(best, len(self._Bridson_poisson_3d(box=box, r=r, k=k)))
        return best

    def _find_bridson_r_for_count(self, box, target_count, r_init, r_floor,
                                repeats=2, bracket_steps=10, bin_steps=10, k=30):
        
        '''
        Binary-search for the largest r such that _Bridson_poisson_3d still
        produces at least target_count points. Starts by bracketing the
        transition from feasible to infeasible r, then refines with binary search.
        Raises ValueError if target_count cannot be achieved even at r_floor.

        Parameters:
        ----------
        box : list of list of float
            Box extents [[xlo, xhi], [ylo, yhi], [zlo, zhi]].
        target_count : int
            Desired minimum number of sample points.
        r_init : float
            Initial guess for r, typically derived from the solvent radius.
        r_floor : float
            Hard lower bound on r to prevent the search from running indefinitely.
        repeats : int, optional
            Bridson runs per r evaluation to reduce variance (default 2).
        bracket_steps : int, optional
            Maximum iterations to find the feasibility boundary (default 10).
        bin_steps : int, optional
            Binary-search refinement steps (default 10).
        k : int, optional
            Candidate attempts per active point passed to _Bridson_poisson_3d (default 30).

        Returns:
        -------
        r_best : float
            Largest r that still yields at least target_count sample points.

        Raises:
        ------
        ValueError
            If target_count cannot be achieved even at r_floor.
        '''

        r = float(r_init)
        my_log_file.info(f"Bridson r-search: target={target_count} molecules, r_init={r:.3f}, r_floor={r_floor:.3f}.")
        n = self._bridson_best_count(box, r, repeats=repeats, k=k)
        my_log_file.info(f"Bridson r-search: initial r={r:.3f} gives n={n}.")

        if n >= target_count:
            lo = r
            hi = r
            my_log_file.info(f"Bridson r-search: r_init fits, expanding upward to find ceiling...")
            for _ in range(bracket_steps):
                hi_try = hi * 1.25
                n_try = self._bridson_best_count(box, hi_try, repeats=repeats, k=k)
                my_log_file.info(f"Bridson r-search: trying hi={hi_try:.3f}, got n={n_try}.")
                if n_try >= target_count:
                    lo = hi_try
                    hi = hi_try
                else:
                    hi = hi_try
                    break
            else:
                my_log_file.info(f"Bridson r-search: bracket complete, returning lo={lo:.3f}.")
                return lo

        else:
            hi = r
            lo = r
            my_log_file.info(f"Bridson r-search: r_init too large, shrinking to find feasible r...")
            for _ in range(bracket_steps):
                lo_try = max(r_floor, lo * 0.8)
                n_try = self._bridson_best_count(box, lo_try, repeats=repeats, k=k)
                my_log_file.info(f"Bridson r-search: trying lo={lo_try:.3f}, got n={n_try}.")
                lo = lo_try
                if n_try >= target_count:
                    break
                if abs(lo - r_floor) < 1e-12:
                    break

            n_floor = self._bridson_best_count(box, lo, repeats=repeats, k=k)
            if n_floor < target_count:
                msg = (f"Cannot fit target_count={target_count} even at r_floor={r_floor}. \n "
                    f"Best achieved was {n_floor}. \n "
                    f"r_floor is the larger of bridson_params.r_floor and the solvation "
                    f"group's min_distance, so either lower min_distance, lower the "
                    f"requested count/concentration, or use a bigger box.")
                my_log_file.error(msg)
                raise ValueError(msg)

        best = lo
        my_log_file.info(f"Bridson r-search: bracket found lo={lo:.3f}, hi={hi:.3f}. Refining with binary search ({bin_steps} steps)...")
        for i in range(bin_steps):
            mid = 0.5 * (lo + hi)
            n_mid = self._bridson_best_count(box, mid, repeats=repeats, k=k)
            my_log_file.info(f"Bridson r-search: binary step {i+1}/{bin_steps}; mid={mid:.3f}, n={n_mid}.")
            if n_mid >= target_count:
                best = mid
                lo = mid
            else:
                hi = mid

        my_log_file.info(f"Bridson r-search: complete. Best r={best:.3f} for target={target_count}.")
        return best
    
    # VOLUME HELP

    def _box_volume_ang3(self, max_box_size):

        '''
        Calculate the volume of a rectangular box in cubic Angstroms.

        Parameters:
        ----------
        max_box_size : list of list of float
            Box extents [[xlo, xhi], [ylo, yhi], [zlo, zhi]].

        Returns:
        -------
        volume : float
            Box volume in Angstrom^3.

        Raises:
        ------
        None
        '''

        (x0, x1), (y0, y1), (z0, z1) = max_box_size 

        # Angstrom^3 is the standard unit for volume in molecular simulations
        return (x1 - x0) * (y1 - y0) * (z1 - z0)

    def _ang3_to_liters(self, v_ang3):

        '''
        Convert a volume from cubic Angstroms to litres using the exact conversion
        factor (1 Angstrom^3 = 1x10^-27 L).

        Parameters:
        ----------
        v_ang3 : float
            Volume in Angstrom^3.

        Returns:
        -------
        volume_L : float
            Volume in litres.

        Raises:
        ------
        None
        '''

        # Angstrom^3 is the standard unit converted to liters for molarity calculations
        return v_ang3 * 1e-27  

    def _count_from_molarity(self, max_box_size, molarity_mol_per_L, molecules_per_unit=1):

        '''
        Compute the integer number of solvent units to inject to achieve a target
        molarity, given the simulation box volume and Avogadro's number.
        molecules_per_unit allows for multi-molecule templates (e.g. ion pairs).

        Parameters:
        ----------
        max_box_size : list of list of float
            Box extents [[xlo, xhi], [ylo, yhi], [zlo, zhi]].
        molarity_mol_per_L : float
            Target concentration in mol L^-1 (e.g. 55.5 for pure water).
        molecules_per_unit : int, optional
            Number of physical molecules represented by one template unit (default 1).

        Returns:
        -------
        n_units : int
            Number of solvent units to inject (at least 0).

        Raises:
        ------
        None
        '''

        NA = 6.02214076e23  # mol^-1
        vL = self._ang3_to_liters(self._box_volume_ang3(max_box_size))
        n_molecules = molarity_mol_per_L * vL * NA

        n_units = int(np.rint(n_molecules / molecules_per_unit)) # round to int in a predictable way:

        return max(0, n_units)
    

    # -------------------------------------------------------------------------
    # SOLVENT LINE BUILDERS
    # -------------------------------------------------------------------------

    def write_solvent(self, sol, box_size, bridson_params=None, is_last_ionic=True):

        '''
        Dispatcher for solvent placement. Builds solvent data in memory then
        injects it into the main file.

        For an ionic group this also maintains the .salts sidecar, and enforces the
        rule that charge neutrality is applied by the LAST ionic group only; see the
        is_last_ionic parameter.

        Parameters:
        ----------
        sol : dict
            Solvation group dict from the JSON input.
        box_size : list of list of float
            Box extents [[xlo, xhi], [ylo, yhi], [zlo, zhi]].
        bridson_params : dict, optional
            Sampling tuning parameters passed through to _build_solvent.
        is_last_ionic : bool, optional
            True if this is the last ionic group in injection order, which is
            priority order, highest number first, so the last ionic group is the one
            with the lowest priority number, not the last one in the JSON. Charge
            neutrality is honoured only on that group, even if an earlier group set
            'charge-neutrality': true, because each correction is computed from the
            system charge at the time it runs. Applying it more than once means the
            later corrections react to charge the earlier ones already cancelled.
            The last group is also the only one that sees the fully accumulated
            charge, so a single correction there lands closest (default True, which
            is correct for a non-ionic group or a lone salt).

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        is_ionic = sol.get('placement') == 'ionic'
        min_distance = sol.get('min_distance', self.config.default_min_distance)

        if is_ionic and sol.get('charge-neutrality', True) and not is_last_ionic:
            my_log_file.warning(
                f"Solvation group '{sol['type']}' requested charge-neutrality but is not the "
                f"last ionic group in priority order. The correction is deferred to the last "
                f"ionic group, the one with the lowest priority number, which sees the full "
                f"accumulated system charge."
            )

        # Anything above this residue belongs to the group being placed.
        residues_before = self.file_manager.find_last_residue() if is_ionic else None

        solvent_data = self._build_solvent(
            max_box_size=       box_size,
            solvent_type=       sol['type'],
            placement_type=     sol['placement'],
            count=              sol.get('count', None),
            concentration=      sol.get('concentration', None),
            charge_neutral=     sol.get('charge-neutrality', True) and is_last_ionic,
            priority=           sol['priority'],
            placement_pad=      sol.get('placement_pad', None),
            min_distance=       min_distance,
            molecules_per_unit= sol.get('molecules_per_unit', 1),
            bridson_params=     bridson_params,
        )

        self._inject_solvent(solvent_data, min_distance=min_distance)

        # Record ions after the cull so deleted ions don't block later groups.
        if is_ionic:
            self.update_salt_record(residues_before)


    def _build_solvent(self, 
        max_box_size, solvent_type, 
        priority=100, min_distance=None,
        placement_pad=None, concentration=None, count=None, 
        charge_neutral=True, placement_type="packed", random_rotation=True,
        molecules_per_unit=1,
        bridson_params=None,
        ):


        '''
        Generate the position data for solvent molecules or ions and return a
        structured data dictionary ready for bulk injection. Supports two
        placement modes: 'packed' (Bridson Poisson-disk) for neutral solvents and
        'ionic' (random with exclusion zones) for electrolyte ions. Calls
        _charge_neutrality when charge_neutral is True to determine ion counts.

        Parameters:
        ----------
        max_box_size : list of list of float
            Box extents [[xlo, xhi], [ylo, yhi], [zlo, zhi]].
        solvent_type : str
            Solvent or ion identifier matching an entry in the solvents xdata file
            (e.g. 'SWM4-NDP', 'KCl').
        priority : int, optional
            Overlap-removal priority level for these molecules (default 100).
        min_distance : float, optional
            Minimum distance between placement centres (default 2.5 Angstrom).
        placement_pad : float, optional
            Overlap radius multiplier for Bridson placement; mutually exclusive
            with concentration and count.
        concentration : float, optional
            Target molarity (mol L^-1); mutually exclusive with placement_pad and count.
        count : int, optional
            Explicit molecule count; mutually exclusive with placement_pad and concentration.
        charge_neutral : bool, optional
            If True and placement_type is 'ionic', automatically compute the ion
            ratio needed to neutralise the system charge (default True).
        placement_type : str, optional
            'packed' for Bridson Poisson-disk placement, 'ionic' for random
            placement with no-go zones (default 'packed').
        random_rotation : bool, optional
            If True, apply a random quaternion rotation to each solvent molecule
            before placing it (default True).
        molecules_per_unit : int, optional
            Number of physical molecules per solvent template unit (default 1).
        bridson_params : dict, optional
            Sampling tuning parameters. Falls back to config.bridson_params if None.
            Keys: k, r_floor, repeats, bracket_steps, bin_steps, final_repeats,
            tol_frac, injection_limit.

        Returns:
        -------
        bulk_data : list of dict
            One dict per ion species (1 for packed, 2 for ionic); each dict maps
            section names to lists of formatted line arrays ready for
            _inject_solvent.

        Raises:
        ------
        ValueError
            If none or more than one of placement_pad, count, and concentration
            is provided for packed placement.
        ValueError
            If the system charge cannot be balanced within the ion injection limit.
        ValueError
            If a valid ionic position cannot be found within 10 000 attempts.
        '''

        if min_distance is None:
            min_distance = self.config.default_min_distance

        # JSON Bridson settings override the defaults one key at a time.
        _defaults = self.config.bridson_params
        bp = {**_defaults, **(bridson_params or {})}

        x_lim, y_lim, z_lim = max_box_size

        provided = (count is not None) + (concentration is not None) + (placement_pad is not None)

        if provided != 1:
            msg = (f"Packed placement requires exactly ONE of: placement_pad, count, concentration. \n          "
                   f"count={count is not None} \n          "
                   f"concentration={concentration is not None} \n          "
                   f"placement_pad={placement_pad is not None} ")
            my_log_file.error(msg)
            raise ValueError(msg)

        if placement_type != "ionic":

            my_log_file.info(f"Solvent placement: loading template for '{solvent_type}'...")
            molecule_lines = self.file_manager.find_and_extract_subsection(self.config.solvents_file, "Solvent", solvent_type)

            if molecule_lines == []:
                my_log_file.error(f"Solvent type {solvent_type} not found in solvents file.")
                raise ValueError(f"Solvent type {solvent_type} not found in solvents file.")
            
            position_lines = self.file_manager.find_and_extract_lines(molecule_lines, "Atoms")

            ref_positions, ref_idx, ref_types, ref_res = self.force_field_manager.get_atom_positions_all(position_lines)
            centroid = np.mean(ref_positions, axis=0)
            ref_centred_positions = ref_positions - centroid

            ref_radii = [np.max(ref_centred_positions[:,0]) - np.min(ref_centred_positions[:,0]),
                        np.max(ref_centred_positions[:,1]) - np.min(ref_centred_positions[:,1]),
                        np.max(ref_centred_positions[:,2]) - np.min(ref_centred_positions[:,2])]
            
            ref_radius = np.max(ref_radii) / 2
            my_log_file.info(f"Solvent placement: '{solvent_type}' template loaded, ref_radius={ref_radius:.3f} Angstrom.")

            if placement_pad is not None:
                overlap_radius = ref_radius * placement_pad
            else:
                overlap_radius = ref_radius

            self.force_field_manager.inject_atom_coefficients(molecule_lines)
            self.force_field_manager.inject_atom_priorities(molecule_lines, priority_level=priority)

        match placement_type:

            case "packed":

                target_count = None

                if count is not None:
                    target_count = int(count)
                elif concentration is not None:
                    target_count = self._count_from_molarity(max_box_size, float(concentration),
                                                            molecules_per_unit=molecules_per_unit)
                
                my_log_file.info(f"Packed placement: target={target_count} molecules of '{solvent_type}'.")

                if placement_pad is not None:
                    r = max(float(overlap_radius), float(min_distance))
                    if r > overlap_radius:
                        my_log_file.info(
                            f"Packed placement: placement_pad radius {overlap_radius:.3f} raised to "
                            f"min_distance {float(min_distance):.3f} angs."
                        )
                    my_log_file.info(f"Packed placement: using fixed placement_pad, r={r:.3f}. Running Bridson...")
                    centres = self._Bridson_poisson_3d(box=max_box_size, r=r)
                    my_log_file.info(f"Packed placement: Bridson complete, placed {len(centres)} centres.")

                else:
                    # Keep generated centres at least as far apart as the later cull requires.
                    r_floor = max(float(bp["r_floor"]), float(min_distance))
                    if r_floor > bp["r_floor"]:
                        my_log_file.info(
                            f"Packed placement: r_floor raised from {float(bp['r_floor']):.3f} to "
                            f"min_distance {float(min_distance):.3f} angs."
                        )

                    # Start from packing density; ref_radius is too small here.
                    # The 1.15 nudge usually keeps the search short.

                    vol = ((x_lim[1] - x_lim[0]) *
                        (y_lim[1] - y_lim[0]) *
                        (z_lim[1] - z_lim[0]))
                    
                    r_geometric = ((vol / target_count) * (3.0 / (4.0 * np.pi))) ** (1.0 / 3.0) * 1.15 # this one 
                    r_init_guess = float(np.maximum(r_geometric, r_floor))

                    my_log_file.info(
                        f"Packed placement: geometric r estimate = {r_geometric:.3f} angs "
                        f"(vol={vol:.1f} mu / target={target_count}), "
                        f"using r_init = {r_init_guess:.3f} ang [r_floor={r_floor:.3f} angs]."
                    )

                    r = self._find_bridson_r_for_count(
                        box=max_box_size,
                        target_count=target_count,
                        r_init=r_init_guess,       # was overlap_radius
                        r_floor=r_floor,
                        repeats=bp["repeats"],
                        bracket_steps=bp["bracket_steps"],
                        bin_steps=bp["bin_steps"],
                        k=bp["k"],
                    )

                    my_log_file.info(f"Packed placement: optimal r={r:.3f} found. Running final Bridson placement...")

                    if r < ref_radius:
                        my_log_file.warning(
                            f"Bridson cutoff r={r:.3f} < solvent characteristic ref_radius={ref_radius:.3f}. "
                            f"This will create severe overlaps and may not equilibrate at all."
                        )
                    tol_frac = bp["tol_frac"]
                    tol = int(np.ceil(tol_frac * target_count))
                    min_ok = target_count - tol
                    max_ok = target_count + tol

                    N_brid_r = bp["final_repeats"]

                    best_centres = None
                    best_n = -1
                    best_score = None

                    for i in range(N_brid_r):
                        my_log_file.info(f"Packed placement: Bridson attempt {i+1}/{N_brid_r} with r={r:.3f}...")
                        c = self._Bridson_poisson_3d(box=max_box_size, r=r, k=bp["k"])
                        n = len(c)
                        my_log_file.info(f"Packed placement: attempt {i+1}/{N_brid_r} placed {n} centres (target={target_count}, tol=+/-{tol}).")
                        score = (abs(n - target_count), -n)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_centres = c
                            best_n = n
                        if min_ok <= n <= max_ok:
                            my_log_file.info(f"Packed placement: within tolerance on attempt {i+1}, stopping early.")
                            best_centres = c
                            best_n = n
                            break

                    centres = best_centres

                    if best_n < min_ok:
                        my_log_file.warning(
                            f"Packed count below tolerance: target={target_count}, got={best_n}, "
                            f"tol=+/-{tol} (min_ok={min_ok}), r={r:.3f}, repeats={N_brid_r}. "
                            f"Accepting anyway (box will breathe)."
                        )
                    else:
                        my_log_file.info(
                            f"Packed placement: target={target_count}, placed={best_n}, "
                            f"tol=+-{tol}, r={r:.3f}, repeats={N_brid_r}."
                        )

                    if best_n > target_count:
                        my_log_file.warning(
                            f"Packed count above tolerance: target={target_count}, got={best_n}, "
                            f"tol=+/-{tol} (max_ok={max_ok}), r={r:.3f}, repeats={N_brid_r}. "
                            f"Randomly removing {best_n - target_count} centres to fit target."
                        )
                        idx = np.random.choice(best_n, size=target_count, replace=False)
                        centres = centres[idx]
                        best_n = target_count

                # Non-bulk packed groups check themselves too, so warn if their spacing may self-cull.
                if priority != getattr(self.config, 'bulk_priority', None):
                    self_clear = float(min_distance) + 2.0 * float(ref_radius)
                    if r < self_clear:
                        my_log_file.warning(
                            f"Solvent '{solvent_type}' is not the bulk phase, so it is checked "
                            f"against itself: Bridson centre spacing r={r:.3f} is below "
                            f"min_distance + 2*ref_radius = {self_clear:.3f} angs "
                            f"({float(min_distance):.3f} + 2 x {float(ref_radius):.3f}), so "
                            f"neighbouring molecules can fall within min_distance and be culled. "
                            f"Lower min_distance, reduce the count/concentration, or make this "
                            f"the bulk phase by giving it the highest priority number."
                        )

                my_log_file.info(f"Packed placement: rotating and positioning {len(centres)} molecules...")
                solvent_positions = np.empty((0, 3))
                if random_rotation:
                    for i, centre in enumerate(centres):
                        if i % 500 == 0 and i > 0:
                            my_log_file.info(f"Packed placement: rotated and placed {i}/{len(centres)} molecules...")
                        random_q = self.gt.random_unit_quaternion()
                        rotated_centered_positions = self.gt.rotate_points_quat(
                            ref_centred_positions, random_q, center=(0,0,0)
                        )
                        new = rotated_centered_positions + centre
                        solvent_positions = np.vstack([solvent_positions, new])
                else:
                    print("Why do you want them ordered? Weird dipoles will be induced.")

                my_log_file.info(f"Packed placement: compiling data lines for {len(centres)} {solvent_type} molecules...")
                data = self.compile_bulk_solvent_molecule_lines_IN_MEMORY(solvent_positions, molecule_lines)
                my_log_file.info(f"Built {len(centres)} {solvent_type} molecules (r={r:.3f}).")

                return [data]
        
            case "ionic":
                
                # I should change parameters rather than override, I will everntually. For now this is fine.
                electrolyte_priority = priority
                close_ion_distance = min_distance

                cathode_type, cathode_count, anode_type, anode_count = self._process_salt_formula(solvent_type)
                my_log_file.info(f"Ionic placement: '{solvent_type}' -> {cathode_type} x {cathode_count}, {anode_type} x {anode_count}.")

                cathode_lines = self.file_manager.find_and_extract_subsection(self.config.solvents_file, "Solvent", cathode_type)
                if cathode_lines == []:
                    my_log_file.error(f"Cathode type {cathode_type} not found in solvents file.")
                    raise ValueError(f"Cathode type {cathode_type} not found in solvents file.")
                self.force_field_manager.inject_atom_coefficients(cathode_lines)
                self.force_field_manager.inject_atom_priorities(cathode_lines, priority_level=electrolyte_priority)

                anode_lines = self.file_manager.find_and_extract_subsection(self.config.solvents_file, "Solvent", anode_type)
                if anode_lines == []:
                    my_log_file.error(f"Anode type {anode_type} not found in solvents file.")
                    raise ValueError(f"Anode type {anode_type} not found in solvents file.")
                self.force_field_manager.inject_atom_coefficients(anode_lines)
                self.force_field_manager.inject_atom_priorities(anode_lines, priority_level=electrolyte_priority)

                cathode_charge = self.force_field_manager.get_system_charge(cathode_lines)
                anode_charge = self.force_field_manager.get_system_charge(anode_lines)
                sys_charge = self.force_field_manager.get_system_charge()
                nearest_charge = self.force_field_manager._round_charge(sys_charge)

                if nearest_charge != 0:
                    n_cathode = int(count) * cathode_count
                    n_anode = int(count) * anode_count 

                    if charge_neutral:
                        my_log_file.warning(f"System charge is not neutral, {sys_charge}, attempting to correct with ions, correcting to {nearest_charge}.")
                        addition_cathodes, addition_anodes, final_charge = self._charge_neutrality(nearest_charge, cathode_count, anode_count, cathode_charge, anode_charge, injection_limit=bp["injection_limit"])
                        n_cathode += addition_cathodes
                        n_anode += addition_anodes
                        my_log_file.info(f"Ionic placement: charge correction applied, +{addition_cathodes} {cathode_type}, +{addition_anodes} {anode_type}.")
                else:
                    n_cathode = int(count) * cathode_count
                    n_anode = int(count) * anode_count

                my_log_file.info(f"Ionic placement: placing {n_cathode} {cathode_type} and {n_anode} {anode_type} (total={n_cathode+n_anode}).")
                my_log_file.info(f"System charge is {self.force_field_manager._round_charge(sys_charge)}, injecting {n_cathode} {cathode_type} {cathode_charge} and {n_anode} {anode_type} {anode_charge}.")

                # Read the structure sidecar; using the full file would make this cuboid cover the box.
                no_go_zones = []
                struct_pos, struct_types, _ = self._read_record_file(getattr(self.config, 'structure_record_file', None))
                for i, t in enumerate(struct_types):
                    p = self._get_atom_priority(t)
                    if p < electrolyte_priority:
                        no_go_zones.append(struct_pos[i, :])

                # Check earlier ions one by one; a cuboid around them would cover most of the box.
                previously_placed_ions, _, _ = self._read_record_file(getattr(self.config, 'salts_record_file', None))
                if previously_placed_ions.shape[0] > 0:
                    my_log_file.info(
                        f"Ionic placement: honouring {previously_placed_ions.shape[0]} ions "
                        f"already placed by earlier solvation groups."
                    )

                if no_go_zones != []:
                    no_go_zones = np.array(no_go_zones)
                    no_go_cuboid = [np.min(no_go_zones[:,0]) - min_distance - 0.001, np.max(no_go_zones[:,0]) + min_distance + 0.001,
                                    np.min(no_go_zones[:,1]) - min_distance - 0.001, np.max(no_go_zones[:,1]) + min_distance + 0.001,
                                    np.min(no_go_zones[:,2]) - min_distance - 0.001, np.max(no_go_zones[:,2]) + min_distance + 0.001]
                    my_log_file.info(f"Ionic placement: no-go cuboid set from {len(no_go_zones)} higher-priority atoms.")
                else:
                    my_log_file.warning("Solution atoms are highest priority, there is no CNT, just FYI.")

                cathodes = np.empty((0, 3))
                anodes = np.empty((0, 3))
                total_ions = n_anode + n_cathode

                for i in range(total_ions):

                    if i % 10 == 0 and i > 0:
                        my_log_file.info(f"Ionic placement: placed {i}/{total_ions} ions...")

                    valid_position = False 
                    attempts = 0

                    while not valid_position and attempts < 10000:

                        random_position = np.array([np.random.uniform(x_lim[0], x_lim[1]),
                                                    np.random.uniform(y_lim[0], y_lim[1]),
                                                    np.random.uniform(z_lim[0], z_lim[1])])
                        
                        if no_go_zones is not None and len(no_go_zones) > 0:
                            if (no_go_cuboid[0] < random_position[0] < no_go_cuboid[1] and
                                no_go_cuboid[2] < random_position[1] < no_go_cuboid[3] and
                                no_go_cuboid[4] < random_position[2] < no_go_cuboid[5] ):
                                attempts += 1
                                continue

                        if cathodes.shape[0] > 0:
                            dists_cathode = np.linalg.norm(cathodes - random_position, axis=1)
                            if np.any(dists_cathode < close_ion_distance):
                                attempts += 1
                                continue

                        if anodes.shape[0] > 0:
                            dists_anode = np.linalg.norm(anodes - random_position, axis=1)
                            if np.any(dists_anode < close_ion_distance):
                                attempts += 1
                                continue

                        # Keep this salt away from ions placed by earlier groups.
                        if previously_placed_ions.shape[0] > 0:
                            dists_previous = np.linalg.norm(previously_placed_ions - random_position, axis=1)
                            if np.any(dists_previous < close_ion_distance):
                                attempts += 1
                                continue

                        valid_position = True

                    if attempts == 10000:
                        my_log_file.error("Could not find valid position for electrolyte ion after 10000 attempts. Make a bigger box.")
                        raise ValueError("Could not find valid position for electrolyte ion after 10000 attempts. Make a bigger box.")

                    if i < n_cathode:
                        cathodes = np.vstack([cathodes, random_position])
                        if self.config.drude_polarisable == True:
                            cathodes = np.vstack([cathodes, random_position])
                    else:
                        anodes = np.vstack([anodes, random_position])
                        if self.config.drude_polarisable == True:
                            anodes = np.vstack([anodes, random_position])

                my_log_file.info(f"Ionic placement: compiling data lines for {len(cathodes)} {cathode_type} and {len(anodes)} {anode_type}...")
                cathode_data = self.compile_bulk_solvent_molecule_lines_IN_MEMORY(cathodes, cathode_lines)

                # Offset the anode block so its residue and topology ids stay unique.
                cathode_molecule_size = len(self.file_manager.find_and_extract_lines(cathode_lines, "Atoms")) - 1
                n_cathode_molecules = len(cathodes) // cathode_molecule_size
                section_offsets = {section: len(rows) for section, rows in cathode_data.items()}

                anode_data = self.compile_bulk_solvent_molecule_lines_IN_MEMORY(
                    anodes,
                    anode_lines,
                    residue_offset=n_cathode_molecules,
                    section_offsets=section_offsets,
                )

                my_log_file.info(f"Built {len(cathodes) + len(anodes)} {solvent_type} ions.")

                sys_charge = self.force_field_manager._round_charge(self.force_field_manager.get_system_charge())
                my_log_file.info(f"Close system charge: {sys_charge}")

                return [cathode_data, anode_data]
    
    def compile_bulk_solvent_molecule_lines_IN_MEMORY(
        self,
        positions,
        molecule_lines,
        residue_offset=0,
        section_offsets=None,
    ):

            '''
        Translate a set of molecule template lines to a list of positions,
        updating all indices (atom, bond, angle, dihedral, improper, lone pair,
        anisotropy) to be globally consistent with the current state of the main
        file. Returns a dict keyed by section name.

        Parameters:
        ----------
        positions : (N,3) np.ndarray
            Cartesian coordinates at which to place each molecule unit.
        molecule_lines : list of str
            Lines from the solvent xdata sub-block (including section headers).
        residue_offset : int, optional
            Added to the current residue high-water mark (default 0).
        section_offsets : dict, optional
            Per-section offsets added to the current index high-water marks.

        Returns:
        -------
        data : defaultdict of list
            Keys are section names (e.g. 'Atoms', 'Bonds'); values are lists of
            formatted line arrays ready for standardise_line.

        Raises:
        ------
        None
        '''

            section_offsets = section_offsets or {}
            sections = [
                "Atoms",
                *(
                    section for section in [
                        "Bonds", "Angles", "Dihedrals", "Impropers",
                        "Lone Pairs", "Anisotropy",
                    ]
                    if section in self.config.enabled_xdata_sections
                ),
            ]
            
            main_lines = self.file_manager.read_file(self.config.main_file)
            molecule_size = len(self.file_manager.find_and_extract_lines(molecule_lines, "Atoms")) - 1
            last_atom_idx = (
                self.file_manager.find_last_section_idx(self.config.main_file, "Atoms")
                + section_offsets.get("Atoms", 0)
            )

            data = defaultdict(list)

            for section in sections:
                lines = self.file_manager.find_and_extract_lines(molecule_lines, section)
                last_section_idx = (
                    self.file_manager.find_last_section_idx(self.config.main_file, section)
                    + section_offsets.get(section, 0)
                )

                section_data  = []
                
                # extract molecule data
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    section_data.append(self.file_manager.line_to_array(line))
                
                # molecule doesnt contain the section 
                if section_data == []:
                    continue
                
                if section == "Atoms":
                    
                    last_residue_idx = self.file_manager.find_last_residue() + residue_offset

                    for ii in range(len(positions)):
                        local_index = ii % molecule_size

                        # we need to get type of main file not local definition
                        nametype1 = section_data[local_index][-1]
                        type1 = self.force_field_manager.get_new_atom_type(
                            nametype1,
                            source_atom_number_type=section_data[local_index][2],
                            source_lines=molecule_lines,
                        )
                        charge1  = section_data[local_index][3]

                        new_line = [
                            ii + last_section_idx + 1,
                            last_residue_idx +(ii//molecule_size) + 1,
                            type1,
                            charge1,
                            *positions[ii],
                            "#",
                            nametype1
                        ]
                        data[section].append(new_line)

                if section == "Bonds":
                    
                    for ii in range(len(positions) // molecule_size):

                        for iii, bond in enumerate(section_data):

                            nametype1 = bond[-2]
                            nametype2 = bond[-1]
                            
                            bondtype = self.force_field_manager.get_new_bond_type(nametype1, nametype2)
                            new1_idx = int(bond[2]) + last_atom_idx + ii * molecule_size
                            new2_idx = int(bond[3]) + last_atom_idx + ii * molecule_size

                            new_line = [
                                last_section_idx + iii + 1 + ii * len(section_data),
                                bondtype,
                                new1_idx,
                                new2_idx,
                                "#",
                                nametype1,
                                nametype2
                            ]
                            
                            data[section].append(new_line)
                
                if section == "Angles":

                    for ii in range(len(positions) // molecule_size):

                        for iii, angle in enumerate(section_data):

                            nametype1 = angle[-3]
                            nametype2 = angle[-2]
                            nametype3 = angle[-1]

                            angletype, flipped = self.force_field_manager.get_new_angle_type(nametype1, nametype2, nametype3)
                            new1_idx = int(angle[2]) + last_atom_idx + ii * molecule_size
                            new2_idx = int(angle[3]) + last_atom_idx + ii * molecule_size
                            new3_idx = int(angle[4]) + last_atom_idx + ii * molecule_size
                            
                            if flipped:
                                nametype1, nametype3 = nametype3, nametype1
                                new1_idx, new3_idx = new3_idx, new1_idx


                            new_line = [
                                last_section_idx + iii + 1 + ii * len(section_data),
                                angletype,
                                new1_idx,
                                new2_idx,
                                new3_idx,
                                "#",
                                nametype1,
                                nametype2,
                                nametype3
                            ]

                            data[section].append(new_line)

                if section == "Dihedrals":

                    for ii in range(len(positions) // molecule_size):

                        for iii, dih in enumerate(section_data):

                            nametype1 = dih[-4]
                            nametype2 = dih[-3]
                            nametype3 = dih[-2]
                            nametype4 = dih[-1]

                            dihedraltype, flipped = self.force_field_manager.get_new_dihedral_type(nametype1, nametype2, nametype3, nametype4)
                            new1_idx = int(dih[2]) + last_atom_idx + ii * molecule_size
                            new2_idx = int(dih[3]) + last_atom_idx + ii * molecule_size
                            new3_idx = int(dih[4]) + last_atom_idx + ii * molecule_size
                            new4_idx = int(dih[5]) + last_atom_idx + ii * molecule_size

                            if flipped:
                                nametype1, nametype2, nametype3, nametype4 = nametype4, nametype3, nametype2, nametype1
                                new1_idx, new2_idx, new3_idx, new4_idx = new4_idx, new3_idx, new2_idx, new1_idx

                            new_line = [
                                last_section_idx + iii + 1 + ii * len(section_data),
                                dihedraltype,
                                new1_idx,
                                new2_idx,
                                new3_idx,
                                new4_idx,
                                "#",
                                nametype1,
                                nametype2,
                                nametype3,
                                nametype4
                            ]

                            data[section].append(new_line)

                if section == "Impropers":  

                    for ii in range(len(positions) // molecule_size):

                        for iii, imp in enumerate(section_data):

                            nametype1 = imp[-4]
                            nametype2 = imp[-3]
                            nametype3 = imp[-2]
                            nametype4 = imp[-1]

                            impropertype, flipped = self.force_field_manager.get_new_improper_type(nametype1, nametype2, nametype3, nametype4)
                            new1_idx = int(imp[2]) + last_atom_idx + ii * molecule_size
                            new2_idx = int(imp[3]) + last_atom_idx + ii * molecule_size
                            new3_idx = int(imp[4]) + last_atom_idx + ii * molecule_size
                            new4_idx = int(imp[5]) + last_atom_idx + ii * molecule_size

                            if flipped:
                                nametype1, nametype2, nametype3, nametype4 = nametype4, nametype3, nametype2, nametype1
                                new1_idx, new2_idx, new3_idx, new4_idx = new4_idx, new3_idx, new2_idx, new1_idx

                            new_line = [
                                last_section_idx + iii + 1 + ii * len(section_data),
                                impropertype,
                                new1_idx,
                                new2_idx,
                                new3_idx,
                                new4_idx,
                                "#",
                                nametype1,
                                nametype2,
                                nametype3,
                                nametype4
                            ]

                            data[section].append(new_line)
                
                if section == "Lone Pairs":

                    for ii in range(len(positions) // molecule_size): # number of molecules

                        for iii, lone_pair in enumerate(section_data):      
                            lone_pair_dependecies = lone_pair[-1].strip("()").split(",")
                            lpd = ""
                            for item in lone_pair_dependecies:
                                local_index = int(item)
                                global_index = local_index + last_atom_idx + ii * molecule_size
                                lpd += f"{global_index},"
                            
                            new_line = [
                                last_section_idx + ii * len(section_data) + iii + 1, # last entry + no mols * no lones + no lones 
                                int(lone_pair[1]) + last_atom_idx + ii * molecule_size,  # atom index
                                lone_pair[2],  # lone pair type
                                lone_pair[3],  # distance
                                lone_pair[4],  # angle
                                lone_pair[5],  # dihedral
                                lone_pair[6],  # hashtag symbol
                                lone_pair[7],  # name type
                                f"({lpd[:-1]})"  # dependencies
                            ]

                            data[section].append(new_line)

                if section == "Anisotropy":

                    for ii in range(len(positions) // molecule_size): # number of molecules

                        for iii, aniso in enumerate(section_data):      
                            aniso_dependecies = aniso[-1].strip("()").split(",")
                            ans = ""
                            for item in aniso_dependecies:
                                local_index = int(item)
                                global_index = local_index + last_atom_idx + ii * molecule_size
                                ans += f"{global_index},"
                            
                            new_line = [
                                last_section_idx + ii * len(section_data) + iii + 1, # last entry + no mols * no lones + no lones 
                                int(aniso[1]) + last_atom_idx + ii * molecule_size,  # atom index
                                aniso[2], # A11
                                aniso[3], # A22
                                aniso[4], # hashtag symbol
                                aniso[5], # atom_type
                                f"({ans[:-1]})"  # dependencies
                            ]

                            data[section].append(new_line)

            return data

    def _inject_solvent(self, bulk, min_distance=None):

        '''
        Inject the pre-compiled bulk solvent data into the main file section by
        section, then remove any overlapping atoms, re-standardise column widths,
        and update the header counts.

        Parameters:
        ----------
        bulk : list of dict
            Output from _build_solvent; one dict per ion
            species, mapping section names to lists of line arrays.
        min_distance : float, optional
            Overlap distance threshold passed to _remove_overlaps_IN_MEMORY. Applies
            to this group's atoms only; atoms already in the system keep whatever
            separation their own injection established (default
            config.default_min_distance).

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        if min_distance is None:
            min_distance = self.config.default_min_distance

        # Take this once so both species in an ionic bulk count as new during the cull.
        residues_before = self.file_manager.find_last_residue()

        for data in bulk: # could be 2 for ionic
            
            for section in data:

                inject_idx = 0
                main_lines = self.file_manager.read_file(self.config.main_file)    #Pull

                for line in data[section]: #Inject
                    section_end = self.file_manager.find_injection_point(self.config.main_file, section)
                    new_line = self.file_manager.standardise_line(line, section)
                    self.file_manager.insert_line(main_lines, section_end + inject_idx, new_line)
                    inject_idx += 1

                self.file_manager.write_file(main_lines, self.config.main_file)      #Push

        all_pos, all_idx, all_types, all_res = self.force_field_manager.get_atom_positions_all(self.config.main_file)
        
        self._remove_overlaps_IN_MEMORY(all_types, all_idx, all_pos, all_res, min_distance=min_distance,
                                        new_residues_from=residues_before)

        sys_charge = self.force_field_manager._round_charge(self.force_field_manager.get_system_charge())
        my_log_file.info(f"Post-cut system charge: {sys_charge}")

        self.file_manager.standardise_file()
        self.file_manager.standardise_top_counts()


class IO: 

    def __init__(self,
                 force_field_manager: Force_Field_Manager,
                 file_manager: File_Manager,
                 structure: Structure_Generator, 
                 solvation: Solution_Generator,
                 functionalisation: Functional_Group_Generator,
                 config: Config,
                 geometry: Geometry,
                 json_input: str | dict | None = None) -> None:

        self.force_field_manager : Force_Field_Manager = force_field_manager
        self.file_manager : File_Manager = file_manager

        self.structure : Structure_Generator = structure
        self.solvation : Solution_Generator = solvation
        self.functionalisation : Functional_Group_Generator = functionalisation

        self.geometry : Geometry = geometry
        self.config : Config = config

        self.json_input : str | dict | None = json_input

        self.box_size = None
        self._psf_charge_neutrality_report = None

        self.initialise_LOG(json_input)
        self.initialise_JSON(json_input)
        # Seed here so CLI and library runs use the same path.
        self.config.apply_seed(self.json_input)
        self._disabled_xdata_sections = self._get_disabled_xdata_sections()
        self.config.enabled_xdata_sections = (
            set(XDATA_SECTION_SETTINGS.values()) - self._disabled_xdata_sections
        )
        self.config.functionalisation_requested = bool(
            self.json_input["system"]["functionalisation"]
        )
        self.initialise_OUT_FOLDER(json_input)

    # ============================
    # INITIALISATION 
    # ============================

    def initialise_JSON(self, json_input): 

        '''
        Load the simulation settings into self.json_input from one of three
        sources: a dict passed directly (e.g. the module-level INPUT constant),
        a file path string pointing to a JSON file, or None which causes the
        function to look for 'chirality_kit_input.json' next to the script.

        Parameters:
        ----------
        json_input : str, dict, or None
            If a dict, used directly. If a str, treated as a file path. If None,
            the default 'chirality_kit_input.json' in the script directory is read.

        Returns:
        -------
        None

        Raises:
        ------
        FileNotFoundError
            If json_input is None and the default JSON file does not exist.
        ValueError
            If the file at the given path is not valid JSON.
        '''

        if isinstance(json_input, dict): # Dict passed directly (e.g. the INPUT constant)
            self.json_input = json_input

        elif json_input is None: # Normal JSON location
            json_file_path = os.path.join(self.config.script_location, 'chirality_kit_input.json')
            try:
                with open(json_file_path, 'r') as f:
                    self.json_input = json.load(f)  # parse JSON directly from file
            except FileNotFoundError:
                my_log_file.error("Input JSON file not found.")
                raise
            except json.JSONDecodeError:
                my_log_file.error("Input JSON file is not valid JSON.")
                raise ValueError("Input JSON file is not valid JSON.")
            
        else: # Override JSON location (file path string)
            try:
                with open(json_input, 'r') as f:
                    self.json_input = json.load(f)  # parse JSON directly from file
            except Exception as e:
                my_log_file.error(f"Failed to load JSON input: {e}")
                raise

    def initialise_LOG(self, json_input):

        '''
        Provide json simulation setting to the logger, for future support.

        Parameters:
        ----------
        json_input : str, dict, or None
            If a dict, used directly. If a str, treated as a file path. If None,
            the default 'chirality_kit_input.json' in the script directory is read.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''
    
        global my_log_file
        my_log_file = logging.getLogger(__name__)
        my_log_file.info(f"Log file for chirality_kit run on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        my_log_file.info(f"System name: {self.config.system_name}\n")

    def initialise_OUT_FOLDER(self, json_input):

        '''
        Create the output directory, set up the Python logging handler to write
        to a log file inside that directory, and create a blank xdata file with
        all section headers. Sets config.system_folder and config.main_file.

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

        if self.config.out_name:
            base_dir = os.path.expanduser(self.config.out_name)
            if not os.path.isabs(base_dir):
                base_dir = os.path.join(self.config.called_from, base_dir)
            base_dir = os.path.abspath(base_dir)
            if not os.path.basename(base_dir):
                raise ValueError("--out-name must identify a folder name.")
        else:
            system_name = self.json_input['settings']['system_name']
            if system_name == "auto":
                system_name = self.create_system_name()
            base_dir = os.path.join(self.config.called_from, system_name)

        increment = 0
        while True:
            proposed_dir = base_dir if increment == 0 else f"{base_dir}_({increment})"
            try:
                os.makedirs(proposed_dir)
                break
            except FileExistsError:
                increment += 1

        self.config.system_folder = proposed_dir
        self.config.system_name = os.path.basename(proposed_dir)

        log_file_path = os.path.join(self.config.system_folder, self.config.system_name + '.log') # setup the logger
        logging.basicConfig(
            filename=log_file_path,
            filemode="a",                    
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.DEBUG              
        )

        with open(os.path.join(self.config.system_folder, self.config.system_name + '.xdata'), 'w') as f: # make blank xdata file
            for section in self.config.section_order: 
                if section in self._disabled_xdata_sections:
                    continue
                f.write(f"{section}\n\n\n\n")
        
        # this is the main file
        self.config.main_file = os.path.join(self.config.system_folder, self.config.system_name + '.xdata')

        # Clear old sidecars if this output folder is being reused.
        self.config.structure_record_file = os.path.join(self.config.system_folder, self.config.system_name + '.nanotube')
        self.config.salts_record_file = os.path.join(self.config.system_folder, self.config.system_name + '.salts')
        for stale_record in (self.config.structure_record_file, self.config.salts_record_file):
            if os.path.exists(stale_record):
                os.remove(stale_record)


        # copy / serialise input json to output folder
        dst = os.path.join(self.config.system_folder, 'chirality_kit_input.json')
        if isinstance(self.json_input, dict):
            # dict input: save it as JSON
            with open(dst, 'w') as f:
                json.dump(self.json_input, f, indent=4)
        else:
            # path input: copy it as-is
            shutil.copyfile(self.json_input, dst)

        # Save the seed so this run can be replayed.
        self.json_input.setdefault('metadata', {})['rng_seed_used'] = self.config.seed
        with open(dst, 'w') as f:
            json.dump(self.json_input, f, indent=4)

        # The logger starts after apply_seed(), so log the seed again here.
        my_log_file.info(f"RNG seed for this run: {self.config.seed}")
        my_log_file.info(f"Output directory set to {self.config.system_folder}")


    def run(self):

        '''
        Execute the full chirality-kit pipeline: initialise the output folder,
        build the master xdata file from JSON settings, write all requested output
        formats (.pdb, .psf, .xyz, .data), and copy the chosen force-field's
        parameter files (minus the fragments/solvents xdata) into the output
        directory.

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

        # ============================
        # MASTER FILE BUILDING
        # ============================

        self.write_master_file_from_JSON()

        # ============================
        # OTHER .xxx FORMATS
        # ============================

        for data_type in self.json_input['settings']["output_types"]:
            getattr(self, f"write_{data_type[1:]}_file")()

        # ============================
        # CASE SPECIFICS 
        # ============================

        if '.pdb' in self.json_input['settings']["output_types"]: self.write_pdb_restraints_file()

        # ============================
        # SUMMARY FILE
        # ============================

        self._write_summary_file()
        my_log_file.info("Chirality Kit run complete. Thanks for using Chirality Kit! <3")

    def _get_disabled_xdata_sections(self):

        settings = self.json_input["settings"]
        missing = [key for key in XDATA_SECTION_SETTINGS if key not in settings]
        if missing:
            raise ValueError(
                "Missing required settings: " + ", ".join(missing)
            )

        invalid = [
            key for key in XDATA_SECTION_SETTINGS
            if not isinstance(settings[key], bool)
        ]
        if invalid:
            raise ValueError(
                "Xdata section settings must be booleans: " + ", ".join(invalid)
            )

        return {
            section for key, section in XDATA_SECTION_SETTINGS.items()
            if not settings[key]
        }

    def write_master_file_from_JSON(self):

        '''
        Orchestrate the full structure-building sequence driven by the JSON input:
        (0) initialise the force field and settings, (1) generate the nanotube
        structure, (2) attach functional groups, (3) solvate the system, and
        (4) finalise non-bonded interaction coefficients. Logs the final system
        and nanotube charges when complete.

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

        # -------------------------------------------------------------------------
        # 0. INITIALISATION  STEP
        # -------------------------------------------------------------------------
        
        my_log_file.info("Starting Environment Build")
        self.config.reinitialise(
            force_field       = self.json_input['system']['force-field'],
            nanotube          = self.json_input['system']['nanotube'],
            functionalisation = self.json_input['system']['functionalisation'],
            solvation         = self.json_input['system']['solvation'],
            nanotube_type     = self.json_input.get('nanotube', {}).get('type'),
        )
        self.config.verbose=self.json_input['settings']['verbose']
        
        my_log_file.info(f"Using force field: {self.json_input['system']['force-field']}")
        supported_info = self.config.get_supported_info()
        my_log_file.info(f"Supported Info: {supported_info}")
        if 'low_priority_atom_cutoff' in self.json_input.get('system', {}):
            my_log_file.warning(
                "Ignoring 'low_priority_atom_cutoff': the bulk phase is derived from the "
                "solvation groups now (the highest-priority packed group). The key does "
                "nothing and can be deleted from the input."
            )

        pieces = set() # keeps track of all the pieces we are adding
        fg_types = []
        multi_species_nanotubes = {"BNNT", "MoS2NT", "MoS2", "MoSSeNT", "MoSSe"}

        if self.json_input['system']['size'] == "auto" and self.json_input['system']['nanotube'] == False:  # Box size logic.
            my_log_file.error("Only 'auto' size with 'nanotube' system other wise provide a specific size.")
        self.box_size = self.json_input['system']['size']

        my_log_file.info("Starting Build Validation")
        self._validate_input(
            force_field       = self.json_input['system']['force-field'],
            nanotube          = self.json_input['system']['nanotube'],
            functionalisation = self.json_input['system']['functionalisation'],
            solvation         = self.json_input['system']['solvation'],
        )
        my_log_file.info("Build Validation Successful")

        # -------------------------------------------------------------------------
        # 1. NANOTUBE GENERATION
        # -------------------------------------------------------------------------

        my_log_file.info(f"ADDING NANOSTRUCTURE...   {self.json_input['system']['nanotube']}")
        if self.json_input['system']['nanotube'] == True:
            new_res_idx = "42" # because its the answer to everything

            # Finite tubes fold the top edge back by Lz. Periodic tubes keep one-bond edge atoms.
            periodic = self.json_input['nanotube'].get('periodic', False) is True
            if periodic:
                my_log_file.warning("nanotube.periodic is enabled: the tube is built for "
                                    "periodic simulation along z and may contain atoms with a single bond.")

            if self.json_input['nanotube']['type'] in {"CNT", "CNT-Drude"}:
                
                # -------------------------------------------------------------------------
                # CARBON NANOTUBE GENERATION
                # -------------------------------------------------------------------------

                '''
                Iijima, S. (1991). 
                Helical microtubules of graphitic carbon. Nature, 
                354(6348), 56-58. 
                https://doi.org/10.1038/354056a0
                '''

                self.force_field_manager.write_default_species()
                self.structure.write_nanotube(n=self.json_input['nanotube']['n'], 
                                                        m=self.json_input['nanotube']['m'], 
                                                        Ncells=self.json_input['nanotube']['repeats'],
                                                        PBC=periodic)
                pieces.add("Nanotube-Species")
                n_atoms = self.file_manager.find_number_of_entries_in_section('Atoms')
                self._log_cnt_geometry(self.json_input['nanotube']['n'],
                                    self.json_input['nanotube']['m'],
                                    self.json_input['nanotube']['repeats'],
                                    n_atoms=n_atoms)
            
            elif self.json_input['nanotube']['type'] == "BNNT":
                
                # -------------------------------------------------------------------------
                # BORON NITRIDE NANOTUBE GENERATION
                # -------------------------------------------------------------------------

                '''
                Rubio, A., Corkill, J. L., & Cohen, M. L. (1994). 
                Theory of graphitic boron nitride nanotubes. Physical Review B, 49(7), 5081. 
                https://doi.org/10.1103/physrevb.49.5081
                '''

                self.force_field_manager.write_default_species()
                self.structure._write_bnnt_nanotube(
                    n=self.json_input['nanotube']['n'],
                    m=self.json_input['nanotube']['m'],
                    Ncells=self.json_input['nanotube']['repeats'],
                    PBC=periodic,
                )
                pieces.add("Nanotube-Species")
                n_atoms = self.file_manager.find_number_of_entries_in_section('Atoms')
                self._log_cnt_geometry(self.json_input['nanotube']['n'],
                                    self.json_input['nanotube']['m'],
                                    self.json_input['nanotube']['repeats'],
                                    n_atoms=n_atoms)

            elif self.json_input['nanotube']['type'] in {"MoS2NT", "MoS2"}:

                # -------------------------------------------------------------------------
                # MOLYBDENUM DISULFIDE NANOTUBE GENERATION
                # -------------------------------------------------------------------------

                '''
                Luo, L. et al. (2025).
                Symmetry-broken MoS2 nanotubes through sequential sulfurization
                of MoO2 nanowires. Nature Communications, 16, 8394.
                https://doi.org/10.1038/s41467-025-63333-1
                '''

                self.force_field_manager.write_default_species()
                self.structure._write_mos2_nanotube(
                    n=self.json_input['nanotube']['n'],
                    m=self.json_input['nanotube']['m'],
                    Ncells=self.json_input['nanotube']['repeats'],
                    PBC=periodic,
                )
                pieces.add("Nanotube-Species")
                n_atoms = self.file_manager.find_number_of_entries_in_section('Atoms')
                self._log_cnt_geometry(self.json_input['nanotube']['n'],
                                    self.json_input['nanotube']['m'],
                                    self.json_input['nanotube']['repeats'],
                                    n_atoms=n_atoms)

            elif self.json_input['nanotube']['type'] in {"MoSSeNT", "MoSSe"}:

                # -------------------------------------------------------------------------
                # JANUS MOLYBDENUM SULFUR SELENIDE NANOTUBE GENERATION
                # -------------------------------------------------------------------------

                '''
                Tang, M., Pang, Y., Luo, Y. F., Song, Q., & Wang, M. (2019).
                Electronic properties of Janus MoSSe nanotubes. Computational
                Materials Science, 156, 315-320.
                https://doi.org/10.1016/j.commatsci.2018.10.012
                '''

                self.force_field_manager.write_default_species()
                self.structure._write_mosse_nanotube(
                    n=self.json_input['nanotube']['n'],
                    m=self.json_input['nanotube']['m'],
                    Ncells=self.json_input['nanotube']['repeats'],
                    outer_chalcogen=self.json_input['nanotube'].get('outer-chalcogen', 'Se'),
                    PBC=periodic,
                )
                pieces.add("Nanotube-Species")
                n_atoms = self.file_manager.find_number_of_entries_in_section('Atoms')
                self._log_cnt_geometry(self.json_input['nanotube']['n'],
                                    self.json_input['nanotube']['m'],
                                    self.json_input['nanotube']['repeats'],
                                    n_atoms=n_atoms)

            else:
                raise ValueError(f"Unsupported nanotube type: {self.json_input['nanotube']['type']}")
            

            # -------------------------------------------------------------------------
            # 2. FUNCTIONALISATION
            # -------------------------------------------------------------------------

            my_log_file.info(f"ADDING FUNCTIONAL GROUPS...   {self.json_input['system']['functionalisation']}")
            if self.json_input['system']['functionalisation'] == True:
                if self.json_input['nanotube']['type'] in multi_species_nanotubes:
                    raise NotImplementedError(
                        "Functionalisation is not implemented for multi-species nanotubes yet."
                    )

                structure_info, index_info, type_info = self.force_field_manager.get_atom_positions_residue(residue=new_res_idx)
                bond_info = (
                    self.force_field_manager._create_bonds_data(atom_offset=0)
                    if "Bonds" in self.config.enabled_xdata_sections
                    else self.force_field_manager.generated_bonds_data
                )
                
                self._validate_fg_available_carbons(fg_config=self.json_input['functionalisation'], n_atoms=n_atoms)

                fg_idx, fg_vec, fg_pos, fg_types = self.functionalisation._get_points_of_attachment(
                    structure_info, index_info, bond_info,
                    fg_config=self.json_input['functionalisation'],
                )

                if self.config.verbose:
                    my_log_file.info(f"Attaching functional groups of types:\n {fg_types}")
                    my_log_file.info(f"Attaching functional groups at indices:\n {fg_idx}")
                    my_log_file.info(f"With attachment vectors:\n {fg_vec}")

                self.functionalisation.write_functional_groups(
                    self.config.fragments_file,
                    all_types=fg_types,
                    all_indexes=fg_idx,
                    all_vectors=fg_vec,
                    all_positions=fg_pos,
                    residue_idx=new_res_idx,
                )
                self._log_fg_distribution(fg_types, fg_idx, fg_pos)

                if self.config.enabled_xdata_sections & {
                    "Angles", "Dihedrals", "Impropers"
                }:
                    bond_info = (
                        self.force_field_manager._create_bonds_data(atom_offset=0)
                        if "Bonds" in self.config.enabled_xdata_sections
                        else self.force_field_manager.generated_bonds_data
                    )
                    self.force_field_manager.inject_angles_from_bonds(bond_info, angle_idx_offset=0)
                    self.force_field_manager.inject_dihedrals_from_bonds(bond_info, dihedral_idx_offset=0)
                    self.force_field_manager.inject_impropers_from_bonds(bond_info, improper_idx_offset=0)

                pieces.update(fg_types)

        # Save the bare structure now; ion placement can skip the full solvated file later.
        self.solvation.write_structure_record()


        # -------------------------------------------------------------------------
        # 3. SOLVATION
        # -------------------------------------------------------------------------

        my_log_file.info(f"ADDING SOLVENTS...   {self.json_input['system']['solvation']}")

        if self.box_size == "auto":
            
            structure_info, index_info, type_info = self.force_field_manager.get_atom_positions_residue(residue=new_res_idx)       
            PBC_pad_xy, PBC_pad_z = self._resolve_auto_box_padding()
            structure_info = np.array(structure_info)
            x_max, x_min = np.max(structure_info[:,0]), np.min(structure_info[:,0])
            y_max, y_min = np.max(structure_info[:,1]), np.min(structure_info[:,1])
            z_max, z_min = np.max(structure_info[:,2]), np.min(structure_info[:,2])
            self.box_size = [
                            [x_min - PBC_pad_xy, x_max + PBC_pad_xy],
                            [y_min - PBC_pad_xy, y_max + PBC_pad_xy],
                            [z_min - PBC_pad_z, z_max + PBC_pad_z] ]

        if self.json_input['system']['solvation'] == True:

            # Highest priority goes first; ties keep input order.
            solvation_groups = sorted(
                self.json_input['solvation']['groups'],
                key=lambda g: -int(g['priority']),
            )

            # Only the last ionic group neutralises, after all salt charges are present.
            last_ionic_idx = max(
                (i for i, g in enumerate(solvation_groups) if g.get('placement') == 'ionic'),
                default=None,
            )

            for group_idx, sol in enumerate(solvation_groups):

                effective_box = self.box_size
                pad = float(sol.get('pad', 0.0) or 0.0)           # effectively keeping molecules inside the box
                if pad > 0.0 and sol.get('placement') != 'ionic': # ions are points so dont need padding
                    effective_box = [
                        [self.box_size[0][0] + pad, self.box_size[0][1] - pad],
                        [self.box_size[1][0] + pad, self.box_size[1][1] - pad],
                        [self.box_size[2][0] + pad, self.box_size[2][1] - pad],
                    ]
                    my_log_file.info(
                        f"Solvent '{sol['type']}': applying symmetric pad of {pad} Angstrom "
                        f"(effective box "
                        f"{effective_box[0][1]-effective_box[0][0]:.2f} x "
                        f"{effective_box[1][1]-effective_box[1][0]:.2f} x "
                        f"{effective_box[2][1]-effective_box[2][0]:.2f} Angstrom)."
                    )

                n_atom_prior = self.file_manager.find_number_of_entries_in_section('Atoms')
                self.solvation.write_solvent(
                    sol,
                    box_size = effective_box,
                    bridson_params= self.json_input['solvation'].get('bridson_params', None),
                    is_last_ionic= (group_idx == last_ionic_idx),
                )
                n_atoms_after = self.file_manager.find_number_of_entries_in_section('Atoms')

                if sol["placement"] == "ionic":
                    cathode_type, cathode_count, anode_type, anode_count = self.solvation._process_salt_formula(sol['type'])
                    my_log_file.info(f"Added {cathode_count} {cathode_type} and {anode_count} {anode_type} ions for solvation group {sol['type']}.")
                    pieces.add(cathode_type)
                    pieces.add(anode_type)
                else:
                    new_solvs = sum(1 for atom in self.force_field_manager.get_atom_positions_all()[2] if sol['type'] in atom)
                    self._log_solvation_stats(sol, n_atom_prior, n_atoms_after, new_solvs)
                    pieces.add(sol['type'])


        # -------------------------------------------------------------------------
        # 4. FINALISE NON-BONDED INTERACTIONS
        # -------------------------------------------------------------------------

        my_log_file.info("FINALISING NON-BONDED INTERACTIONS...")
        # Sort pieces so NBFIX output doesn't depend on Python's hash order.
        for piece in sorted(pieces):
            mol = self.file_manager.find_and_extract_subsection(self.config.fragments_file, "Fragment", piece)
            if not mol:
                mol = self.file_manager.find_and_extract_subsection(self.config.solvents_file, "Solvent", piece)
            if not mol:
                my_log_file.error(f"Could not find piece {piece} in fragments or solvents file.")
                raise ValueError(f"Could not find piece {piece} in fragments or solvents file.")
            self.force_field_manager.inject_lj_coefficients(mol)
        my_log_file.info("Environment Build Complete")

        # -------------------------------------------------------------------------
        # 5. COPY OVER FORCE FIELD PARAMETER FILES
        # -------------------------------------------------------------------------

        self.copy_other_FF_files()

        # -------------------------------------------------------------------------
        # 6. FINAL LOG MESSAGES
        # -------------------------------------------------------------------------

        self._print_charge_summary()
        self._check_pbc_box()

    def create_system_name(self): 

        '''
        Construct a descriptive system name string from the JSON settings,
        combining the force-field name, nanotube chirality, functional group
        types and counts, and solvent placement descriptors.

        Parameters:
        ----------
        None

        Returns:
        -------
        name : str
            Human-readable system identifier used for naming the output folder
            and all output files.

        Raises:
        ------
        None
        '''

        def sanitize(s):
            return (
                str(s)
                .replace("+", "")
                .replace("-", "")
                .replace(" ", "")
                .replace(".", "p")
            )

        j = self.json_input
        parts = []

        # --- Force field ---
        parts.append(sanitize(j["system"]["force-field"]))

        # --- Nanotube ---
        if j["system"].get("nanotube"):
            nt = j["nanotube"]
            parts.append(f"{sanitize(nt['type'])}_({nt['n']},{nt['m']})x{nt['repeats']}")

        # --- Functionalisation ---
        if j["system"].get("functionalisation") and j["system"].get("nanotube"):
            func = j["functionalisation"]

            summary = {}

            # --- Term groups ---
            term = func.get("term", {})
            for fg in term.get("groups", []):
                if fg["type"] == DUMMY_FG_TYPE:
                    continue
                count = fg.get("count", 0)
                if count > 0:
                    summary[fg["type"]] = summary.get(fg["type"], 0) + count

            # --- Ring groups (count x ring-count) ---
            for ring in func.get("rings", []):
                ring_multiplier = ring.get("ring-count", 0)

                for fg in ring.get("groups", []):
                    if fg["type"] == DUMMY_FG_TYPE:
                        continue
                    count = fg.get("count", 0)
                    if count > 0 and ring_multiplier > 0:
                        total = count * ring_multiplier
                        summary[fg["type"]] = summary.get(fg["type"], 0) + total

            # --- Loose groups ('all-*' placements have no fixed count) ---
            for fg in func.get("loose", {}).get("groups", []):
                fg_type = fg["type"]
                if str(fg.get("loose-placement", "")).startswith("all"):
                    summary[fg_type] = "all"
                    continue
                count = fg.get("count", 0)
                if count > 0 and summary.get(fg_type) != "all":
                    summary[fg_type] = summary.get(fg_type, 0) + count

            # --- Build string with x notation ---
            fg_parts = []
            for k, v in sorted(summary.items()):
                fg_parts.append(f"{sanitize(k)}x{v}")

            if fg_parts:
                parts.append("_".join(fg_parts))

            # --- Terminal hydrogenation flag ---
            if term.get("term-hydrogenate"):
                parts.append("Hterm")

        # --- Solvation ---
        if j["system"].get("solvation"):
            sol_parts = []

            for sol in j["solvation"]["groups"]:
                t = sanitize(sol.get("type"))
                count = sol.get("count", None)

                if sol.get("placement") == "ionic" and sol.get("charge-neutrality"):
                    sol_parts.append(f"{t}_neutral")
                elif count is not None and count > 0:
                    sol_parts.append(f"{t}{count}")
                else:
                    sol_parts.append(t)

            if sol_parts:
                # join separately so trailing "_" behavior is clean
                parts.append("_".join(sol_parts))

        return "_".join(parts)
    
    
    # -------------------------------------------------------------------------
    # REPORTING HELPERS
    # -------------------------------------------------------------------------

    def _nanotube_geometry_vectors(self, nt_type):

        if nt_type == "BNNT":
            bond_length = 1.4460
            a1 = np.array([np.sqrt(3)*bond_length, 0.0])
            a2 = np.array([np.sqrt(3)/2*bond_length, 1.5*bond_length])
            return a1, a2, bond_length

        if nt_type in {"MoS2NT", "MoS2"}:
            bond_length = 2.4100
            lattice_constant = 3.1600
            a1 = np.array([lattice_constant, 0.0])
            a2 = np.array([0.5*lattice_constant, np.sqrt(3)/2*lattice_constant])
            return a1, a2, bond_length

        if nt_type in {"MoSSeNT", "MoSSe"}:
            bond_length = 2.4100
            lattice_constant = 3.2300
            a1 = np.array([lattice_constant, 0.0])
            a2 = np.array([0.5*lattice_constant, np.sqrt(3)/2*lattice_constant])
            return a1, a2, bond_length

        bond_length = self.config.get_default_species_bond_param("bond_length", 1.44)
        a1 = np.array([np.sqrt(3)*bond_length, 0.0])
        a2 = np.array([np.sqrt(3)/2*bond_length, 1.5*bond_length])
        return a1, a2, bond_length

    def _nanochannel_summary_profile(self, nt_type):

        default_cutoff = self.config.get_default_species_bond_param("bond_cutoff", 1.60)
        default_profile = {
            "summary_elements": None,
            "expected_coordination": 3,
            "bond_cutoff": default_cutoff,
            "formula_factor": 4,
            "xyz_prefix": None,
            "xyz_outer_chalcogen": False,
        }

        profiles = {
            "BNNT": {
                "bond_cutoff": 1.65,
                "xyz_prefix": "SW-BNNT",
            },
            "MoS2NT": {
                "summary_elements": {"Mo"},
                "expected_coordination": 6,
                "bond_cutoff": 2.95,
                "formula_factor": 2,
                "xyz_prefix": "SW-MoS2NT",
            },
            "MoS2": {
                "summary_elements": {"Mo"},
                "expected_coordination": 6,
                "bond_cutoff": 2.95,
                "formula_factor": 2,
                "xyz_prefix": "SW-MoS2NT",
            },
            "MoSSeNT": {
                "summary_elements": {"Mo"},
                "expected_coordination": 6,
                "bond_cutoff": 3.05,
                "formula_factor": 2,
                "xyz_prefix": "SW-MoSSeNT",
                "xyz_outer_chalcogen": True,
            },
            "MoSSe": {
                "summary_elements": {"Mo"},
                "expected_coordination": 6,
                "bond_cutoff": 3.05,
                "formula_factor": 2,
                "xyz_prefix": "SW-MoSSeNT",
                "xyz_outer_chalcogen": True,
            },
        }

        profile = default_profile.copy()
        profile.update(profiles.get(nt_type, {}))
        return profile

    def _nanochannel_xyz_candidates(self, nt, profile):

        n, m, Ncells = nt['n'], nt['m'], nt['repeats']
        prefix = profile.get("xyz_prefix")
        if prefix is None:
            primary = self.config.default_species_primary or {}
            element = primary.get("atom_element", "C")
            prefix = f"SW-{element}NT"

        suffix = ""
        if profile.get("xyz_outer_chalcogen"):
            suffix = f"_outer-{nt.get('outer-chalcogen', 'Se')}"
        basenames = [f"{prefix}_{n}_{m}_{Ncells}{suffix}.xyz"]

        folders = [
            getattr(self.config, "system_folder", None),
            os.path.dirname(getattr(self.config, "main_file", "") or ""),
            getattr(self.config, "called_from", None),
        ]
        folders = [folder for folder in folders if folder]

        return [
            os.path.join(folder, basename)
            for folder in folders
            for basename in basenames
        ]

    def _read_xyz_records(self, filename):

        with open(filename, "r") as f:
            lines = f.readlines()

        records = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                records.append({
                    "element": parts[0],
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                })
            except ValueError:
                continue
        return records

    def _nanochannel_summary_indices(self, records, profile):

        summary_elements = profile.get("summary_elements")
        if summary_elements is None:
            return list(range(len(records)))
        return [
            i for i, record in enumerate(records)
            if record["element"] in summary_elements
        ]

    def _nanochannel_count_pair(self, record_i, record_j, profile):

        summary_elements = profile.get("summary_elements")
        if summary_elements is None:
            return True
        return not (
            record_i["element"] in summary_elements
            and record_j["element"] in summary_elements
        )

    def _nanochannel_counts_from_xyz(self, records, profile):

        summary_indices = self._nanochannel_summary_indices(records, profile)
        if not summary_indices:
            return None

        positions = np.array(
            [[record["x"], record["y"], record["z"]] for record in records],
            dtype=float,
        )
        bond_cutoff = float(profile["bond_cutoff"])
        expected_coordination = int(profile["expected_coordination"])

        degrees = {}
        for i in summary_indices:
            degree = 0
            for j, record_j in enumerate(records):
                if i == j:
                    continue
                if not self._nanochannel_count_pair(records[i], record_j, profile):
                    continue
                if np.linalg.norm(positions[i] - positions[j]) <= bond_cutoff:
                    degree += 1
            degrees[i] = degree

        rim_indices = [
            i for i, degree in degrees.items()
            if degree < expected_coordination
        ]
        ring_atoms = None
        if rim_indices:
            midpoint = float(np.mean(positions[summary_indices, 2]))
            entry = sum(1 for i in rim_indices if positions[i, 2] < midpoint)
            exit_ = len(rim_indices) - entry
            ring_atoms = (entry, exit_)

        return {
            "channel_atoms": len(summary_indices),
            "ring_atoms": ring_atoms,
        }

    def _nanochannel_formula_atom_count(self, nt, profile):

        n, m, Ncells = nt['n'], nt['m'], nt['repeats']
        dR = self.geometry.gcd(2*n + m, 2*m + n)
        unit_area = n*n + n*m + m*m
        return int(round(profile["formula_factor"] * unit_area / dR * Ncells))

    def _nanochannel_summary_counts(self, nt):

        profile = self._nanochannel_summary_profile(nt['type'])
        counts = None

        for filename in self._nanochannel_xyz_candidates(nt, profile):
            if not os.path.isfile(filename):
                continue
            try:
                counts = self._nanochannel_counts_from_xyz(
                    self._read_xyz_records(filename),
                    profile,
                )
            except OSError:
                counts = None
            if counts is not None:
                break

        if counts is None:
            counts = {"channel_atoms": None, "ring_atoms": None}

        if counts["channel_atoms"] is None:
            counts["channel_atoms"] = self._nanochannel_formula_atom_count(nt, profile)
        if counts["ring_atoms"] is None:
            fallback_ring_atoms = int(nt['n']) + int(nt['m'])
            counts["ring_atoms"] = (fallback_ring_atoms, fallback_ring_atoms)

        return counts

    def _format_nanochannel_ring_atoms(self, ring_atoms):

        entry, exit_ = ring_atoms
        if entry == exit_:
            return str(entry)
        return f"[{entry}, {exit_}]"

    def _log_cnt_geometry(self, n, m, Ncells, n_atoms=None):

        nt_type  = self.json_input['nanotube']['type']
        a1, a2, bond_length = self._nanotube_geometry_vectors(nt_type)

        Ch_len = np.linalg.norm(n*a1 + m*a2)
        R      = Ch_len / (2 * np.pi)

        dR     = self.geometry.gcd(2*n + m, 2*m + n)
        t1, t2 = (2*m + n) // dR, -(2*n + m) // dR
        T_len  = np.linalg.norm(t1*a1 + t2*a2)
        length = Ncells * T_len

        chiral_angle = np.degrees(np.arctan2(np.sqrt(3)*m, 2*n + m))

        if   m == 0: ctype = "zigzag"
        elif n == m: ctype = "armchair"
        else:        ctype = "chiral"

        element = self.config.default_species_primary["atom_element"]

        msg = (
            f"\n{'='*60}\n"
            f"  {nt_type} GEOMETRY  {element}({n},{m}) x {Ncells}\n"
            f"{'='*60}\n"
            f"  Type          : {ctype}\n"
            f"  Radius        : {R:.3f} Ang\n"
            f"  Length        : {length:.3f} Ang\n"
            f"  Chiral angle  : {chiral_angle:.2f} deg\n"
            f"  Bond length   : {bond_length:.3f} Ang\n"
            f"  Atoms total   : {n_atoms}\n"
            f"{'='*60}"
        )
        my_log_file.info(msg)
        print(msg)

    def _log_fg_distribution(self, fg_types, fg_idx, fg_pos):

        '''
        Log the distribution of functional groups: total count per type and
        how many landed on the entry end (z < 0) vs the exit end (z >= 0).

        Parameters:
        ----------
        fg_types : array-like of str
            Functional group type for each attachment site.
        fg_idx : np.ndarray
            LAMMPS atom indices of the attachment carbons.
        fg_pos : (M,3) np.ndarray
            Cartesian positions of the attachment sites.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        from collections import Counter
        type_counts = Counter(fg_types)
        fg_pos = np.array(fg_pos)

        out = [f"\n{'='*60}", "  FUNCTIONAL GROUP DISTRIBUTION", f"{'='*60}"]
        for fg_type, total in sorted(type_counts.items()):
            mask    = np.array([t == fg_type for t in fg_types])
            n_entry = int(np.sum(fg_pos[mask, 2] <  0))
            n_exit  = int(np.sum(fg_pos[mask, 2] >= 0))
            out.append(f"  {fg_type:<12s}: {total:3d} total  "
                       f"(entry end: {n_entry}, exit end: {n_exit})")
        out.append(f"{'='*60}")
        msg = "\n".join(out)
        my_log_file.info(msg)
        print(msg)

    def _log_solvation_stats(self, sol, n_atoms_prior, n_atoms_after, n_atoms_added):

        '''
        Log placement results for a solvation group.

        n_atoms_added is the post-cull count of atoms belonging to this solvent
        type (computed at the call site by scanning the Atoms section). The
        system-wide delta (n_atoms_after - n_atoms_prior) is reported separately
        because the cull may also remove atoms from earlier groups or the
        structure itself, so a negative or smaller-than-expected net delta does
        NOT necessarily mean this solvent was culled.

        Parameters:
        ----------
        sol : dict
            Solvation group dict from the JSON input.
        n_atoms_prior : int
            System-wide atom count before this group was placed.
        n_atoms_after : int
            System-wide atom count after placement AND overlap removal.
        n_atoms_added : int
            Post-cull count of atoms belonging to this solvent type. Not the
            number attempted; that figure is internal to write_solvent.

        Returns:
        -------
        None

        Raises:
        ------
        None
        '''

        net_system_delta = n_atoms_after - n_atoms_prior

        # Need this for both count and concentration modes.
        try:
            solvent_mol_lines  = self.file_manager.find_and_extract_subsection(
                self.config.solvents_file, 'Solvent', sol['type'])
            atoms_section      = self.file_manager.find_and_extract_lines(
                solvent_mol_lines, 'Atoms', header=False)
            atoms_per_molecule = len(atoms_section) if atoms_section else 1
        except Exception:
            atoms_per_molecule = 1

        mpu               = sol.get('molecules_per_unit', 1)
        n_kept_molecules  = n_atoms_added / atoms_per_molecule / mpu

        out = [
            f"  Solvent '{sol['type']}': "
            f"{n_atoms_added} atoms kept "
            f"({int(round(n_kept_molecules))} molecules, {atoms_per_molecule} atoms/mol).",
            f"=" * 150 + "\n"
            f"  System net atom delta: {net_system_delta:+d} "
            f"(includes any atoms culled from earlier groups or structure)."
        ]

        #  target: concentration (mol/L) 
        if sol.get('concentration') is not None and self.box_size not in (None, "auto"):
            vol_ang3 = self.solvation._box_volume_ang3(self.box_size)
            vol_L    = self.solvation._ang3_to_liters(vol_ang3)
            NA       = 6.02214076e23  # Avocado number
            achieved = n_kept_molecules / (NA * vol_L)
            out.append(
                f"  Target: {sol['concentration']:.3f} mol/L  |  "
                f"Achieved: {achieved:.3f} mol/L"
                f"\n" + f"=" * 150
            )

        #  target: count 
        if sol.get('count') is not None:
            target   = int(sol['count'])
            achieved = int(round(n_kept_molecules))
            delta    = achieved - target
            sign     = "+" if delta >= 0 else ""
            pct      = (delta / target * 100.0) if target else 0.0
            out.append(
                f"  Target: {target} molecules  |  "
                f"Achieved: {achieved} ({sign}{delta}, {sign}{pct:.1f}%)"
                f"\n" + f"=" * 150
            )

        msg = "\n".join(out)
        my_log_file.info(msg)
        print(msg)

    def _print_charge_summary(self):

        '''
        Print and log a formatted charge summary showing tube charge, solution
        charge, and total system charge. Warns if the total charge exceeds 0.01 e.

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

        total = self.force_field_manager._round_charge(
            self.force_field_manager.get_system_charge())

        out = [f"\n{'='*60}", "  CHARGE SUMMARY", f"{'='*60}"]

        if self.json_input['system']['nanotube']:
            tube = self.force_field_manager._round_charge(
                self.force_field_manager.get_system_charge(residue_index="42"))
            sol  = self.force_field_manager._round_charge(total - tube)
            out.append(f"  Nanotube charge  : {tube:+.6f} e")
            out.append(f"  Solution charge  : {sol:+.6f} e")

        out.append(f"  Total charge     : {total:+.6f} e")

        CHARGE_TOL = 0.01
        if abs(total) > CHARGE_TOL:
            out.append(f"  WARNING: System is NOT charge-neutral!")
            out.append(f"{'='*60}")
            msg = "\n".join(out)
            my_log_file.warning(msg)
        else:
            out.append(f"  System is charge-neutral.")
            out.append(f"{'='*60}")
            msg = "\n".join(out)
            my_log_file.info(msg)
        print(msg)

    def _write_summary_file(self):

        '''
        Write a plain-text .summary file to the output folder covering CNT
        geometry, functionalisation, solvation, box dimensions, and final charges.

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

        path = os.path.join(self.config.system_folder,
                            self.config.system_name + '.summary')
        out = []
        SEP = "=" * 60

        out += [SEP, "  CHIRALITY-KIT RUN SUMMARY",
                f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                SEP, ""]

        out.append(f"Force field : {self.json_input['system']['force-field']}")
        out.append(f"System name : {self.config.system_name}")
        if self.json_input['system'].get('pbc-override', False):
            out.append("PBC override: enabled (requested reduced padding retained)")
        out.append("")

        if self.json_input['system']['nanotube']:
            nt = self.json_input['nanotube']
            n, m, Ncells = nt['n'], nt['m'], nt['repeats']
            a1, a2, _ = self._nanotube_geometry_vectors(nt['type'])
            R      = np.linalg.norm(n*a1 + m*a2) / (2*np.pi)
            dR     = self.geometry.gcd(2*n + m, 2*m + n)
            t1, t2 = (2*m + n)//dR, -(2*n + m)//dR
            length = np.linalg.norm(t1*a1 + t2*a2) * Ncells
            chiral_angle = np.degrees(np.arctan2(np.sqrt(3)*m, 2*n + m))
            if   m == 0: ctype = "zigzag"
            elif n == m: ctype = "armchair"
            else:        ctype = "chiral"
            nanochannel_counts = self._nanochannel_summary_counts(nt)

            out += [f"[Nanotube]",
                    f"  Type          : {nt['type']} ({ctype})",
                    f"  Indices       : ({n},{m})",
                    f"  Repeats       : {Ncells}",
                    f"  Radius        : {R:.4f} Ang",
                    f"  Length        : {length:.4f} Ang",
                    f"  Chiral angle  : {chiral_angle:.3f} deg",
                    f"  Channel atoms : {nanochannel_counts['channel_atoms']}",
                    f"  Ring atoms    : {self._format_nanochannel_ring_atoms(nanochannel_counts['ring_atoms'])}",
                    ""]

        if self.json_input['system']['functionalisation'] and self.json_input['system']['nanotube']:
            fg_cfg = self.json_input['functionalisation']
            out.append("[Functionalisation]")

            # term
            term_cfg = fg_cfg.get('term', {}) or {}
            out.append(f"  term-hydrogenate : {term_cfg.get('term-hydrogenate', False)}")
            out.append(f"  term-start-highest-x : {term_cfg.get('term-start-highest-x', False)}")
            term_groups = [
                g for g in term_cfg.get('groups', [])
                if g.get('count', 0) > 0 and g.get('type') != DUMMY_FG_TYPE
            ]
            if term_groups:
                out.append("  term groups:")
                for fg in term_groups:
                    tag = f" (side {fg.get('side', 'both')})"
                    out.append(f"    {fg['type']:<10s}: {fg['count']}{tag}")

            # rings
            rings_cfg = fg_cfg.get('rings', []) or []
            for r_idx, ring in enumerate(rings_cfg):
                groups_here = [
                    g for g in ring.get('groups', [])
                    if g.get('count', 0) > 0 and g.get('type') != DUMMY_FG_TYPE
                ]
                if not groups_here:
                    continue
                out.append(f"  ring block {r_idx} ({ring.get('ring-placement', '?')}):")
                out.append(f"    ring-count       : {ring.get('ring-count', 0)}")
                out.append(f"    phase-increment  : {ring.get('ring-phase-increment', 0.0):.5f} rad")
                out.append(f"    phase-start      : {ring.get('ring-phase-start-offset', 0.0):.5f} rad")
                out.append(f"    padding          : {ring.get('ring-padding', True)}")
                for fg in groups_here:
                    out.append(f"    {fg['type']:<10s}: {fg['count']}")

            # loose
            loose_groups = (fg_cfg.get('loose', {}) or {}).get('groups', []) or []
            loose_groups = [g for g in loose_groups
                            if g.get('count', 0) > 0
                            or str(g.get('loose-placement', '')).startswith('all')]
            if loose_groups:
                out.append("  loose groups:")
                for fg in loose_groups:
                    mode = fg.get('loose-placement', 'random')
                    if str(mode).startswith('all'):
                        z_from = float(fg.get('z-from', 0.0))
                        z_to   = float(fg.get('z-to', 1.0))
                        out.append(f"    {fg['type']:<10s}: all free atoms, "
                                   f"z {z_from:.2f}-{z_to:.2f} ({mode})")
                    else:
                        out.append(f"    {fg['type']:<10s}: {fg['count']} ({mode})")

            out.append("")

        if self.json_input['system']['solvation']:
            out.append("[Solvation]")
            for sol in self.json_input['solvation']['groups']:
                out.append(f"  {sol['type']:<14s}: {sol['placement']}")
                if sol.get('concentration'):
                    out.append(f"    target conc : {sol['concentration']} mol/L")
                if sol.get('count'):
                    out.append(f"    count       : {sol['count']}")
            out.append("")

        if self.box_size not in (None, "auto"):
            b = self.box_size
            out += ["[Box size]",
                    f"  x : {b[0][0]:.3f} to {b[0][1]:.3f} Ang  ({b[0][1]-b[0][0]:.3f} Ang)",
                    f"  y : {b[1][0]:.3f} to {b[1][1]:.3f} Ang  ({b[1][1]-b[1][0]:.3f} Ang)",
                    f"  z : {b[2][0]:.3f} to {b[2][1]:.3f} Ang  ({b[2][1]-b[2][0]:.3f} Ang)", ""]

        n_atoms = self.file_manager.find_number_of_entries_in_section("Atoms")
        total   = self.force_field_manager._round_charge(
                      self.force_field_manager.get_system_charge())
        out += ["[Final system]", f"  Total atoms   : {n_atoms}",
                f"  Total charge  : {total:+.6f} e"]

        psf_report = getattr(self, "_psf_charge_neutrality_report", None)
        if psf_report:
            if psf_report.get("applied"):
                out.append(
                    "  PSF charge fix: "
                    f"atom {psf_report['psf_atom_index']} "
                    f"{psf_report['atom_type']} "
                    f"residue {psf_report['residue_index']}; "
                    f"q {psf_report['original_charge']:+.6f} -> "
                    f"{psf_report['adjusted_charge']:+.6f}; "
                    f"PSF total {psf_report['original_total']:+.6f} -> "
                    f"{psf_report['adjusted_total']:+.6f} e"
                )
                out.append("  PSF charge fix snapshot:")
                out.extend(f"    {line}" for line in psf_report.get("snapshot", []))
            else:
                out.append(
                    "  PSF charge fix: skipped; "
                    f"PSF total {psf_report['original_total']:+.6f} e; "
                    f"{psf_report.get('reason', 'no adjustment made')}"
                )

        if self.json_input['system']['nanotube']:
            tube = self.force_field_manager._round_charge(
                self.force_field_manager.get_system_charge(residue_index="42"))
            out.append(f"  Tube charge   : {tube:+.6f} e")

        out += ["", SEP]

        with open(path, 'w') as f:
            f.write("\n".join(out) + "\n")

        my_log_file.info(f"Summary written to {path}")
        print(f"\nSummary written to: {path}")

    def _convert_to_base36(self, n):

        '''
        Convert a non-negative integer to a base-36 string using digits 0-9
        followed by uppercase letters A-Z. Used to generate compact, unique
        atom-name suffixes for nanotube carbons in PDB output (e.g. C0, CA, CZ).

        Parameters:
        ----------
        n : int
            Non-negative integer to convert.

        Returns:
        -------
        label : str
            Base-36 string representation.

        Raises:
        ------
        None
        '''

        symbols = list(string.digits + string.ascii_uppercase)
        base = len(symbols)

        label = ''
        if n == 0:
            return '0'
        while n:             
            label = symbols[n % base] + label
            n = n // base
        return label
    
    def _check_pbc_box(self):

        '''
        Check that the z-padding between the nanotube and the box boundary is
        sufficient to avoid periodic image interactions (minimum 10 Ang). Logs
        a warning if either side is too short.

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

        if self.box_size in (None, "auto") or not self.json_input['system']['nanotube']:
            return

        min_pad = 10.0

        atom_pos, _, atom_types, _ = self.force_field_manager.get_atom_positions_all()
        tube_types = set(self.config.default_species_by_type)
        tube_mask = np.array([t in tube_types for t in atom_types])
        if not tube_mask.any():
            return

        z_tube_min = float(np.min(atom_pos[tube_mask, 2]))
        z_tube_max = float(np.max(atom_pos[tube_mask, 2]))
        z_box_lo   = self.box_size[2][0]
        z_box_hi   = self.box_size[2][1]
        pad_lo     = z_tube_min - z_box_lo
        pad_hi     = z_box_hi  - z_tube_max

        out = [f"\n{'='*60}", "  PBC BOX CHECK", f"{'='*60}",
               f"  Tube z-extent : {z_tube_min:.2f} to {z_tube_max:.2f} Ang",
               f"  Box z-extent  : {z_box_lo:.2f} to {z_box_hi:.2f} Ang",
               f"  Padding lo    : {pad_lo:.2f} Ang",
               f"  Padding hi    : {pad_hi:.2f} Ang"]

        warn = (pad_lo < min_pad or pad_hi < min_pad)
        if warn:
            out.append(f"  WARNING: z-padding < {min_pad} Ang: "
                       f"periodic image interactions likely!")
        else:
            out.append("  Box z-padding OK.")
        out.append(f"{'='*60}")
        msg = "\n".join(out)
        my_log_file.warning(msg) if warn else my_log_file.info(msg)
        print(msg)

    def _resolve_auto_box_padding(self):

        '''
        Resolve automatic box padding, enforcing the recommended padding unless
        pbc-override is explicitly enabled. Override keeps user-provided small
        boxes but still logs risk warnings.

        Returns:
        -------
        tuple[float, float]
            The xy and z padding values to use for automatic box construction.
        '''

        recommended_pad = 14.0
        system_cfg = self.json_input['system']
        PBC_pad_xy = float(system_cfg.get('size-padding-xy', recommended_pad))
        PBC_pad_z = float(system_cfg.get('size-padding-z', recommended_pad))
        pbc_override = system_cfg.get('pbc-override', False) is True

        if pbc_override:
            if PBC_pad_xy < recommended_pad:
                my_log_file.warning(
                    f"XY Padding for box size is below the recommended "
                    f"{recommended_pad} Angstroms because pbc-override is enabled; "
                    "not increasing."
                )
            if PBC_pad_z < recommended_pad:
                my_log_file.warning(
                    f"Z Padding for box size is below the recommended "
                    f"{recommended_pad} Angstroms because pbc-override is enabled; "
                    "not increasing."
                )
            return PBC_pad_xy, PBC_pad_z

        if PBC_pad_xy < recommended_pad:
            my_log_file.warning(
                f"XY Padding for box size too small for non-bonded interactions, "
                f"increasing to {recommended_pad} Angstroms."
            )
            PBC_pad_xy = recommended_pad
        if PBC_pad_z < recommended_pad:
            my_log_file.warning(
                f"Z Padding for box size too small for non-bonded interactions, "
                f"increasing to {recommended_pad} Angstroms."
            )
            PBC_pad_z = recommended_pad

        return PBC_pad_xy, PBC_pad_z


    # -------------------------------------------------------------------------
    # VALIDATION HELPERS
    # -------------------------------------------------------------------------

    def _validate_savepath(self, og_path):

        '''
        Return the file path to use for saving an output file. Currently acts
        as a passthrough; intended as an override point for subclasses that need
        to redirect output to a different location.

        Parameters:
        ----------
        og_path : str
            The original proposed output file path.

        Returns:
        -------
        og_path : str
            The same path, unchanged.

        Raises:
        ------
        None
        '''

        return og_path 

    def _validate_input(self, force_field, nanotube, functionalisation, solvation):

        '''
        Validate all user-facing fields in self.json_input against the lists of
        supported options declared in Config.  Checks are ordered:

            1. Force-field name -> Config.supported_force_fields
            (must pass before any file-existence check, because the path
            depends on the name)
            2. Force-field folder exists on disk
            3. chirality_kit_fragments.xdata and chirality_kit_solvents.xdata
            exist inside that folder
            4. Every functional-group type -> fragments file section headers
            5. Every solvent / ion type    -> solvents file section headers

        Raises
        ------
        ValueError
            If any named option is not in the supported list.
        FileNotFoundError
            If the force-field folder or its data files are missing.
        '''

        # ------------------------------------------------------------------
        # 0. Optional top-level seed
        # ------------------------------------------------------------------
        seed_value = self.json_input.get('seed')

        # ------------------------------------------------------------------
        # 1. Force-field name
        # ------------------------------------------------------------------
        ff = self.json_input['system']['force-field'].lower()

        if seed_value is not None:
            try:
                int(seed_value)
            except (TypeError, ValueError):
                msg = (
                    f"\n{'='*60}\n"
                    f"  VALIDATION ERROR: Seed\n"
                    f"{'='*60}\n"
                    f"  Attempted  : {seed_value!r}\n"
                    f"  Status     : NOT AN INTEGER\n"
                    f"  Expected   : an integer, or omit the key for a random seed\n"
                    f"{'='*60}"
                )
                my_log_file.error(msg)
                raise ValueError(msg)

        if nanotube:
            nt = self.json_input.get('nanotube', {})
            n, m, repeats = (nt.get(key) for key in ('n', 'm', 'repeats'))
            valid_indices = (
                type(n) is int
                and type(m) is int
                and n >= m >= 0
                and n + m >= 3
            )
            if not valid_indices or type(repeats) is not int or repeats < 1:
                msg = (
                    f"\n{'='*60}\n"
                    f"  VALIDATION ERROR: Nanotube Geometry\n"
                    f"{'='*60}\n"
                    f"  Attempted  : ({n}, {m}) x {repeats}\n"
                    f"  Status     : IMPOSSIBLE NANOTUBE\n"
                    f"  Expected   : integer n >= m >= 0, n + m >= 3,\n"
                    f"               and integer repeats >= 1\n"
                    f"{'='*60}"
                )
                my_log_file.error(msg)
                raise ValueError(msg)

        available_ffs = [d for d in os.listdir(self.config.force_fields_root)
                        if os.path.isdir(os.path.join(self.config.force_fields_root, d))]

        if ff.lower() not in [f.lower() for f in available_ffs]:
            msg = (
                f"\n{'='*60}\n"
                f"  VALIDATION ERROR: Force Field\n"
                f"{'='*60}\n"
                f"  Attempted  : {ff}\n"
                f"  Status     : NOT FOUND\n"
                f"  Available  : {', '.join(sorted(available_ffs))}\n"
                f"{'='*60}"
            )
            my_log_file.error(msg)
            raise ValueError(msg)

        # ------------------------------------------------------------------
        # 2. Force-field folder
        # ------------------------------------------------------------------
        ff_folder = os.path.join(self.config.force_fields_root, ff.lower())

        if not os.path.isdir(ff_folder):
            msg = (
                f"\n{'='*60}\n"
                f"  VALIDATION ERROR: Force Field Folder\n"
                f"{'='*60}\n"
                f"  Attempted  : {ff}\n"
                f"  Status     : FOLDER NOT FOUND\n"
                f"  Expected   : {ff_folder}\n"
                f"{'='*60}"
            )
            my_log_file.error(msg)
            raise FileNotFoundError(msg)

        # ------------------------------------------------------------------
        # 3. Data files inside that folder
        # ------------------------------------------------------------------
        fragments_path = os.path.join(ff_folder, "chirality_kit_fragments.xdata")
        solvents_path  = os.path.join(ff_folder, "chirality_kit_solvents.xdata")

        if nanotube:
            if not os.path.isfile(fragments_path):
                msg = (
                    f"\n{'='*60}\n"
                    f"  VALIDATION ERROR: Fragments File\n"
                    f"{'='*60}\n"
                    f"  Force field : {ff}\n"
                    f"  Status      : FRAGMENTS FILE NOT FOUND\n"
                    f"  Expected    : {fragments_path}\n"
                    f"{'='*60}"
                )
                my_log_file.error(msg)
                raise FileNotFoundError(msg)

        if solvation:
            if not os.path.isfile(solvents_path):
                msg = (
                    f"\n{'='*60}\n"
                    f"  VALIDATION ERROR: Solvents File\n"
                    f"{'='*60}\n"
                    f"  Force field : {ff}\n"
                    f"  Status      : SOLVENTS FILE NOT FOUND\n"
                    f"  Expected    : {solvents_path}\n"
                    f"{'='*60}"
                )
                my_log_file.error(msg)
                raise FileNotFoundError(msg)

        # ------------------------------------------------------------------
        # 4. check functional-group names
        # ------------------------------------------------------------------
        nt_type = self.json_input.get('nanotube', {}).get('type')
        if nanotube and nt_type in {"MoSSeNT", "MoSSe"}:
            outer_chalcogen = self.json_input.get('nanotube', {}).get('outer-chalcogen', 'Se')
            if outer_chalcogen not in {"S", "Se"}:
                msg = (
                    f"\n{'='*60}\n"
                    f"  VALIDATION ERROR: MoSSeNT\n"
                    f"{'='*60}\n"
                    f"  outer-chalcogen : {outer_chalcogen}\n"
                    f"  Status          : INVALID\n"
                    f"  Expected        : 'S' or 'Se'\n"
                    f"{'='*60}"
                )
                my_log_file.error(msg)
                raise ValueError(msg)

        if nanotube and functionalisation and nt_type in {"BNNT", "MoS2NT", "MoS2", "MoSSeNT", "MoSSe"}:
            msg = (
                f"\n{'='*60}\n"
                f"  VALIDATION ERROR: Functionalisation\n"
                f"{'='*60}\n"
                f"  Nanotube type : {nt_type}\n"
                f"  Status        : NOT IMPLEMENTED\n"
                f"  Reason        : multi-species functionalisation requires\n"
                f"                  material-aware attachment-site logic.\n"
                f"{'='*60}"
            )
            my_log_file.error(msg)
            raise NotImplementedError(msg)

        if self.json_input['system'].get('functionalisation', False):
            available_fragments = set()
            with open(fragments_path, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("Fragment "):
                        available_fragments.add(stripped.split(None, 1)[1].strip())

            func = self.json_input.get('functionalisation', {})

            # Terminating Groups
            for group in func.get('term', {}).get('groups', []):
                fg_type = group['type']
                count = int(group['count'])
                unsupported = set(group) - TERM_GROUP_FIELDS
                if unsupported:
                    raise ValueError(
                        f"Unsupported term group field(s): "
                        f"{', '.join(sorted(unsupported))}."
                    )
                side = group.get('side', 'both')
                if side not in TERM_SIDES:
                    raise ValueError(
                        f"Invalid term side {side!r}; expected 'both', '-', or '+'."
                    )

                if (fg_type != DUMMY_FG_TYPE
                        and fg_type not in available_fragments
                        and count > 0):
                    msg = (
                        f"\n{'='*60}\n"
                        f"  VALIDATION ERROR: Functional Group Term\n"
                        f"{'='*60}\n"
                        f"  Force field : {ff}\n"
                        f"  Attempted   : {fg_type}\n"
                        f"  Status      : NOT FOUND in fragments file\n"
                        f"  Available   : {', '.join(sorted(available_fragments))}\n"
                        f"{'='*60}"
                    )
                    my_log_file.error(msg)
                    raise ValueError(msg)
                
            # hydrogenation needs the H-term fragment
            if func.get('term', {}).get('term-hydrogenate', False):
                if 'H-term' not in available_fragments:
                    msg = (
                        f"\n{'='*60}\n"
                        f"  VALIDATION ERROR: Term Hydrogenate\n"
                        f"{'='*60}\n"
                        f"  Force field : {ff}\n"
                        f"  Status      : 'H-term' fragment NOT FOUND\n"
                        f"  Reason      : term-hydrogenate is True but this force\n"
                        f"                field has no 'H-term' fragment defined.\n"
                        f"  Fix         : set term-hydrogenate to False, or add an\n"
                        f"                'H-term' Fragment entry to:\n"
                        f"                {fragments_path}\n"
                        f"  Available   : {', '.join(sorted(available_fragments))}\n"
                        f"{'='*60}"
                    )
                    my_log_file.error(msg)
                    raise ValueError(msg)

            # Ring Groups
            for ring in func.get('rings', []):
                for group in ring.get('groups', []):
                    fg_type = group['type']
                    count = int(group['count'])

                    if (fg_type != DUMMY_FG_TYPE
                            and fg_type not in available_fragments
                            and count > 0):
                        msg = (
                            f"\n{'='*60}\n"
                            f"  VALIDATION ERROR: Functional Group Ring\n"
                            f"{'='*60}\n"
                            f"  Force field : {ff}\n"
                            f"  Attempted   : {fg_type}\n"
                            f"  Status      : NOT FOUND in fragments file\n"
                            f"  Available   : {', '.join(sorted(available_fragments))}\n"
                            f"{'='*60}"
                        )
                        my_log_file.error(msg)
                        raise ValueError(msg)

            # Loose Groups
            for group in func.get('loose', {}).get('groups', []):
                fg_type = group['type']
                count = int(group.get('count', 0))
                mode = str(group.get('loose-placement', 'random'))

                z_from = float(group.get('z-from', 0.0))
                z_to   = float(group.get('z-to', 1.0))
                if not (0.0 <= z_from < z_to <= 1.0):
                    msg = (
                        f"\n{'='*60}\n"
                        f"  VALIDATION ERROR: Functional Group Loose\n"
                        f"{'='*60}\n"
                        f"  Attempted   : {fg_type}\n"
                        f"  Status      : INVALID z-window\n"
                        f"  Reason      : need 0 <= z-from < z-to <= 1\n"
                        f"                (got z-from={z_from}, z-to={z_to})\n"
                        f"{'='*60}"
                    )
                    my_log_file.error(msg)
                    raise ValueError(msg)

                if fg_type not in available_fragments and (count > 0 or mode.startswith('all')):
                    msg = (
                        f"\n{'='*60}\n"
                        f"  VALIDATION ERROR: Functional Group Loose\n"
                        f"{'='*60}\n"
                        f"  Force field : {ff}\n"
                        f"  Attempted   : {fg_type}\n"
                        f"  Status      : NOT FOUND in fragments file\n"
                        f"  Available   : {', '.join(sorted(available_fragments))}\n"
                        f"{'='*60}"
                    )
                    my_log_file.error(msg)
                    raise ValueError(msg)
                

        # ------------------------------------------------------------------
        # 5. check solvent and ion names
        # ------------------------------------------------------------------
        if self.json_input['system'].get('solvation', False):
            available_solvents = set()
            with open(solvents_path, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("Solvent "):
                        available_solvents.add(stripped.split(None, 1)[1].strip())

            for group in self.json_input.get('solvation', {}).get('groups', []):
                sol_type = group['type']

                if group.get('placement') == 'ionic':
                    # split the salt formula into its ions
                    cathode_type, _, anode_type, _ = self.solvation._process_salt_formula(sol_type)
                    ions_to_check = [cathode_type, anode_type]
                else:
                    ions_to_check = [sol_type]

                for ion in ions_to_check:
                    if ion not in available_solvents:
                        if group.get('placement') == 'ionic':
                            attempted_str = f"{sol_type}  ->  checking ion '{ion}'"
                        else:
                            attempted_str = ion

                        msg = (
                            f"\n{'='*60}\n"
                            f"  VALIDATION ERROR: Solvent / Ion\n"
                            f"{'='*60}\n"
                            f"  Force field : {ff}\n"
                            f"  Attempted   : {attempted_str}\n"
                            f"  Status      : NOT FOUND in solvents file\n"
                            f"  Available   : {', '.join(sorted(available_solvents))}\n"
                            f"{'='*60}"
                        )
                        my_log_file.error(msg)
                        raise ValueError(msg)

            # ------------------------------------------------------------------
            # 6. pick the bulk phase
            # ------------------------------------------------------------------
            # Only the highest-priority packed group skips self-overlap checks.
            # Ion groups still need the check, and tied bulk priorities are ambiguous.
            solvation_groups = sorted(
                self.json_input.get('solvation', {}).get('groups', []),
                key=lambda g: (int(g['priority']), str(g['type'])),
            )
            packed_groups = [g for g in solvation_groups if g.get('placement') == 'packed']

            if not packed_groups:
                self.config.bulk_priority = None
                my_log_file.info(
                    "Bulk phase: none; no packed solvation group, so every overlapping "
                    "pair is checked and resolved by priority."
                )
            else:
                bulk_priority = max(int(g['priority']) for g in packed_groups)
                holders = [g for g in solvation_groups if int(g['priority']) == bulk_priority]

                if len(holders) > 1:
                    listed = ", ".join(
                        f"'{g['type']}' ({g.get('placement', 'packed')})" for g in holders
                    )
                    msg = (
                        f"\n{'='*60}\n"
                        f"  VALIDATION ERROR: Bulk Phase\n"
                        f"{'='*60}\n"
                        f"  Priority    : {bulk_priority}  (the highest in the solvation list)\n"
                        f"  Held by     : {listed}\n"
                        f"  Status      : AMBIGUOUS; more than one group claims the bulk\n"
                        f"  Reason      : the highest-priority packed group is the bulk\n"
                        f"                phase, and its molecules are exempt from the\n"
                        f"                overlap check because Bridson placement already\n"
                        f"                sets their spacing. Only one group can hold that\n"
                        f"                exemption: shared, the two would never be checked\n"
                        f"                against each other and would interpenetrate.\n"
                        f"                An ionic group can never hold it; ionic placement\n"
                        f"                does not guarantee its own spacing.\n"
                        f"  Fix         : give all but one of them a lower priority number.\n"
                        f"                A lowered group is culled against the bulk, so keep\n"
                        f"                it dilute; a dense second solvent would need a true\n"
                        f"                co-bulk phase, which is not implemented.\n"
                        f"{'='*60}"
                    )
                    my_log_file.error(msg)
                    raise ValueError(msg)

                self.config.bulk_priority = bulk_priority
                my_log_file.info(
                    f"Bulk phase: '{holders[0]['type']}' at priority {bulk_priority} (packed). "
                    f"Its molecules are exempt from the overlap check; Bridson placement sets "
                    f"their spacing. Every other pair is checked and resolved by priority."
                )

    def _validate_fg_available_carbons(self, fg_config, n_atoms):

        '''
        Validate that there are enough available carbons on the nanotube to
        '''

        n_fgs = 0 

        # Terminating Groups
        for group in fg_config.get('term', {}).get('groups', []):
            fg_type = group['type']
            count = int(group['count'])
            n_fgs += count

        # Ring Groups
        for ring in fg_config.get('rings', []):
            for group in ring.get('groups', []):
                fg_type = group['type']
                count = int(group['count'])
            n_fgs += count

        # all-* placement is bounded by atom count, so it doesn't use this budget
        for group in fg_config.get('loose', {}).get('groups', []):
            fg_type = group['type']
            if str(group.get('loose-placement', 'random')).startswith('all'):
                continue
            count = int(group.get('count', 0))
            n_fgs += count

        if n_fgs > n_atoms:
            msg = (
                f"\n{'='*60}\n"
                f"  VALIDATION ERROR: Functional Group Count\n"
                f"{'='*60}\n"
                f"  Attempted   : {n_fgs} functional groups\n"
                f"  Available   : {n_atoms} carbon atoms in the nanotube\n"
                f"  Status      : NOT ENOUGH AVAILABLE CARBONS\n"
                f"  Fix         : reduce the total count of functional groups, or\n"
                f"                increase the nanotube size (n, m, repeats)\n"
                f"{'='*60}"
            )
            my_log_file.error(msg)
            raise ValueError(msg)

        return True 
   
   
    # -------------------------------------------------------------------------
    # MD DATA FILES
    # -------------------------------------------------------------------------

    def write_data_file(self):

        '''
        Write a LAMMPS .data file from the main xdata file by stripping all
        comment fields and non-ASCII characters. The result is saved alongside
        the other output files in the system folder.

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

        src = self.config.main_file
        lammps_file = src[:-6] + ".data"
        lammps_file = self._validate_savepath(lammps_file)

        shutil.copy(src, lammps_file)

        with open(lammps_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        text = re.sub(r"#.*$", "", text, flags=re.MULTILINE)

        text = "".join(
            ch for ch in text 
            if ch in ("\t", "\n", "\r") or 0x20 <= ord(ch) <= 0x7E
        )

        if text and not text.endswith("\n"):
            text += "\n"

        with open(lammps_file, "w", encoding="utf-8") as f:
            f.write(text)

        my_log_file.info(f"File type .data file written to {lammps_file}")

    def write_xyz_file(self): 

        '''
        Write an XYZ-format coordinate file from the Atoms section of the main
        file. Each atom's element symbol is resolved from its type name and the
        output includes the standard two-line XYZ header.

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

        src = self.config.main_file
        filename = src[:-6] + ".xyz"
        filename = self._validate_savepath(filename)

        position_lines = self.file_manager.find_and_extract_lines(self.config.main_file, "Atoms", header=False)

        with open(filename, 'w') as f:
            f.write(f"{len(position_lines)}\n")
            f.write("XYZ generated with chirality_kit\n")
            for line in position_lines:
                parts = self.file_manager.line_to_array(line)
                x, y, z = float(parts[4]), float(parts[5]), float(parts[6])

                atom_type = parts[-1]
                element = self.force_field_manager.get_atom_element(atom_type)  
                f.write(f"{element:<3s} {x:>12.5f} {y:>12.5f} {z:>12.5f}\n")

        my_log_file.info(f"File type .xyz file written to {filename}")

    def write_pdb_file(self):

        '''
        Write a PDB-format coordinate file from the main xdata file. Residue
        names, segment IDs, and B-factor fields are assigned based on atom type:
        nanotube carbons receive residue 'TUBE' and B-factor 1.00; ions keep
        their CHARMM atom/residue names and use their net charge as the B-factor;
        Drude particles and lone pairs use B-factor 0.00.

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

        src = self.config.main_file
        pdb_file = src[:-6] + ".pdb"
        pdb_file = self._validate_savepath(pdb_file)

        position_lines = self.file_manager.find_and_extract_lines(self.config.main_file, "Atoms", header=False)

        with open(pdb_file, 'w') as f:
            f.write(f"REMARK  GENERATED WITH CHIRALITY-KIT\n")
            f.write(f"REMARK  SETUP PERIODIC BOUNDARY CONDITION\n")
            f.write(f"REMARK  DATE {time.strftime('%Y-%m-%d')}\n")
            
            tube_species_counts = defaultdict(int)
            atom_count = 1
            previous_core_props = None

            for line_number, line in enumerate(position_lines):
                parts = self.file_manager.line_to_array(line)
                atom_index, residue_index, charge, atom_type = int(parts[0]), int(parts[1]), float(parts[3]), parts[-1]
                x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                original_atom_type = atom_type

                atom_type_parts = atom_type.split('-')
                is_drude_particle = (
                    (len(atom_type_parts) > 1 and atom_type_parts[1] == "DRUD")
                    or atom_type in self.config.dion_particles
                )
                is_ion_core = atom_type in self.config.ions or atom_type in self.config.dions
                is_ion_particle = atom_type in self.config.dion_particles

                if len(atom_type_parts) > 1 and atom_type_parts[1] != "DRUD":
                    element = self.force_field_manager.get_atom_element(atom_type)
                    res_name = atom_type_parts[1]
                    atom_type = atom_type_parts[0]

                elif len(atom_type_parts) > 1 and atom_type_parts[1] == "DRUD":
                    element = self.force_field_manager.get_atom_element(atom_type)
                    res_name = "IONS" if atom_type_parts[0] in self.config.dions else "TUBE"
                    atom_type = "DRUD"

                else:
                    element = self.force_field_manager.get_atom_element(atom_type)
                    if (atom_type in self.config.ions
                            or atom_type in self.config.dions
                            or atom_type in self.config.dion_particles):
                        res_name = "IONS"
                    else:
                        res_name = "TUBE"

                if is_drude_particle and previous_core_props is not None:
                    residue_index, res_name = previous_core_props

                atom_name = element # for other atoms

                species_entry = self.config.default_species_by_type.get(original_atom_type)
                if (species_entry is not None
                        and not self.config._default_species_entry_is_drude_particle(species_entry)):
                    count = tube_species_counts[original_atom_type]
                    atom_name = species_entry["atom_element"] + str(self._convert_to_base36(count))
                    tube_species_counts[original_atom_type] += 1

                if is_ion_core:
                    atom_name = self.config.dion_pdb_names.get(original_atom_type, original_atom_type)
                    res_name = atom_name
                elif is_ion_particle:
                    atom_name = original_atom_type

                seg_id = "TUBE" if res_name == "TUBE" else "SYS"

                occupancy = 1.00
                temp = 0.00 # default temp factor

                if is_ion_core:
                    temp = charge
                    if line_number + 1 < len(position_lines):
                        next_parts = self.file_manager.line_to_array(position_lines[line_number + 1])
                        if (int(next_parts[1]) == residue_index
                                and next_parts[-1] in self.config.dion_particles):
                            temp += float(next_parts[3])
                elif res_name == "TUBE":
                    temp = 1.00    # set temp factor to 1.00 for tube carbons

                if element in ['DP', 'LP']: # lone pairs or drude particles
                    temp = 0.00

                f.write(
                    "ATOM  {atom_index:5d} {atom_name:^4s} {res_name:>4s} {residue_index:4d}    "
                    "{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp:6.2f}      {seg_id:>5s}\n"
                    .format(
                        atom_index=atom_count,
                        atom_name=atom_name,
                        element=element,
                        res_name=res_name,
                        residue_index=residue_index,
                        x=x, y=y, z=z,
                        occupancy=occupancy,
                        temp=temp,
                        seg_id=seg_id
                    )
                )

                atom_count += 1
                if not is_drude_particle:
                    previous_core_props = (residue_index, res_name)

            f.write("END\n")

        my_log_file.info(f"File type .pdb file written to {pdb_file}")

    def write_psf_file(self): 

        '''
        Write a CHARMM PSF file from the main xdata file. Handles both standard
        and Drude-polarisable topologies (EXT CMAP vs EXT CMAP DRUDE header).
        Sections written: NATOM, NBOND, NTHETA, NPHI, NIMPHI, NDON, NACC, NNB,
        NUMLP/NUMLPH (lone pairs), NUMANISO (anisotropy), and NCRTERM. Raises
        ValueError if a Drude bond coefficient cannot be found for an anisotropic atom.

        Parameters:
        ----------
        None

        Returns:
        -------
        None

        Raises:
        ------
        ValueError
            If the Drude-Drude bond coefficient for an anisotropic atom type
            cannot be found in the Bond Coeffs section.
        '''

        src = self.config.main_file
        psf_file = src[:-6] + ".psf"
        psf_file = self._validate_savepath(psf_file)
        atom_index_corrector = []
        self._psf_charge_neutrality_report = None

        def _base_atom_type(atom_type):
            atom_type_parts = str(atom_type).split('-')
            return atom_type_parts[0], atom_type_parts

        def _is_drude_particle(atom_type):
            base_atom_type, atom_type_parts = _base_atom_type(atom_type)
            return (
                base_atom_type == "DRUD"
                or base_atom_type in self.config.dion_particles
                or (len(atom_type_parts) > 1 and atom_type_parts[1] == "DRUD")
            )

        def _is_ion_core(atom_type):
            base_atom_type, _ = _base_atom_type(atom_type)
            return (
                not _is_drude_particle(atom_type)
                and (
                    base_atom_type in self.config.ions
                    or base_atom_type in self.config.dions
                )
            )

        def _same_sign(a, b):
            return (a > 0 and b > 0) or (a < 0 and b < 0)

        def _make_psf_charge_report(record, original_charge,
                                    adjusted_charge, original_total,
                                    adjusted_total):
            return {
                "applied": True,
                "record_index": record["record_index"],
                "psf_atom_index": record["psf_atom_index"],
                "xdata_atom_index": record["xdata_atom_index"],
                "residue_index": record["residue_index"],
                "atom_type": record["atom_type"],
                "original_charge": float(original_charge),
                "adjusted_charge": float(adjusted_charge),
                "original_total": float(original_total),
                "adjusted_total": float(adjusted_total),
                "snapshot": [],
            }

        def _make_skipped_psf_charge_report(original_total, reason):
            return {
                "applied": False,
                "original_total": float(original_total),
                "reason": reason,
                "snapshot": [],
            }

        def _parse_psf_natom_line(line, record_index, atom_index_corrector):
            parts = line.split()
            psf_atom_index = int(parts[0])
            return {
                "record_index": record_index,
                "psf_atom_index": psf_atom_index,
                "xdata_atom_index": atom_index_corrector[psf_atom_index - 1],
                "residue_index": int(parts[2]),
                "res_name": parts[3],
                "atom_name": parts[4],
                "atom_type": parts[5],
                "charge": float(parts[6]),
            }

        def _replace_psf_charge_field(line, old_charge, new_charge):
            token = f"{old_charge:.6f}"
            token_start = line.find(token, 45)
            if token_start < 0:
                token = str(old_charge)
                token_start = line.find(token, 45)
            if token_start < 0:
                raise ValueError(
                    f"Could not locate PSF charge field for atom line: {line.rstrip()}"
                )
            charge_start = token_start - (14 - len(token))
            charge_end = token_start + len(token)
            return line[:charge_start] + f"{new_charge:14.6f}" + line[charge_end:]

        def _apply_psf_charge_neutrality_patch(psf_file, atom_index_corrector):

            if not self.json_input.get("settings", {}).get("adjust-ion-partial-charge", False):
                return

            with open(psf_file, "r") as f:
                psf_lines = f.readlines()

            natom_header_index = next(
                (i for i, line in enumerate(psf_lines) if "!NATOM" in line),
                None,
            )
            if natom_header_index is None:
                return

            natom_count = int(psf_lines[natom_header_index].split()[0])
            natom_start = natom_header_index + 1
            natom_end = natom_start + natom_count
            records = [
                _parse_psf_natom_line(line, record_index, atom_index_corrector)
                for record_index, line in enumerate(psf_lines[natom_start:natom_end])
            ]

            raw_total = sum(record["charge"] for record in records)
            original_total = self.force_field_manager._round_charge(raw_total)
            if abs(original_total) <= 0.01:
                return

            target_record = None
            target_new_charge = None
            target_adjusted_total = None

            if self.config.drude_polarisable:
                ion_units = []
                for record_index, record in enumerate(records[:-1]):
                    if record["res_name"] != "IONS":
                        continue
                    if not _is_ion_core(record["atom_type"]):
                        continue

                    particle = records[record_index + 1]
                    if particle["res_name"] != "IONS":
                        continue
                    if not _is_drude_particle(particle["atom_type"]):
                        continue

                    ion_units.append({
                        "core_index": record_index,
                        "particle_index": record_index + 1,
                        "net_charge": record["charge"] + particle["charge"],
                    })

                for ion_unit in reversed(ion_units):
                    ion_charge = self.force_field_manager._round_charge(
                        ion_unit["net_charge"]
                    )
                    if not _same_sign(ion_charge, original_total):
                        continue

                    target_record = records[ion_unit["core_index"]]
                    old_charge = target_record["charge"]
                    target_new_charge = old_charge - original_total
                    target_adjusted_total = self.force_field_manager._round_charge(
                        raw_total - old_charge + target_new_charge
                    )
                    break

            else:
                for record_index in range(len(records) - 1, -1, -1):
                    record = records[record_index]
                    atom_type = record["atom_type"]
                    if record["res_name"] != "IONS":
                        continue
                    if not _is_ion_core(atom_type):
                        continue

                    old_charge = record["charge"]
                    if not _same_sign(old_charge, original_total):
                        continue

                    new_charge = old_charge - original_total
                    adjusted_total = self.force_field_manager._round_charge(
                        raw_total - old_charge + new_charge
                    )
                    if abs(new_charge) > abs(old_charge) + 1e-9:
                        continue
                    if abs(adjusted_total) >= abs(original_total):
                        continue

                    target_record = record
                    target_new_charge = new_charge
                    target_adjusted_total = adjusted_total
                    break

            if target_record is not None:
                old_charge = target_record["charge"]
                psf_line_index = natom_start + target_record["record_index"]
                psf_lines[psf_line_index] = _replace_psf_charge_field(
                    psf_lines[psf_line_index],
                    old_charge,
                    target_new_charge,
                )
                self._psf_charge_neutrality_report = _make_psf_charge_report(
                    target_record, old_charge, target_new_charge,
                    original_total, target_adjusted_total
                )
                snapshot_start = max(natom_start, psf_line_index - 3)
                snapshot_end = min(natom_end, psf_line_index + 4)
                self._psf_charge_neutrality_report["snapshot"] = [
                    line.rstrip("\n") for line in psf_lines[snapshot_start:snapshot_end]
                ]

                with open(psf_file, "w") as f:
                    f.writelines(psf_lines)

                system_type = "Drude ion core" if self.config.drude_polarisable else "ion core"
                my_log_file.info(
                    f"PSF charge neutrality adjusted {system_type} "
                    f"{target_record['atom_type']} at PSF atom "
                    f"{target_record['psf_atom_index']}: "
                    f"{old_charge:+.6f} -> {target_new_charge:+.6f}."
                )
                return

            self._psf_charge_neutrality_report = _make_skipped_psf_charge_report(
                original_total,
                "no eligible ion core could be adjusted without violating the charge rules",
            )
            my_log_file.warning(
                "PSF charge neutrality was requested, but no eligible ion core "
                "could be adjusted without violating the charge rules."
            )

        with open(psf_file, 'w') as f:
            
            if self.config.drude_polarisable == False:
                f.write("PSF EXT CMAP\n\n")
            else:
                f.write("PSF EXT CMAP DRUDE\n\n")
            f.write("     0 !NTITLE\n\n")
            
            #!NATOM lines
            lines = self.file_manager.find_and_extract_lines(self.config.main_file, "Atoms", header=False)
            f.write(f"{len(lines):8d} !NATOM\n")
            
            tube_species_counts = defaultdict(int)
            atom_count = 1
            previous_core_props = None

            for line in lines:
                parts = self.file_manager.line_to_array(line)
                atom_index, residue_index, atom_number_type, charge, atom_type = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), str(parts[-1])
                x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                mass = self.force_field_manager.get_atom_mass(atom_number_type)
                original_atom_type = atom_type

                atom_index_corrector.append(atom_index)
                atom_type_parts = atom_type.split('-')
                is_drude_particle = (
                    (len(atom_type_parts) > 1 and atom_type_parts[1] == "DRUD")
                    or atom_type in self.config.dion_particles
                )

                if len(atom_type_parts) > 1 and atom_type_parts[1] != "DRUD":
                    element = self.force_field_manager.get_atom_element(atom_type)
                    alpha, thole = self.force_field_manager._get_drude_coeffs(
                        drude_type=atom_type, atom_number_type=atom_number_type
                    )
                    res_name = atom_type_parts[1]
                    atom_type = atom_type_parts[0]

                elif len(atom_type_parts) > 1 and atom_type_parts[1] == "DRUD":
                    element = self.force_field_manager.get_atom_element(atom_type)
                    alpha, thole = self.force_field_manager._get_drude_coeffs(
                        drude_type=atom_type, atom_number_type=atom_number_type
                    )
                    res_name = "IONS" if atom_type_parts[0] in self.config.dions else "TUBE"

                else:
                    element = self.force_field_manager.get_atom_element(atom_type)
                    alpha, thole = self.force_field_manager._get_drude_coeffs(
                        drude_type=atom_type, atom_number_type=atom_number_type
                    )
                    if (atom_type in self.config.ions
                            or atom_type in self.config.dions
                            or atom_type in self.config.dion_particles):
                        res_name = "IONS"
                    else:
                        res_name = "TUBE"

                if is_drude_particle and previous_core_props is not None:
                    residue_index, res_name = previous_core_props

                atom_name = element
                species_entry = self.config.default_species_by_type.get(original_atom_type)
                if (species_entry is not None
                        and not self.config._default_species_entry_is_drude_particle(species_entry)):
                    count = tube_species_counts[original_atom_type]
                    atom_name = species_entry["atom_element"] + str(self._convert_to_base36(count))
                    tube_species_counts[original_atom_type] += 1

                seg_id = "TUBE" if res_name == "TUBE" else "SYS"

                if self.config.drude_polarisable == False:
                    line_write = (
                        "{:10d} {:<8s} {:8d} {:<8s} {:<8s} {:<6s}{:14.6f}{:14.4f}           0\n".format(
                            atom_count, seg_id, residue_index, res_name, atom_name,
                            atom_type, charge, mass
                        )
                    )
                else:
                    if element == "LP":
                        val = -1
                    else:
                        val = 0

                    if is_drude_particle:
                        atom_type = "DRUD"
                        atom_name = "DP"
                    # Drude ion cores keep their real atom type.

                    line_write = (
                        "{:10d} {:<8s} {:8d} {:<8s} {:<8s} {:<6s}{:14.6f}{:14.4f}          {:2d}  {:12.3f} {:12.3f}\n".format(
                            atom_count, seg_id, residue_index, res_name, atom_name,
                            atom_type, charge, mass, val, alpha, thole
                        )
                    )

                f.write(line_write)
                atom_count += 1
                if not is_drude_particle:
                    previous_core_props = (residue_index, res_name)

            # Map atom ids once instead of scanning the correction list every time.
            atom_index_map = {orig: new + 1 for new, orig in enumerate(atom_index_corrector)}

            #!NBONDS lines
            BOND_INT_WIDTH = 10
            lines = (
                self.file_manager.find_and_extract_lines(self.config.main_file, "Bonds", header=False)
                if self.json_input["settings"]["bonds"] else []
            )
            f.write(f"\n{len(lines):10d} !NBOND: bonds\n")
            for i in range(len(lines))[::4]:
                parts1 = self.file_manager.line_to_array(lines[i]) if i < len(lines) else None
                parts2 = self.file_manager.line_to_array(lines[i+1]) if i+1 < len(lines) else None
                parts3 = self.file_manager.line_to_array(lines[i+2]) if i+2 < len(lines) else None
                parts4 = self.file_manager.line_to_array(lines[i+3]) if i+3 < len(lines) else None
                parts = [parts1, parts2, parts3, parts4]
                line = []
                for part in parts:
                    if part is not None:
                        atom1 = int(part[2])
                        atom2 = int(part[3])

                        atom1_corrected = atom_index_map[atom1]
                        atom2_corrected = atom_index_map[atom2]
                        
                        line.append(atom1_corrected)
                        line.append(atom2_corrected)

                line_write = "".join(f"{v:{BOND_INT_WIDTH}d}" for v in line) + "\n"
                f.write(line_write)

            #!NANGLES lines 
            ANGLE_INT_WIDTH = 10
            lines = (
                self.file_manager.find_and_extract_lines(self.config.main_file, "Angles", header=False)
                if self.json_input["settings"]["angles"] else []
            )
            f.write(f"\n{len(lines):10d} !NTHETA: angles\n")
            for i in range(len(lines))[::3]:
                parts1 = self.file_manager.line_to_array(lines[i]) if i < len(lines) else None
                parts2 = self.file_manager.line_to_array(lines[i+1]) if i+1 < len(lines) else None
                parts3 = self.file_manager.line_to_array(lines[i+2]) if i+2 < len(lines) else None
                parts = [parts1, parts2, parts3]
                line = []
                for part in parts:
                    if part is not None:
                        atom1 = int(part[2])
                        atom2 = int(part[3])
                        atom3 = int(part[4])
                        atom1_corrected = atom_index_map[atom1]
                        atom2_corrected = atom_index_map[atom2]
                        atom3_corrected = atom_index_map[atom3]
                        line.append(atom1_corrected)
                        line.append(atom2_corrected)
                        line.append(atom3_corrected)

                line_write = "".join(f"{v:{ANGLE_INT_WIDTH}d}" for v in line) + "\n"
                f.write(line_write)

            #!NDIHEDRALS lines
            DIHED_INT_WIDTH = 10
            lines = (
                self.file_manager.find_and_extract_lines(self.config.main_file, "Dihedrals", header=False)
                if self.json_input["settings"]["dihedrals"] else []
            )
            f.write(f"\n{len(lines):10d} !NPHI: dihedrals\n")
            for i in range(len(lines))[::2]:
                parts1 = self.file_manager.line_to_array(lines[i]) if i < len(lines) else None
                parts2 = self.file_manager.line_to_array(lines[i+1]) if i+1 < len(lines) else None
                parts = [parts1, parts2]
                line = []
                for part in parts:
                    if part is not None:
                        atom1 = int(part[2])
                        atom2 = int(part[3])
                        atom3 = int(part[4])
                        atom4 = int(part[5])
                        atom1_corrected = atom_index_map[atom1]
                        atom2_corrected = atom_index_map[atom2]
                        atom3_corrected = atom_index_map[atom3]
                        atom4_corrected = atom_index_map[atom4]
                        line.append(atom1_corrected)
                        line.append(atom2_corrected)
                        line.append(atom3_corrected)
                        line.append(atom4_corrected)

                line_write = "".join(f"{v:{DIHED_INT_WIDTH}d}" for v in line) + "\n"
                f.write(line_write)

            #!NIMPROPERS lines
            IMPROPER_INT_WIDTH = 10 
            lines = (
                self.file_manager.find_and_extract_lines(self.config.main_file, "Impropers", header=False)
                if self.json_input["settings"]["impropers"] else []
            )
            f.write(f"\n{len(lines):10d} !NIMPHI: impropers\n")
            for i in range(len(lines))[::2]:
                parts1 = self.file_manager.line_to_array(lines[i]) if i < len(lines) else None
                parts2 = self.file_manager.line_to_array(lines[i+1]) if i+1 < len(lines) else None
                parts = [parts1, parts2]
                line = []
                for part in parts:
                    if part is not None:
                        atom1 = int(part[2])
                        atom2 = int(part[3])
                        atom3 = int(part[4])
                        atom4 = int(part[5])
                        atom1_corrected = atom_index_map[atom1]
                        atom2_corrected = atom_index_map[atom2]
                        atom3_corrected = atom_index_map[atom3]
                        atom4_corrected = atom_index_map[atom4]
                        line.append(atom1_corrected)
                        line.append(atom2_corrected)
                        line.append(atom3_corrected)
                        line.append(atom4_corrected)
                line_write = "".join(f"{v:{IMPROPER_INT_WIDTH}d}" for v in line) + "\n"
                f.write(line_write)

            _ = 0

            f.write(f"\n{_:10d} !NDON: donors\n")

            f.write(f"\n{_:10d} !NACC: acceptors\n")

            f.write(f"\n{_:10d} !NNB\n")

            #!NUMLP NUMLPH lines
            lines = (
                self.file_manager.find_and_extract_lines(self.config.main_file, "Lone Pairs", header=False)
                if self.json_input["settings"]["lone-pairs"] else []
            )
            n_lp    = len(lines)
            n_lph   = 4 * n_lp
            f.write(f"\n{n_lp:10d}{n_lph:10d} !NUMLP NUMLPH\n")
            for i in range(len(lines)):
                parts = self.file_manager.line_to_array(lines[i])
                lp_idx = i*4 + 1 # each lone pair has 4 anisotropic atoms associated with it
                line_write = f"{'3':>10s}{lp_idx:10d}{parts[2]:>3s}{parts[3]:>15s}{parts[4]:>15s}{parts[5]:>15s}" + "\n"
                f.write(line_write) 

            LONEPAIR_INT_WIDTH = 10
            for i in range(len(lines))[::2]:
                parts1 = self.file_manager.line_to_array(lines[i]) if i < len(lines) else None
                parts2 = self.file_manager.line_to_array(lines[i+1]) if i+1 < len(lines) else None
                parts = [parts1, parts2]
                line = []
                for part in parts:
                    if part is not None:
                        dependecies = part[-1].strip("()").split(',')
                        line.append(atom_index_map[int(part[1])]) # lone pair index
                        for dep in dependecies:
                            atom_dep = int(dep)
                            atom_dep_corrected = atom_index_map[atom_dep]
                            line.append(atom_dep_corrected)
                line_write = "".join(f"{v:{LONEPAIR_INT_WIDTH}d}" for v in line) + "\n"
                f.write(line_write)

            #!NUMANISO lines
            if self.config.drude_polarisable: # bricks for non-polarisable sims, anisotropy for polarisable sims
                if self.json_input["settings"]["anisotropy"]:
                    lines = self.file_manager.find_and_extract_lines(
                        self.config.main_file, "Anisotropy", header=False
                    )
                    bond_coefs = self.file_manager.find_and_extract_lines(
                        self.config.main_file, "Bond Coeffs", header=False
                    )
                else:
                    lines = []
                    bond_coefs = []
                f.write(f"\n{len(lines):10d} !NUMANISO\n")
                for i in range(len(lines)):
                    parts = self.file_manager.line_to_array(lines[i])
                    anisotropy_type = parts[-2]
                    drude_type = f"{anisotropy_type.split('-', 1)[0]}-DRUD"
                    Kdrude = None
                    for bond in bond_coefs:
                        bond_parts = self.file_manager.line_to_array(bond)
                        if {anisotropy_type, drude_type} == set(bond_parts[-2:]):
                            Kdrude = float(bond_parts[1])
                            break
                    if Kdrude is None:
                        # parts[-1] is the dependency tuple string; the atom type is parts[-2].
                        my_log_file.error(f"Could not find Drude bond coefficient for anisotropic atom type {parts[-2]}.")
                        raise ValueError(f"Could not find Drude bond coefficient for anisotropic atom type {parts[-2]}.")
                    
                    K11, K22, K33 = self.force_field_manager._get_anisotropic_coeffs_CHARMM36_NDP(A11=parts[2], A22=parts[3], Kdrude=Kdrude)
                    f.write(f"{K11:>20f}{K22:>15f}{K33:>15f}\n")
                
                ANISO_INT_WIDTH = 10
                for i in range(len(lines))[::2]:
                    parts1 = self.file_manager.line_to_array(lines[i]) if i < len(lines) else None
                    parts2 = self.file_manager.line_to_array(lines[i+1]) if i+1 < len(lines) else None
                    parts = [parts1, parts2]
                    line = []
                    for part in parts:
                        if part is not None:
                            dependecies = part[-1].strip("()").split(',')
                            line.append(atom_index_map[int(part[1])])  # atom index
                            for dep in dependecies:
                                atom_dep = int(dep)
                                atom_dep_corrected = atom_index_map[atom_dep]
                                line.append(atom_dep_corrected)
                    line_write = "".join(f"{v:{ANISO_INT_WIDTH}d}" for v in line) + "\n"
                    f.write(line_write)

            f.write(f"\n{_:10d} !NCRTERM: cross-terms\n")

            f.write(f"\n{_:10d} !AUTOGEN\n") # useless, tell CHARMM which angles etc were guessed

        _apply_psf_charge_neutrality_patch(psf_file, atom_index_corrector)

        my_log_file.info(f"File type .psf file written to {psf_file}")

    def write_pdb_restraints_file(self):

        '''
        Generate a positional-restraint reference PDB for the Drude simulation.
        Nanotube carbons receive occupancy and B-factor 1.00 to activate the
        harmonic restraint; all others receive 0.00. Skips if no PDB is found.

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

        spring_constant = 1.00 # how strongly the harmonic potential acts
        B_factor = 1.00 # 1.00 for yes 0.00 for no

        src = self.config.main_file
        pdb_file = src[:-6] + ".pdb"

        if os.path.exists(pdb_file) == False:
            my_log_file.error("PDB file not found for generating restraints.")
            return
        
        restraints_file = os.path.join(self.config.system_folder, f"restraints.ref")
        
        with open(pdb_file, 'r') as f:
            with open(restraints_file, 'w') as rf:
                lines = f.readlines()
                previous_core_line = None
                for line in lines:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        atom_name = line[12:16].strip()
                        if atom_name == "DP" and previous_core_line is not None:
                            atom_index = int(previous_core_line[6:11]) + 1
                            line = (
                                line[:6] + f"{atom_index:5d}" + line[11:17]
                                + previous_core_line[17:26] + line[26:66]
                                + previous_core_line[66:]
                            )
                        res_name = line[17:21].strip()  # residue name (columns 18-20)

                        # it is a carbon in the nanotube, but not the functional group.
                        if atom_name != "DP" and res_name == "TUBE" and "C" in line[12:16].strip() and line[12:16].strip() != "C":

                            # Format occupancy (cols 55-150) and B-factor (cols 61-66)
                            new_occ = f"{B_factor:6.2f}"
                            new_b = f"{spring_constant:6.2f}"
                            line = line[:54] + new_occ + new_b + line[66:]
                        else:
                            new_occ = f"{0.00:6.2f}" # No restraint
                            new_b = f"{0.00:6.2f}"
                            line = line[:54] + new_occ + new_b + line[66:]
                        if atom_name != "DP":
                            previous_core_line = line
                    if "END" in line or "REMARK" in line:
                        continue
                    rf.write(line)

    # -------------------------------------------------------------------------
    # MD FORCE FIELD PARAMETER FILES
    # -------------------------------------------------------------------------

    def copy_other_FF_files(self):

        '''
        Copy all other files apart from the _solvents and _fragments files.

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

        ff_low = self.config.force_field.lower()
        base_folder = os.path.join(self.config.force_fields_root, ff_low)
        dest = os.path.join(self.config.system_folder, os.path.basename(base_folder.rstrip("/")))

        ignore = shutil.ignore_patterns(
            "chirality_kit_solvents.data",
            "chirality_kit_fragments.data"
        )
        shutil.copytree(base_folder, dest, ignore=ignore, dirs_exist_ok=True)

        my_log_file.info(f"{self.config.force_field} other force field files copied to system folder.")


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

if __name__ == "__main__":

    def _parse_args():

        '''
        Parse command-line arguments.

        Four flags, all optional:
            --force-fields <path>   Path to the force-fields directory. Must be
                                    a folder literally named 'force-fields'.
                                    Defaults to ../force-fields next to this script's folder.
            --json          <path>  Path to a JSON input file that overrides the
                                    in-file INPUT dict.
            --seed          <int>   Integer RNG seed for reproducibility. Overrides the SEED constant defined at the top of the file.]
            --out-name      <path>  Output folder path or name. Generated files
                                    use the folder basename.

        Example JSON files are provided in the repository.
        '''

        parser = argparse.ArgumentParser(
            prog="chirality_kit",
            description="Chirality Kit: CNT system builder",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument("--force-fields", metavar="PATH",
                            help="Path to the 'force-fields' directory. The folder "
                                 "must be named exactly 'force-fields'. "
                                 "Defaults to ../force-fields next to this script's folder.")
        
        parser.add_argument("--json", metavar="FILE",
                            help="Path to a JSON input file. Overrides the in-file "
                                 "INPUT dict entirely. See the repo for examples.")
        
        parser.add_argument("--seed", type=int, metavar="N",
                            help="Integer RNG seed for reproducible runs. "
                                 "Overrides the SEED constant defined at the top of the file.")

        parser.add_argument("--out-name", metavar="PATH",
                            help="Output folder path or name. Generated files use "
                                 "the folder basename.")
        
        return parser.parse_args()

    def _resolve_force_fields_path(cli_path, script_location):

        '''
        Return an absolute path to the force-fields directory. Falls back to
        <script_location>/../force-fields if nothing was supplied on the CLI.
        Validates that the resolved path exists, is a directory, and is
        literally named 'force-fields'.
        '''
        
        path = cli_path if cli_path else os.path.join(script_location, "..", "force-fields")
        path = os.path.abspath(path)

        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"  --force-fields path not found or not a directory\n"
                f"{'='*60}\n"
                f"  Supplied : {path}\n"
                f"{'='*60}"
            )
        
        if os.path.basename(path.rstrip(os.sep)) != "force-fields":
            raise ValueError(
                f"\n{'='*60}\n"
                f"  --force-fields path must end in a folder named 'force-fields'\n"
                f"{'='*60}\n"
                f"  Supplied : {path}\n"
                f"  Basename : {os.path.basename(path.rstrip(os.sep))}\n"
                f"{'='*60}"
            )
        return path

    def full_run():

        args = _parse_args()

        # SEEDING
        # Just save the CLI override here; Config.apply_seed sorts out the rest after loading JSON.
        config = Config()
        config.cli_seed = args.seed
        config.out_name = args.out_name

        config.force_fields_root = _resolve_force_fields_path(
            args.force_fields, config.script_location
        )

        json_input = args.json if args.json else INPUT

        file_manager        = File_Manager(config)
        geometry            = Geometry(config)
        force_field_manager = Force_Field_Manager(file_manager, config, geometry)
        functionalisation   = Functional_Group_Generator(force_field_manager, file_manager, config, geometry)
        chirality           = Structure_Generator(force_field_manager, file_manager, config, geometry)
        solvation           = Solution_Generator(force_field_manager, file_manager, config, geometry)

        io = IO(force_field_manager, file_manager,
                chirality, solvation, functionalisation,
                config, geometry, json_input=json_input)
        
        io.run()

    full_run()
