from __future__ import annotations

import logging


class _LoggerProxy:

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def set_logger(self, logger):
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self._logger, name)


my_log_file = _LoggerProxy()


def set_logger(logger):
    my_log_file.set_logger(logger)


SEED = None

# system parameters here or with .json
INPUT = {
    # Handy default seed. --seed wins, then JSON/archive, then this; None picks a fresh one.
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
            "r_floor"        : 1.0,   # Angstrom; hard lower bound on exclusion radius during r-search
            "repeats"        : 2,     # Bridson runs per r evaluation to reduce stochastic variance
            "bracket_steps"  : 10,    # iterations to bracket the feasibility boundary
            "bin_steps"      : 10,    # binary-search refinement steps after bracketing
            "final_repeats"  : 10,    # independent runs at the chosen r; best result is kept
            "tol_frac"       : 0.05,  # +/- fractional tolerance on target count before a warning is raised
            "injection_limit": 1024   # max ion pairs tried during charge-neutrality correction
        }

    }
}
