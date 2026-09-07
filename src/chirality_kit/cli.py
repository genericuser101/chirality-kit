from __future__ import annotations

import argparse
import os

from ._runtime import INPUT
from .config import Config
from .file_manager import File_Manager
from .force_field_manager import Force_Field_Manager
from .functional_group_generator import Functional_Group_Generator
from .geometry import Geometry
from .io import IO
from .solution_generator import Solution_Generator
from .structure_generator import Structure_Generator


def parse_args():
    parser = argparse.ArgumentParser(
        prog="chirality_kit",
        description="Chirality Kit split-package system builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force-fields",
        metavar="PATH",
        help="Path to the 'force-fields' directory. The folder must be named exactly 'force-fields'.",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Path to a JSON input file. Overrides the in-file INPUT dict entirely.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="Integer RNG seed for reproducible runs. Takes precedence over a 'seed' "
             "key in the JSON, over an archived metadata.rng_seed_used, and over the "
             "SEED constant.",
    )
    parser.add_argument(
        "--out-name",
        metavar="PATH",
        help="Output folder path or name. Generated files use the folder basename.",
    )
    return parser.parse_args()


def resolve_force_fields_path(cli_path, script_location):
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
    args = parse_args()

    # Just save the CLI override here; Config.apply_seed sorts out the rest after loading JSON.
    config = Config()
    config.cli_seed = args.seed
    config.out_name = args.out_name
    config.force_fields_root = resolve_force_fields_path(args.force_fields, config.script_location)

    json_input = args.json if args.json else INPUT

    file_manager = File_Manager(config)
    geometry = Geometry(config)
    force_field_manager = Force_Field_Manager(file_manager, config, geometry)
    functionalisation = Functional_Group_Generator(force_field_manager, file_manager, config, geometry)
    chirality = Structure_Generator(force_field_manager, file_manager, config, geometry)
    solvation = Solution_Generator(force_field_manager, file_manager, config, geometry)

    io = IO(
        force_field_manager,
        file_manager,
        chirality,
        solvation,
        functionalisation,
        config,
        geometry,
        json_input=json_input,
    )
    io.run()


if __name__ == "__main__":
    full_run()
