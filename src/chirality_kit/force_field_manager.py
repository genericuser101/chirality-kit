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

from ._runtime import my_log_file


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
