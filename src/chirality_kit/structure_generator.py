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
