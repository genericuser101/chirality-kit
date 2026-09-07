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


DUMMY_FG_TYPE = "DUMMY"
TERM_SIDES = {
    "both": (True, True),
    "-": (True, False),
    "+": (False, True),
}
TERM_GROUP_FIELDS = {"type", "count", "side"}


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
