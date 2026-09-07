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
                my_log_file.info(f"Ionic placement: '{solvent_type}' -> {cathode_type} x{cathode_count}, {anode_type} x{anode_count}.")

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
