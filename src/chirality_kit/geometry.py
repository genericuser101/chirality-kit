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
