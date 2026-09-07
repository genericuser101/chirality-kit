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
