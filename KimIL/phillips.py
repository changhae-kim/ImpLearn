import numpy
import os

from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list

from .phillips_tools import rotate_vector, step_attractive, step_repulsive


class Phillips():

    def __init__(self, file_path, file_type,
            peripheral_oxygens=[2, 3],
            bond_cutoffs={
                ('Si', 'Si'): 2.0, ('O', 'O'): 2.0, ('Si', 'O'): 2.3, ('O', 'H'): 1.2,
                ('Cr', 'O'): 2.3, ('Cr', 'C'): 2.3, ('C', 'C'): 2.0, ('C', 'H'): 1.2,
                ('F', 'F'): 2.0, ('O', 'F'): 2.0, ('Si', 'F'): 2.3, ('F', 'H'): 1.2,
                },
            bond_lengths={('Cr', 'O'): 1.82, ('Cr', 'C'): 2.02, ('C', 'C'): 1.53, ('C', 'H'): 1.09},
            ethylene_bond_lengths={('Cr', 'C'): 2.5, ('C', 'C'): 1.34, ('C', 'H'): 1.09},
            transition_state_lengths={('Cr', 'C1'): 2.1, ('C2', 'C3'): 2.2, ('C3', 'Cr'): 2.1},
            OO_radius=3.0,
            alkyl_radius=2.0
            ):

        self.bond_cutoffs = bond_cutoffs
        self.bond_lengths = bond_lengths
        self.ethylene_bond_lengths = ethylene_bond_lengths
        self.transition_state_lengths = transition_state_lengths
        self.OO_radius = OO_radius
        self.alkyl_radius = alkyl_radius

        self.cluster = self.load_cluster(file_path, file_type)
        self.axes, self.chromium_cluster = self.attach_chromium(peripheral_oxygens)

        self.done_polymer = False

        return

    def do_polymer(self, alkyl_lengths=[4, 6]):

        self.L_butyl_cluster = self.attach_alkyl(alkyl_lengths[0], point_y=True, pucker=+1)
        self.L_butyl_R_ethylene_cluster = self.attach_ethylene(
                cluster=self.attach_alkyl(alkyl_lengths[0], point_y=True, relax=False),
                point_y=False, pucker=+1, relax=True)
        self.LR_transition_state_cluster = self.attach_transition_state(alkyl_lengths[1], point_y=False, pucker=+1)

        self.R_butyl_cluster = self.attach_alkyl(alkyl_lengths[0], point_y=False, pucker=-1)
        self.R_butyl_L_ethylene_cluster = self.attach_ethylene(
                cluster=self.attach_alkyl(alkyl_lengths[0], point_y=False, relax=False),
                point_y=True, pucker=-1, relax=True)
        self.RL_transition_state_cluster = self.attach_transition_state(alkyl_lengths[1], point_y=True, pucker=-1)

        self.done_polymer = True

        return

    def load_cluster(self, file_path, file_type):
        cluster = read(file_path, 0, file_type)
        return cluster

    def get_axes(self, cluster, peripheral_oxygens, bond_cutoffs=None):

        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        chasis_silicons = []
        for i in peripheral_oxygens:
            i_neighbors = bonds[1][bonds[0] == i]
            for j in i_neighbors:
                if atoms[j] == 'Si' and j not in chasis_silicons:
                    chasis_silicons.append(j)

        n, m = peripheral_oxygens
        p, q = chasis_silicons
        axes = numpy.empty((3, 3))
        axes[0] = coords[m] - coords[n]
        axes[0] = axes[0] / numpy.linalg.norm(axes[0])
        axes[2] = coords[n] + coords[m] - coords[p] - coords[q]
        axes[2] = axes[2] - numpy.dot(axes[2], axes[0]) * axes[0]
        axes[2] = axes[2] / numpy.linalg.norm(axes[2])
        axes[1] = numpy.cross(axes[2], axes[0])

        return axes

    def attach_chromium(self, peripheral_oxygens, cluster=None, bond_cutoffs=None, bond_lengths=None,
            relax=True, OO_radius=None, max_iter=50):

        if cluster is None:
            cluster = self.cluster
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if bond_lengths is None:
            bond_lengths = self.bond_lengths
        if OO_radius is None:
            OO_radius = self.OO_radius

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        peripheral_hydrogens = []
        chasis_silicons = []
        for i in peripheral_oxygens:
            i_neighbors = bonds[1][bonds[0] == i]
            for j in i_neighbors:
                if atoms[j] == 'H' and j not in peripheral_hydrogens:
                    peripheral_hydrogens.append(j)
                elif atoms[j] == 'Si' and j not in chasis_silicons:
                    chasis_silicons.append(j)

        n, m = peripheral_oxygens
        p, q = chasis_silicons
        r, s = peripheral_hydrogens

        if relax:

            O1_nonneighbors = [j for j, X in enumerate(atoms) if j not in [n, p, r, s]]
            OX1_radii = []
            for j in O1_nonneighbors:
                if atoms[j] == 'Si':
                    OX_bond_cutoff = bond_cutoffs[('Si', 'O')]
                elif atoms[j] == 'O':
                    OX_bond_cutoff = bond_cutoffs[('O', 'O')]
                elif atoms[j] == 'H':
                    OX_bond_cutoff = bond_cutoffs[('O', 'H')]
                OX1_radii.append([OX_bond_cutoff])
            OX1_radii = numpy.array(OX1_radii)

            O2_nonneighbors = [j for j, X in enumerate(atoms) if j not in [m, q, r, s]]
            OX2_radii = []
            for j in O2_nonneighbors:
                if atoms[j] == 'Si':
                    OX_bond_cutoff = bond_cutoffs[('Si', 'O')]
                elif atoms[j] == 'O':
                    OX_bond_cutoff = bond_cutoffs[('O', 'O')]
                elif atoms[j] == 'H':
                    OX_bond_cutoff = bond_cutoffs[('O', 'H')]
                OX2_radii.append([OX_bond_cutoff])
            OX2_radii = numpy.array(OX2_radii)

            status = -1
            for i in range(max_iter):
                if status == 0:
                    break
                else:
                    status = 0
                    new_coords = step_attractive(coords[n], coords[m], coords[p], coords[q], OO_radius)
                    if numpy.any(new_coords[0] != coords[n]) or numpy.any(new_coords[1] != coords[m]):
                        coords[n], coords[m] = new_coords[0], new_coords[1]
                        status = -1
                    new_coords = step_repulsive(coords[n][numpy.newaxis, :], coords[p], coords[O1_nonneighbors], OX1_radii)
                    if numpy.any(new_coords[0] != coords[n]):
                        coords[n] = new_coords[0]
                        status = -1
                    new_coords = step_repulsive(coords[m][numpy.newaxis, :], coords[q], coords[O2_nonneighbors], OX2_radii)
                    if numpy.any(new_coords[0] != coords[m]):
                        coords[m] = new_coords[0]
                        status = -1

        axes = self.get_axes(Atoms(atoms, coords), peripheral_oxygens)

        OO_dist = numpy.linalg.norm(coords[m] - coords[n])
        if OO_dist < 2.0 * bond_lengths[('Cr', 'O')]:
            Cr_coord = 0.5 * (coords[n] + coords[m]) + axes[2] * ((bond_lengths[('Cr', 'O')])**2.0 - (0.5 * OO_dist)**2.0)**0.5
        else:
            Cr_coord = 0.5 * (coords[n] + coords[m])

        chromium_atoms = ['Cr', 'O', 'O']
        chromium_coords = [Cr_coord, coords[n], coords[m]]
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if i not in peripheral_hydrogens + peripheral_oxygens:
                chromium_atoms.append(X)
                chromium_coords.append(coord)

        chromium_cluster = Atoms(chromium_atoms, chromium_coords)

        return axes, chromium_cluster

    def attach_alkyl(self, alkyl_length, cluster=None, bond_cutoffs=None, bond_lengths=None, axes=None,
            point_y=True, rotate_2=False, pucker=0, relax=True, alkyl_radius=None, max_iter=50):

        if cluster is None:
            cluster = self.chromium_cluster
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if bond_lengths is None:
            bond_lengths = self.bond_lengths
        if axes is None:
            axes = self.axes
        if alkyl_radius is None:
            alkyl_radius = self.alkyl_radius

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        o = len(atoms)
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'Cr':
                o = i
                break

        if point_y:
            tilts = [
                    axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) + axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) + axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0),
                    ]
        else:
            tilts = [
                    axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) - axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) - axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0),
                    ]

        C_coords = [coords[o] + tilts[0] * bond_lengths[('Cr', 'C')]]
        for i in range(1, alkyl_length):
            C_coords.append(C_coords[-1] + tilts[i%len(tilts)] * bond_lengths[('C', 'C')])
        H_coords = []
        for i in range(0, alkyl_length):
            if i == alkyl_length-1:
                H_coords.append(C_coords[-1] + tilts[(i+1)%len(tilts)] * bond_lengths[('C', 'H')])
            H_coords.append(C_coords[i] + rotate_vector(tilts[(i+1)%len(tilts)], -tilts[(i+0)%len(tilts)], +120.0) * bond_lengths[('C', 'H')])
            H_coords.append(C_coords[i] + rotate_vector(tilts[(i+1)%len(tilts)], -tilts[(i+0)%len(tilts)], -120.0) * bond_lengths[('C', 'H')])

        if rotate_2:
            for i in range(2, alkyl_length):
                C_coords[i] = C_coords[1] + rotate_vector(C_coords[i] - C_coords[1], -tilts[1], 180.0)
            for i in range(2, 2*alkyl_length+1):
                H_coords[i] = C_coords[1] + rotate_vector(H_coords[i] - C_coords[1], -tilts[1], 180.0)

        if pucker > 0:
            Cr_neighbors = bonds[1][bonds[0] == o]
            n = Cr_neighbors[0]
            oxygens = [i for i, X in enumerate(atoms) if X == 'O' and i not in Cr_neighbors]
            angle = 15.0
            for i in range(max_iter):
                Cr_coord = coords[n] + rotate_vector(coords[o] - coords[n], -axes[0], angle)
                CrO_dists = numpy.linalg.norm(coords[oxygens] - Cr_coord, axis=-1)
                if numpy.amin(CrO_dists) > bond_cutoffs[('Cr', 'O')]:
                    angle += 5.0
                else:
                    break
                if angle > 45.0:
                    angle = 45.0
                    break
            coords[o] = coords[n] + rotate_vector(coords[o] - coords[n], -axes[0], angle)
            for i in range(alkyl_length):
                C_coords[i] = coords[n] + rotate_vector(C_coords[i] - coords[n], -axes[0], angle)
            for i in range(2*alkyl_length+1):
                H_coords[i] = coords[n] + rotate_vector(H_coords[i] - coords[n], -axes[0], angle)
            if 'C' in atoms:
                c = len(atoms)
                for i, X in enumerate(atoms):
                    if X == 'C':
                        c = i
                        break
                ethylene = [i for i in range(c, len(atoms))]
                for i in ethylene:
                    coords[i] = coords[n] + rotate_vector(coords[i] - coords[n], -axes[0], angle)

        elif pucker < 0:
            Cr_neighbors = bonds[1][bonds[0] == o]
            n = Cr_neighbors[0]
            oxygens = [i for i, X in enumerate(atoms) if X == 'O' and i not in Cr_neighbors]
            angle = 15.0
            for i in range(max_iter):
                Cr_coord = coords[n] + rotate_vector(coords[o] - coords[n], +axes[0], angle)
                CrO_dists = numpy.linalg.norm(coords[oxygens] - Cr_coord, axis=-1)
                if numpy.amin(CrO_dists) > bond_cutoffs[('Cr', 'O')]:
                    angle += 5.0
                else:
                    break
                if angle > 45.0:
                    angle = 45.0
                    break
            coords[o] = coords[n] + rotate_vector(coords[o] - coords[n], +axes[0], angle)
            for i in range(alkyl_length):
                C_coords[i] = coords[n] + rotate_vector(C_coords[i] - coords[n], +axes[0], angle)
            for i in range(2*alkyl_length+1):
                H_coords[i] = coords[n] + rotate_vector(H_coords[i] - coords[n], +axes[0], angle)
            if 'C' in atoms:
                c = len(atoms)
                for i, X in enumerate(atoms):
                    if X == 'C':
                        c = i
                        break
                ethylene = [i for i in range(c, len(atoms))]
                for i in ethylene:
                    coords[i] = coords[n] + rotate_vector(coords[i] - coords[n], +axes[0], angle)

        if relax:
            if 'C' in atoms:
                c = len(atoms)
                for i, X in enumerate(atoms):
                    if X == 'C':
                        c = i
                        break
                support = [i for i in range(0, c) if i != o]
                ethylene = [i for i in range(c, len(atoms))]
                alkyl_radii = numpy.full((len(support)+len(ethylene), 3*alkyl_length+1), alkyl_radius)
                ethylene_radii = numpy.full((len(support)+3*alkyl_length+1, len(ethylene)), alkyl_radius)
                status = -1
                for i in range(max_iter):
                    if status == 0:
                        break
                    else:
                        status = 0
                        new_coords = step_repulsive(numpy.concatenate([C_coords, H_coords]), coords[o], coords[support+ethylene], alkyl_radii)
                        if numpy.any(new_coords[0] != C_coords[0]):
                            C_coords = new_coords[:alkyl_length]
                            H_coords = new_coords[alkyl_length:]
                            status = -1
                        new_coords = step_repulsive(coords[ethylene], coords[o], numpy.concatenate([coords[support], C_coords, H_coords]), ethylene_radii)
                        if numpy.any(new_coords[0] != coords[c]):
                            coords[ethylene] = new_coords
                            status = -1

            else:
                nonneighbors = [j for j, X in enumerate(atoms) if j != o]
                alkyl_radii = numpy.full((len(nonneighbors), 3*alkyl_length+1), alkyl_radius)
                status = -1
                for i in range(max_iter):
                    if status == 0:
                        break
                    else:
                        status = 0
                        new_coords = step_repulsive(numpy.concatenate([C_coords, H_coords]), coords[o], coords[nonneighbors], alkyl_radii)
                        if numpy.any(new_coords[0] != C_coords[0]):
                            C_coords = new_coords[:alkyl_length]
                            H_coords = new_coords[alkyl_length:]
                            status = -1

        c = len(atoms)
        alkyl_atoms = []
        alkyl_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'C':
                c = i
                break
            else:
                alkyl_atoms.append(X)
                alkyl_coords.append(coord)
        for X, coord in zip(atoms[c:], coords[c:]):
            if X == 'C':
                alkyl_atoms.append('C')
                alkyl_coords.append(coord)
        for coord in C_coords:
            alkyl_atoms.append('C')
            alkyl_coords.append(coord)
        for X, coord in zip(atoms[c:], coords[c:]):
            if X == 'H':
                alkyl_atoms.append('H')
                alkyl_coords.append(coord)
        for coord in H_coords:
            alkyl_atoms.append('H')
            alkyl_coords.append(coord)

        alkyl_cluster = Atoms(alkyl_atoms, alkyl_coords)

        return alkyl_cluster

    def attach_ethylene(self, cluster=None, bond_cutoffs=None, bond_lengths=None, ethylene_bond_lengths=None, axes=None,
            point_y=False, pucker=0, relax=True, alkyl_radius=None, max_iter=50):

        if cluster is None:
            cluster = self.chromium_cluster
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if bond_lengths is None:
            bond_lengths = self.bond_lengths
        if ethylene_bond_lengths is None:
            ethylene_bond_lengths = self.ethylene_bond_lengths
        if axes is None:
            axes = self.axes
        if alkyl_radius is None:
            alkyl_radius = self.alkyl_radius

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        o = len(atoms)
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'Cr':
                o = i

        if point_y:
            tilt0 = axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) + axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0)
            tilt1 = - axes[0]
        else:
            tilt0 = axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) - axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0)
            tilt1 = + axes[0]
        C1_coord = coords[o] + tilt0 * ethylene_bond_lengths[('Cr', 'C')]
        C2_coord = C1_coord + tilt1 * ethylene_bond_lengths[('C', 'C')]
        C_coords = [C1_coord, C2_coord]
        H1_coord = C1_coord + rotate_vector(tilt1, tilt0, -120.0) * bond_lengths[('C', 'H')]
        H2_coord = C1_coord + rotate_vector(tilt1, tilt0, +120.0) * bond_lengths[('C', 'H')]
        H3_coord = C2_coord + rotate_vector(tilt1, tilt0, -60.0) * bond_lengths[('C', 'H')]
        H4_coord = C2_coord + rotate_vector(tilt1, tilt0, +60.0) * bond_lengths[('C', 'H')]
        H_coords = [H1_coord, H2_coord, H3_coord, H4_coord]

        if pucker > 0:
            Cr_neighbors = bonds[1][bonds[0] == o]
            n = Cr_neighbors[0]
            oxygens = [i for i, X in enumerate(atoms) if X == 'O' and i not in Cr_neighbors]
            angle = 15.0
            for i in range(max_iter):
                Cr_coord = coords[n] + rotate_vector(coords[o] - coords[n], -axes[0], angle)
                CrO_dists = numpy.linalg.norm(coords[oxygens] - Cr_coord, axis=-1)
                if numpy.amin(CrO_dists) > bond_cutoffs[('Cr', 'O')]:
                    angle += 5.0
                else:
                    break
                if angle > 45.0:
                    angle = 45.0
                    break
            coords[o] = coords[n] + rotate_vector(coords[o] - coords[n], -axes[0], angle)
            for i in range(2):
                C_coords[i] = coords[n] + rotate_vector(C_coords[i] - coords[n], -axes[0], angle)
            for i in range(4):
                H_coords[i] = coords[n] + rotate_vector(H_coords[i] - coords[n], -axes[0], angle)
            if 'C' in atoms:
                c = len(atoms)
                for i, X in enumerate(atoms):
                    if X == 'C':
                        c = i
                        break
                ethylene = [i for i in range(c, len(atoms))]
                for i in ethylene:
                    coords[i] = coords[n] + rotate_vector(coords[i] - coords[n], -axes[0], angle)

        elif pucker < 0:
            Cr_neighbors = bonds[1][bonds[0] == o]
            n = Cr_neighbors[0]
            oxygens = [i for i, X in enumerate(atoms) if X == 'O' and i not in Cr_neighbors]
            angle = 15.0
            for i in range(max_iter):
                Cr_coord = coords[n] + rotate_vector(coords[o] - coords[n], +axes[0], angle)
                CrO_dists = numpy.linalg.norm(coords[oxygens] - Cr_coord, axis=-1)
                if numpy.amin(CrO_dists) > bond_cutoffs[('Cr', 'O')]:
                    angle += 5.0
                else:
                    break
                if angle > 45.0:
                    angle = 45.0
                    break
            coords[o] = coords[n] + rotate_vector(coords[o] - coords[n], +axes[0], angle)
            for i in range(2):
                C_coords[i] = coords[n] + rotate_vector(C_coords[i] - coords[n], +axes[0], angle)
            for i in range(4):
                H_coords[i] = coords[n] + rotate_vector(H_coords[i] - coords[n], +axes[0], angle)
            if 'C' in atoms:
                c = len(atoms)
                for i, X in enumerate(atoms):
                    if X == 'C':
                        c = i
                        break
                alkyl = [i for i in range(c, len(atoms))]
                for i in alkyl:
                    coords[i] = coords[n] + rotate_vector(coords[i] - coords[n], +axes[0], angle)

        if relax:
            if 'C' in atoms:
                c = len(atoms)
                for i, X in enumerate(atoms):
                    if X == 'C':
                        c = i
                        break
                support = [j for j in range(0, c) if j != o]
                alkyl = [j for j in range(c, len(atoms))]
                alkyl_radii = numpy.full((len(support)+6, len(alkyl)), alkyl_radius)
                ethylene_radii = numpy.full((len(support)+len(alkyl), 6), alkyl_radius)
                status = -1
                for i in range(max_iter):
                    if status == 0:
                        break
                    else:
                        status = 0
                        new_coords = step_repulsive(coords[alkyl], coords[o], numpy.concatenate([coords[support], C_coords, H_coords]), alkyl_radii)
                        if numpy.any(new_coords[0] != coords[c]):
                            coords[alkyl] = new_coords
                            status = -1
                        new_coords = step_repulsive(numpy.concatenate([C_coords, H_coords]), coords[o], coords[support+alkyl], ethylene_radii)
                        if numpy.any(new_coords[0] != C_coords[0]):
                            C_coords = new_coords[:2]
                            H_coords = new_coords[2:]
                            status = -1

            else:
                nonneighbors = [j for j, X in enumerate(atoms) if j != o]
                alkyl_radii = numpy.full((len(nonneighbors), 6), alkyl_radius)
                status = -1
                for i in range(max_iter):
                    if status == 0:
                        break
                    else:
                        status = 0
                        new_coords = step_repulsive(numpy.concatenate([C_coords, H_coords]), coords[o], coords[nonneighbors], alkyl_radii)
                        if numpy.any(new_coords[0] != C_coords[0]):
                            C_coords = new_coords[:2]
                            H_coords = new_coords[2:]
                            status = -1

        c = len(atoms)
        ethylene_atoms = []
        ethylene_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'C':
                c = i
                break
            else:
                ethylene_atoms.append(X)
                ethylene_coords.append(coord)
        for coord in C_coords:
            ethylene_atoms.append('C')
            ethylene_coords.append(coord)
        for X, coord in zip(atoms[c:], coords[c:]):
            if X == 'C':
                ethylene_atoms.append('C')
                ethylene_coords.append(coord)
        for coord in H_coords:
            ethylene_atoms.append('H')
            ethylene_coords.append(coord)
        for X, coord in zip(atoms[c:], coords[c:]):
            if X == 'H':
                ethylene_atoms.append('H')
                ethylene_coords.append(coord)

        ethylene_cluster = Atoms(ethylene_atoms, ethylene_coords)

        return ethylene_cluster

    def attach_transition_state(self, alkyl_length,
            cluster=None, bond_cutoffs=None, bond_lengths=None, ethylene_bond_lengths=None, transition_state_lengths=None, axes=None,
            point_y=False, pucker=0, relax=True, alkyl_radius=None, max_iter=50):

        if cluster is None:
            cluster = self.chromium_cluster
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if bond_lengths is None:
            bond_lengths = self.bond_lengths
        if ethylene_bond_lengths is None:
            ethylene_bond_lengths = self.ethylene_bond_lengths
        if transition_state_lengths is None:
            transition_state_lengths = self.transition_state_lengths
        if axes is None:
            axes = self.axes
        if alkyl_radius is None:
            alkyl_radius = self.alkyl_radius

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        o = len(atoms)
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'Cr':
                o = i

        if point_y:
            tilts = [
                    axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) + axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) + axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0),
                    ]
        else:
            tilts = [
                    axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) - axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) - axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0),
                    ]

        C1_coord = coords[o] + tilts[0] * transition_state_lengths[('Cr', 'C1')]
        C2_coord = C1_coord + tilts[1] * ethylene_bond_lengths[('C', 'C')]
        C3_coord = C2_coord + tilts[0] * transition_state_lengths[('C2', 'C3')]
        C_coords = [C1_coord, C2_coord, C3_coord]
        for i in range(3, alkyl_length):
            C_coords.append(C_coords[-1] + tilts[i%len(tilts)] * bond_lengths[('C', 'C')])
        H_coords = []
        for i in range(0, alkyl_length):
            if i == alkyl_length-1:
                H_coords.append(C_coords[-1] + tilts[(i+1)%len(tilts)] * bond_lengths[('C', 'H')])
            H_coords.append(C_coords[i] + rotate_vector(tilts[(i+1)%len(tilts)], -tilts[(i+0)%len(tilts)], +120.0) * bond_lengths[('C', 'H')])
            H_coords.append(C_coords[i] + rotate_vector(tilts[(i+1)%len(tilts)], -tilts[(i+0)%len(tilts)], -120.0) * bond_lengths[('C', 'H')])

        for i in range(2, alkyl_length):
            C_coords[i] = C_coords[1] + rotate_vector(C_coords[i] - C_coords[1], -tilts[1], 180.0)
        for i in range(2, 2*alkyl_length+1):
            H_coords[i] = C_coords[1] + rotate_vector(H_coords[i] - C_coords[1], -tilts[1], 180.0)

        if point_y:
            axis = +axes[0]
        else:
            axis = -axes[0]

        x = numpy.array([1.0, 0.0])
        y = numpy.array([0.0, 1.0])
        ab = ethylene_bond_lengths[('C', 'C')] * x
        bc = transition_state_lengths[('C2', 'C3')] * (x * numpy.cos((180.0-109.5)/180.0*numpy.pi) + y * numpy.sin((180.0-109.5)/180.0*numpy.pi))
        ac = ab + bc
        ph3 = numpy.arccos(numpy.dot(ab, ac)/(numpy.linalg.norm(ab)*numpy.linalg.norm(ac)))
        l1 = transition_state_lengths[('C3', 'Cr')]
        l2 = numpy.linalg.norm(ac)
        th1 = numpy.arccos(0.5*l2/l1)
        th2 = 2.0 * numpy.arcsin(0.5*l2/l1)

        angle = 0.5 * (109.5 - th2 * 180.0 / numpy.pi)
        for i in range(0, alkyl_length):
            C_coords[i] = coords[o] + rotate_vector(C_coords[i] - coords[o], axis, angle)
        for i in range(0, 2*alkyl_length+1):
            H_coords[i] = coords[o] + rotate_vector(H_coords[i] - coords[o], axis, angle)

        angle = 109.5 - (th1 + ph3) * 180.0 / numpy.pi
        for i in range(1, alkyl_length):
            C_coords[i] = C_coords[0] + rotate_vector(C_coords[i] - C_coords[0], axis, angle)
        for i in range(1, 2*alkyl_length+1):
            H_coords[i] = C_coords[0] + rotate_vector(H_coords[i] - C_coords[0], axis, angle)

        if pucker > 0:
            Cr_neighbors = bonds[1][bonds[0] == o]
            n = Cr_neighbors[0]
            oxygens = [i for i, X in enumerate(atoms) if X == 'O' and i not in Cr_neighbors]
            angle = 15.0
            for i in range(max_iter):
                Cr_coord = coords[n] + rotate_vector(coords[o] - coords[n], -axes[0], angle)
                CrO_dists = numpy.linalg.norm(coords[oxygens] - Cr_coord, axis=-1)
                if numpy.amin(CrO_dists) > bond_cutoffs[('Cr', 'O')]:
                    angle += 5.0
                else:
                    break
                if angle > 45.0:
                    angle = 45.0
                    break
            coords[o] = coords[n] + rotate_vector(coords[o] - coords[n], -axes[0], angle)
            for i in range(alkyl_length):
                C_coords[i] = coords[n] + rotate_vector(C_coords[i] - coords[n], -axes[0], angle)
            for i in range(2*alkyl_length+1):
                H_coords[i] = coords[n] + rotate_vector(H_coords[i] - coords[n], -axes[0], angle)

        elif pucker < 0:
            Cr_neighbors = bonds[1][bonds[0] == o]
            n = Cr_neighbors[0]
            oxygens = [i for i, X in enumerate(atoms) if X == 'O' and i not in Cr_neighbors]
            angle = 15.0
            for i in range(max_iter):
                Cr_coord = coords[n] + rotate_vector(coords[o] - coords[n], +axes[0], angle)
                CrO_dists = numpy.linalg.norm(coords[oxygens] - Cr_coord, axis=-1)
                if numpy.amin(CrO_dists) > bond_cutoffs[('Cr', 'O')]:
                    angle += 5.0
                else:
                    break
                if angle > 45.0:
                    angle = 45.0
                    break
            coords[o] = coords[n] + rotate_vector(coords[o] - coords[n], +axes[0], angle)
            for i in range(alkyl_length):
                C_coords[i] = coords[n] + rotate_vector(C_coords[i] - coords[n], +axes[0], angle)
            for i in range(2*alkyl_length+1):
                H_coords[i] = coords[n] + rotate_vector(H_coords[i] - coords[n], +axes[0], angle)

        if relax:
            status = -1
            for i in range(max_iter):
                if status == 0:
                    break
                else:
                    status = 0
                    nonneighbors = [j for j, X in enumerate(atoms) if j != o]
                    alkyl_radii = numpy.full((len(atoms)-1, 3*alkyl_length+1), alkyl_radius)
                    new_coords = step_repulsive(numpy.concatenate([C_coords, H_coords]), coords[o], coords[nonneighbors], alkyl_radii)
                    if numpy.any(new_coords[0] != C_coords[0]):
                        C_coords = new_coords[:alkyl_length]
                        H_coords = new_coords[alkyl_length:]
                        status = -1

        c = len(atoms)
        transition_state_atoms = []
        transition_state_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'C':
                c = i
                break
            else:
                transition_state_atoms.append(X)
                transition_state_coords.append(coord)
        for X, coord in zip(atoms[c:], coords[c:]):
            if X == 'C':
                transition_state_atoms.append('C')
                transition_state_coords.append(coord)
        for coord in C_coords:
            transition_state_atoms.append('C')
            transition_state_coords.append(coord)
        for X, coord in zip(atoms[c:], coords[c:]):
            if X == 'H':
                transition_state_atoms.append('H')
                transition_state_coords.append(coord)
        for coord in H_coords:
            transition_state_atoms.append('H')
            transition_state_coords.append(coord)

        transition_state_cluster = Atoms(transition_state_atoms, transition_state_coords)

        return transition_state_cluster

    def save_clusters(self, file_path, file_type, labels=None, clusters=None):
        if labels is None:
            labels = []
            clusters = []
            if self.done_polymer:
                labels += [
                        'L-butyl', 'L-butyl-R-ethylene', 'LR-transition-state',
                        'R-butyl', 'R-butyl-L-ethylene', 'RL-transition-state',
                        ]
                clusters += [
                        self.L_butyl_cluster, self.L_butyl_R_ethylene_cluster, self.LR_transition_state_cluster,
                        self.R_butyl_cluster, self.R_butyl_L_ethylene_cluster, self.RL_transition_state_cluster,
                        ]
        for label, cluster in zip(labels, clusters):
            if not os.path.exists(file_path.format(label)):
                write(file_path.format(label), cluster, file_type)
        return


if __name__ == '__main__':


    clusters = Phillips('tests/A_0001.xyz', 'xyz', [2, 3])
    clusters.do_polymer()
    clusters.save_clusters('A_0001_{:s}.xyz', 'xyz')


