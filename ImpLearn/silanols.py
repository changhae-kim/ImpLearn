import numpy

from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list
from matplotlib import pyplot
from scipy.spatial.transform import Rotation

from .silanols_tools import reorder_podal_oxygens, reorder_podal_oxygens_v2


class Silanols():

    def __init__(self, file_path,
            file_type='xyz',
            pbc=[21.01554, 21.01554, 90.23032],
            bond_cutoffs={('Si', 'O'): 2.3, ('O', 'H'): 1.2},
            viable_cutoff=5.5,
            OH_bond_length=0.956,
            exclude_waters=True,
            exclude_geminals=True,
            reorder_podals=True,
            F_capping=True,
            reorient_clusters=True,
            reorder_atoms=True
            ):

        self.slab = self.load_slab(file_path, file_type, pbc)

        self.bond_cutoffs = bond_cutoffs
        self.viable_cutoff = viable_cutoff
        self.OH_bond_length = OH_bond_length

        self.exclude_waters = exclude_waters
        self.exclude_geminals = exclude_geminals
        self.reorder_podals = reorder_podals
        self.F_capping = F_capping
        self.reorient_clusters = reorient_clusters
        self.reorder_atoms = reorder_atoms

        self.OH_groups = self.get_OH_groups()
        self.geminal_OH_pairs = self.get_geminal_OH_pairs()
        self.vicinal_OH_pairs = self.get_vicinal_OH_pairs()
        self.viable_OH_pairs = self.get_viable_OH_pairs()
        self.minimal_clusters = self.carve_minimal_clusters()

        return

    def load_slab(self, file_path, file_type, pbc=None):
        slab = read(file_path, 0, file_type)
        if pbc is not None:
            slab.set_pbc(True)
            slab.set_cell(pbc)
        return slab

    def get_OH_groups(self, slab=None, bond_cutoffs=None, exclude_waters=None):

        if slab is None:
            slab = self.slab
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if exclude_waters is None:
            exclude_waters = self.exclude_waters

        atoms = slab.get_chemical_symbols()
        bonds = neighbor_list('ij', slab, bond_cutoffs)

        OH_groups = []
        for i, X in enumerate(atoms):
            if X == 'H':
                H_neighbors = bonds[1][bonds[0] == i]
                for j in H_neighbors:
                    if atoms[j] == 'O':
                        O_neighbors = bonds[1][bonds[0] == j]
                        if exclude_waters:
                            if 'Si' not in [atoms[i] for i in O_neighbors]:
                                continue
                        OH_groups.append([i, j])

        return OH_groups

    def get_geminal_OH_pairs(self, slab=None, bond_cutoffs=None, OH_groups=None):

        if slab is None:
            slab = self.slab
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if OH_groups is None:
            OH_groups = self.OH_groups

        atoms = slab.get_chemical_symbols()
        bonds = neighbor_list('ij', slab, bond_cutoffs)

        geminal_OH_pairs = []
        for n, OH1_group in enumerate(OH_groups):
            for m, OH2_group in enumerate(OH_groups):
                if n < m:
                    O1_neighbors = bonds[1][bonds[0] == OH1_group[1]]
                    O2_neighbors = bonds[1][bonds[0] == OH2_group[1]]
                    for i in numpy.intersect1d(O1_neighbors, O2_neighbors):
                        if atoms[i] == 'Si':
                            geminal_OH_pairs.append([n, m])

        return geminal_OH_pairs

    def get_vicinal_OH_pairs(self, slab=None, bond_cutoffs=None, OH_groups=None):

        if slab is None:
            slab = self.slab
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if OH_groups is None:
            OH_groups = self.OH_groups

        atoms = slab.get_chemical_symbols()
        bonds = neighbor_list('ij', slab, bond_cutoffs)

        vicinal_OH_pairs = []
        for n, OH1_group in enumerate(OH_groups):
            for m, OH2_group in enumerate(OH_groups):
                if n < m:
                    O1_neighbors = bonds[1][bonds[0] == OH1_group[1]]
                    O2_neighbors = bonds[1][bonds[0] == OH2_group[1]]
                    geminal = False
                    for i in numpy.intersect1d(O1_neighbors, O2_neighbors):
                        if atoms[i] == 'Si':
                            geminal = True
                    if geminal:
                        continue
                    Si1_candidates = [i for i in O1_neighbors if atoms[i] == 'Si']
                    Si2_candidates = [i for i in O2_neighbors if atoms[i] == 'Si']
                    Si1_candidate_neighbors = [bonds[1][bonds[0] == i] for i in Si1_candidates]
                    Si2_candidate_neighbors = [bonds[1][bonds[0] == i] for i in Si2_candidates]
                    for i in numpy.intersect1d(numpy.concatenate(Si1_candidate_neighbors), numpy.concatenate(Si2_candidate_neighbors)):
                        if atoms[i] == 'O':
                            vicinal_OH_pairs.append([n, m])

        return vicinal_OH_pairs

    def get_viable_OH_pairs(self, slab=None, OH_groups=None, viable_cutoff=None, exclude_geminals=None, geminal_OH_pairs=None):

        if slab is None:
            slab = self.slab
        if OH_groups is None:
            OH_groups = self.OH_groups
        if viable_cutoff is None:
            viable_cutoff = self.viable_cutoff
        if exclude_geminals is None:
            exclude_geminals = self.exclude_geminals
        if geminal_OH_pairs is None:
            geminal_OH_pairs = self.geminal_OH_pairs

        viable_OH_pairs = []
        for n, OH1_group in enumerate(OH_groups):
            for m, OH2_group in enumerate(OH_groups):
                if n < m:
                    if slab.get_distance(OH1_group[1], OH2_group[1], mic=True) < viable_cutoff:
                        if exclude_geminals:
                            geminal = False
                            for pair in geminal_OH_pairs:
                                if n in pair or m in pair:
                                    geminal = True
                                    break
                            if geminal:
                                continue
                        viable_OH_pairs.append([n, m])

        return viable_OH_pairs

    def carve_minimal_clusters(self, slab=None, bond_cutoffs=None, OH_groups=None, viable_OH_pairs=None, OH_bond_length=None, reorder_podals=None, F_capping=None, reorient_clusters=None, reorder_atoms=None):

        if slab is None:
            slab = self.slab
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if OH_groups is None:
            OH_groups = self.OH_groups
        if viable_OH_pairs is None:
            viable_OH_pairs = self.viable_OH_pairs
        if OH_bond_length is None:
            OH_bond_length = self.OH_bond_length
        if reorder_podals is None:
            reorder_podals = self.reorder_podals
        if F_capping is None:
            F_capping = self.F_capping
        if reorient_clusters is None:
            reorient_clusters = self.reorient_clusters
        if reorder_atoms is None:
            reorder_atoms = self.reorder_atoms

        atoms = slab.get_chemical_symbols()
        bonds = neighbor_list('ij', slab, bond_cutoffs)

        minimal_clusters = []
        for a, b in viable_OH_pairs:

            OH1_group = OH_groups[a]
            OH2_group = OH_groups[b]

            cluster_atoms = []
            cluster_coords = []

            peripheral_hydrogens = [OH1_group[0], OH2_group[0]]
            cluster_atoms.append('H')
            cluster_coords.append(numpy.zeros(3))
            cluster_atoms.append('H')
            cluster_coords.append(cluster_coords[0] + slab.get_distance(OH1_group[0], OH2_group[0], mic=True, vector=True))

            peripheral_oxygens = [OH1_group[1], OH2_group[1]]
            cluster_atoms.append('O')
            cluster_coords.append(cluster_coords[0] + slab.get_distance(OH1_group[0], OH1_group[1], mic=True, vector=True))
            cluster_atoms.append('O')
            cluster_coords.append(cluster_coords[1] + slab.get_distance(OH2_group[0], OH2_group[1], mic=True, vector=True))

            chasis_silicons = []
            for n, i in enumerate(peripheral_oxygens):
                i_neighbors = bonds[1][bonds[0] == i]
                for j in i_neighbors:
                    if atoms[j] == 'Si' and j not in chasis_silicons:
                        chasis_silicons.append(j)
                        cluster_atoms.append('Si')
                        cluster_coords.append(cluster_coords[len(peripheral_hydrogens) + n] + slab.get_distance(i, j, mic=True, vector=True))

            bridging_oxygens = []
            podal_oxygens = []
            for n, i in enumerate(chasis_silicons):
                i_neighbors = bonds[1][bonds[0] == i]
                for j in i_neighbors:
                    if atoms[j] == 'O' and j not in peripheral_oxygens + bridging_oxygens + podal_oxygens:
                        j_neighbors = bonds[1][bonds[0] == j]
                        m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens)
                        if len(numpy.intersect1d(chasis_silicons, j_neighbors)) > 1:
                            bridging_oxygens.append(j)
                            cluster_atoms.insert(m, 'O')
                            cluster_coords.insert(m, cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))
                        else:
                            podal_oxygens.append(j)
                            cluster_atoms.append('O')
                            cluster_coords.append(cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))

            if reorder_podals:
                n = len(peripheral_hydrogens)
                m = len(peripheral_hydrogens) + len(peripheral_oxygens)
                l = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens)
                p = l + len(chasis_silicons)
                q = l + len(chasis_silicons) + len(podal_oxygens)
                if len(bridging_oxygens) > 0:
                    # reordered = reorder_podal_oxygens(cluster_coords[p:q], cluster_coords[n+0], cluster_coords[n+1], cluster_coords[l+0], cluster_coords[l+1], cluster_coords[l-1])
                    reordered = reorder_podal_oxygens_v2(cluster_coords[p:q], cluster_coords[n:m], cluster_coords[l:p], cluster_coords[m:l])
                else:
                    # reordered = reorder_podal_oxygens(cluster_coords[p:q], cluster_coords[n+0], cluster_coords[n+1], cluster_coords[l+0], cluster_coords[l+1])
                    reordered = reorder_podal_oxygens_v2(cluster_coords[p:q], cluster_coords[n:m], cluster_coords[l:p])
                old_coords = cluster_coords
                old_podals = podal_oxygens
                cluster_coords = []
                podal_oxygens = []
                for coord in old_coords[:p]:
                    cluster_coords.append(coord)
                for i in reordered:
                    podal_oxygens.append(old_podals[i])
                    cluster_coords.append(old_coords[p:q][i])
                for coord in old_coords[q:]:
                    cluster_coords.append(coord)

            m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens) + len(chasis_silicons)
            podal_hydrogens = []
            if F_capping:
                for n, i in enumerate(podal_oxygens):
                    cluster_atoms[m + n] = 'F'
            else:
                for n, i in enumerate(podal_oxygens):
                    i_neighbors = bonds[1][bonds[0] == i]
                    for j in i_neighbors:
                        if atoms[j] == 'Si' and j not in chasis_silicons:
                            podal_hydrogens.append(j)
                            cluster_atoms.append('H')
                            axis = slab.get_distance(i, j, mic=True, vector=True)
                            axis = axis / numpy.linalg.norm(axis)
                            cluster_coords.append(cluster_coords[m + n] + axis * OH_bond_length)

            if reorient_clusters:
                n = len(peripheral_hydrogens)
                m = len(peripheral_hydrogens) + 1
                p = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens)
                q = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens) + 1
                origin = 0.5 * (cluster_coords[p] + cluster_coords[q])
                axes = numpy.empty((3, 3))
                axes[0] = cluster_coords[q] - cluster_coords[p]
                axes[0] = axes[0] / numpy.linalg.norm(axes[0])
                axes[2] = cluster_coords[n] + cluster_coords[m] - cluster_coords[p] - cluster_coords[q]
                axes[2] = axes[2] - axes[0] * numpy.inner(axes[0], axes[2])
                axes[2] = axes[2] / numpy.linalg.norm(axes[2])
                axes[1] = numpy.cross(axes[2], axes[0])
                cluster_coords = [numpy.einsum('ij,j->i', axes, coords - origin) for coords in cluster_coords]

            if reorder_atoms:
                n = len(peripheral_hydrogens)
                m = len(peripheral_hydrogens) + len(peripheral_oxygens)
                l = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens)
                p = l + len(chasis_silicons)
                q = l + len(chasis_silicons) + len(podal_oxygens)
                r = l + len(chasis_silicons) + len(podal_oxygens) + len(podal_hydrogens)
                r = list(range(q, r))
                q = list(range(p, q))
                p = list(range(l, p))
                l = list(range(m, l))
                m = list(range(n, m))
                n = list(range(0, n))
                indices = r + q + p + l + m + n
                cluster_atoms = [cluster_atoms[i] for i in indices]
                cluster_coords = [cluster_coords[i] for i in indices]

            cluster = Atoms(cluster_atoms, cluster_coords)
            minimal_clusters.append(cluster)

        return minimal_clusters

    def carve_secondary_clusters(self, slab=None, bond_cutoffs=None, OH_groups=None, viable_OH_pairs=None, OH_bond_length=None, reorder_podals=None, F_capping=None, reorient_clusters=None, reorder_atoms=None):

        if slab is None:
            slab = self.slab
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if OH_groups is None:
            OH_groups = self.OH_groups
        if viable_OH_pairs is None:
            viable_OH_pairs = self.viable_OH_pairs
        if OH_bond_length is None:
            OH_bond_length = self.OH_bond_length
        if reorder_podals is None:
            reorder_podals = self.reorder_podals
        if F_capping is None:
            F_capping = self.F_capping
        if reorient_clusters is None:
            reorient_clusters = self.reorient_clusters
        if reorder_atoms is None:
            reorder_atoms = self.reorder_atoms

        atoms = slab.get_chemical_symbols()
        bonds = neighbor_list('ij', slab, bond_cutoffs)

        secondary_clusters = []
        for a, b in viable_OH_pairs:

            OH1_group = OH_groups[a]
            OH2_group = OH_groups[b]

            cluster_atoms = []
            cluster_coords = []

            peripheral_hydrogens = [OH1_group[0], OH2_group[0]]
            cluster_atoms.append('H')
            cluster_coords.append(numpy.zeros(3))
            cluster_atoms.append('H')
            cluster_coords.append(cluster_coords[0] + slab.get_distance(OH1_group[0], OH2_group[0], mic=True, vector=True))

            peripheral_oxygens = [OH1_group[1], OH2_group[1]]
            cluster_atoms.append('O')
            cluster_coords.append(cluster_coords[0] + slab.get_distance(OH1_group[0], OH1_group[1], mic=True, vector=True))
            cluster_atoms.append('O')
            cluster_coords.append(cluster_coords[1] + slab.get_distance(OH2_group[0], OH2_group[1], mic=True, vector=True))

            chasis_silicons = []
            for n, i in enumerate(peripheral_oxygens):
                i_neighbors = bonds[1][bonds[0] == i]
                for j in i_neighbors:
                    if atoms[j] == 'Si' and j not in chasis_silicons:
                        chasis_silicons.append(j)
                        cluster_atoms.append('Si')
                        cluster_coords.append(cluster_coords[len(peripheral_hydrogens) + n] + slab.get_distance(i, j, mic=True, vector=True))

            bridging_oxygens = []
            podal_oxygens = []
            for n, i in enumerate(chasis_silicons):
                i_neighbors = bonds[1][bonds[0] == i]
                for j in i_neighbors:
                    if atoms[j] == 'O' and j not in peripheral_oxygens + bridging_oxygens + podal_oxygens:
                        j_neighbors = bonds[1][bonds[0] == j]
                        m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens)
                        if len(numpy.intersect1d(chasis_silicons, j_neighbors)) > 1:
                            bridging_oxygens.append(j)
                            cluster_atoms.insert(m, 'O')
                            cluster_coords.insert(m, cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))
                        else:
                            podal_oxygens.append(j)
                            cluster_atoms.append('O')
                            cluster_coords.append(cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))

            if reorder_podals:
                n = len(peripheral_hydrogens)
                m = len(peripheral_hydrogens) + len(peripheral_oxygens)
                l = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens)
                p = l + len(chasis_silicons)
                q = l + len(chasis_silicons) + len(podal_oxygens)
                if len(bridging_oxygens) > 0:
                    # reordered = reorder_podal_oxygens(cluster_coords[p:q], cluster_coords[n+0], cluster_coords[n+1], cluster_coords[l+0], cluster_coords[l+1], cluster_coords[l-1])
                    reordered = reorder_podal_oxygens_v2(cluster_coords[p:q], cluster_coords[n:m], cluster_coords[l:p], cluster_coords[m:l])
                else:
                    # reordered = reorder_podal_oxygens(cluster_coords[p:q], cluster_coords[n+0], cluster_coords[n+1], cluster_coords[l+0], cluster_coords[l+1])
                    reordered = reorder_podal_oxygens_v2(cluster_coords[p:q], cluster_coords[n:m], cluster_coords[l:p])
                old_coords = cluster_coords
                old_podals = podal_oxygens
                cluster_coords = []
                podal_oxygens = []
                for coord in old_coords[:p]:
                    cluster_coords.append(coord)
                for i in reordered:
                    podal_oxygens.append(old_podals[i])
                    cluster_coords.append(old_coords[p:q][i])
                for coord in old_coords[q:]:
                    cluster_coords.append(coord)

            m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens) + len(chasis_silicons)
            secondary_chasis_silicons = []
            for n, i in enumerate(podal_oxygens):
                i_neighbors = bonds[1][bonds[0] == i]
                for j in i_neighbors:
                    if atoms[j] == 'Si' and j not in chasis_silicons + secondary_chasis_silicons:
                        secondary_chasis_silicons.append(j)
                        cluster_atoms.append('Si')
                        cluster_coords.append(cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))

            secondary_bridging_oxygens = []
            secondary_podal_oxygens = []
            for n, i in enumerate(secondary_chasis_silicons):
                i_neighbors = bonds[1][bonds[0] == i]
                for j in i_neighbors:
                    if atoms[j] == 'O' and j not in peripheral_oxygens + bridging_oxygens + podal_oxygens + secondary_bridging_oxygens + secondary_podal_oxygens:
                        j_neighbors = bonds[1][bonds[0] == j]
                        m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens) + len(chasis_silicons) + len(podal_oxygens) + len(secondary_bridging_oxygens)
                        if len(numpy.intersect1d(secondary_chasis_silicons, j_neighbors)) > 1:
                            secondary_bridging_oxygens.append(j)
                            cluster_atoms.insert(m, 'O')
                            cluster_coords.insert(m, cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))
                        else:
                            secondary_podal_oxygens.append(j)
                            cluster_atoms.append('O')
                            cluster_coords.append(cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))

            if reorder_podals:
                n = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens) + len(chasis_silicons)
                m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens) + len(chasis_silicons) + len(podal_oxygens)
                l = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens) + len(chasis_silicons) + len(podal_oxygens) + len(secondary_bridging_oxygens)
                p = l + len(secondary_chasis_silicons)
                q = l + len(secondary_chasis_silicons) + len(secondary_podal_oxygens)
                if len(secondary_bridging_oxygens) > 0:
                    # reordered = reorder_podal_oxygens(cluster_coords[p:q], cluster_coords[n+0], cluster_coords[n+1], cluster_coords[l+0], cluster_coords[l+1], cluster_coords[l-1])
                    reordered = reorder_podal_oxygens_v2(cluster_coords[p:q], cluster_coords[n:m], cluster_coords[l:p], cluster_coords[m:l])
                else:
                    # reordered = reorder_podal_oxygens(cluster_coords[p:q], cluster_coords[n+0], cluster_coords[n+1], cluster_coords[l+0], cluster_coords[l+1])
                    reordered = reorder_podal_oxygens_v2(cluster_coords[p:q], cluster_coords[n:m], cluster_coords[l:p])
                old_coords = cluster_coords
                old_podals = secondary_podal_oxygens
                cluster_coords = []
                secondary_podal_oxygens = []
                for coord in old_coords[:p]:
                    cluster_coords.append(coord)
                for i in reordered:
                    secondary_podal_oxygens.append(old_podals[i])
                    cluster_coords.append(old_coords[p:q][i])
                for coord in old_coords[q:]:
                    cluster_coords.append(coord)

            m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens) + len(chasis_silicons) + len(podal_oxygens) + len(secondary_bridging_oxygens) + len(secondary_chasis_silicons)
            secondary_podal_hydrogens = []
            if F_capping:
                for n, i in enumerate(secondary_podal_oxygens):
                    cluster_atoms[m + n] = 'F'
            else:
                for n, i in enumerate(secondary_podal_oxygens):
                    i_neighbors = bonds[1][bonds[0] == i]
                    for j in i_neighbors:
                        if atoms[j] == 'Si' and j not in chasis_silicons + secondary_chasis_silicons:
                            secondary_podal_hydrogens.append(j)
                            cluster_atoms.append('H')
                            axis = slab.get_distance(i, j, mic=True, vector=True)
                            axis = axis / numpy.linalg.norm(axis)
                            cluster_coords.append(cluster_coords[m + n] + axis * OH_bond_length)

            if reorient_clusters:
                n = len(peripheral_hydrogens)
                m = len(peripheral_hydrogens) + 1
                p = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens)
                q = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens) + 1
                origin = 0.5 * (cluster_coords[p] + cluster_coords[q])
                axes = numpy.empty((3, 3))
                axes[0] = cluster_coords[q] - cluster_coords[p]
                axes[0] = axes[0] / numpy.linalg.norm(axes[0])
                axes[2] = cluster_coords[n] + cluster_coords[m] - cluster_coords[p] - cluster_coords[q]
                axes[2] = axes[2] - axes[0] * numpy.inner(axes[0], axes[2])
                axes[2] = axes[2] / numpy.linalg.norm(axes[2])
                axes[1] = numpy.cross(axes[2], axes[0])
                cluster_coords = [numpy.einsum('ij,j->i', axes, coords - origin) for coords in cluster_coords]

            if reorder_atoms:
                n = len(peripheral_hydrogens)
                m = len(peripheral_hydrogens) + len(peripheral_oxygens)
                l = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(bridging_oxygens)
                p = l + len(chasis_silicons)
                q = l + len(chasis_silicons) + len(podal_oxygens)
                r = l + len(chasis_silicons) + len(podal_oxygens) + len(secondary_bridging_oxygens)
                u = r + len(secondary_chasis_silicons)
                v = r + len(secondary_chasis_silicons) + len(secondary_podal_oxygens)
                w = r + len(secondary_chasis_silicons) + len(secondary_podal_oxygens) + len(secondary_podal_hydrogens)
                w = list(range(v, w))
                v = list(range(u, v))
                u = list(range(r, u))
                r = list(range(q, r))
                q = list(range(p, q))
                p = list(range(l, p))
                l = list(range(m, l))
                m = list(range(n, m))
                n = list(range(0, n))
                indices = w + v + u + r + q + p + l + m + n
                cluster_atoms = [cluster_atoms[n] for n in indices]
                cluster_coords = [cluster_coords[n] for n in indices]

            cluster = Atoms(cluster_atoms, cluster_coords)
            secondary_clusters.append(cluster)

        return secondary_clusters

    def analyze_bonds(self, slab=None, bond_cutoffs=None):

        if slab is None:
            slab = self.slab
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs

        atoms = slab.get_chemical_symbols()
        bonds = neighbor_list('ij', slab, bond_cutoffs)

        coord_numbers = {'Si': 4, 'O': 2, 'H': 1}
        issues = []
        for i, X in enumerate(atoms):
            i_neighbors = bonds[1][bonds[0] == i]
            if len(i_neighbors) != coord_numbers[X]:
                issues.append([X, i, i_neighbors])

        print('Bonding Analysis')
        for X, i, i_neighbors in issues:
            message = '{:s}{:d} is bonded to'.format(X, i+1)
            for j in i_neighbors:
                message += ' {:s}{:d} {:f}'.format(atoms[j], j+1, slab.get_distance(i, j, mic=True))
            print(message)

        return

    def analyze_distances(self, file_path, slab=None):

        if slab is None:
            slab = self.slab

        atoms = slab.get_chemical_symbols()
        distances = slab.get_all_distances(mic=True)

        elements = ['Si', 'O', 'H']
        for n, A in enumerate(elements):
            for m, B in enumerate(elements):
                if n <= m:
                    if n == m:
                        AB_distances = [distances[i, j] for i, X in enumerate(atoms) for j, Y in enumerate(atoms) if i < j and X == A and Y == B]
                    else:
                        AB_distances = [distances[i, j] for i, X in enumerate(atoms) for j, Y in enumerate(atoms) if X == A and Y == B]
                    pyplot.figure()
                    pyplot.hist(AB_distances, bins=[i * 0.1 for i in range(0, 51)])
                    pyplot.xlabel('{:s}-{:s} Distances (Å)'.format(A, B))
                    pyplot.ylabel('Frequency')
                    pyplot.tight_layout()
                    pyplot.savefig(file_path.format(A, B))
                    pyplot.close()

        return

    def save_clusters(self, file_path, file_type='xyz', which_OH_pairs=None, which_clusters=None):
        if which_OH_pairs is None:
            which_OH_pairs = self.viable_OH_pairs
        if which_clusters is None:
            which_clusters = self.minimal_clusters
        for (a, b), cluster in zip(which_OH_pairs, which_clusters):
            write(file_path.format(a, b), cluster, file_type)
        return

