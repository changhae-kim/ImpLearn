import numpy

from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list
from matplotlib import pyplot


class Silanols():

    def __init__(self, file_path, file_type,
            bond_cutoffs = {('Si', 'Si'): 2.0, ('O', 'O'): 2.0, ('Si', 'O'): 2.3, ('O', 'H'): 1.2},
            viable_cutoff = 4.5,
            OH_bond_length = 0.96
            ):

        self.slab = self.load_slab(file_path, file_type)

        self.bond_cutoffs = bond_cutoffs
        self.viable_cutoff = viable_cutoff
        self.OH_bond_length = OH_bond_length

        self.OH_groups = self.find_OH_groups()
        self.geminal_OH_pairs = self.find_geminal_OH_pairs()
        self.vicinal_OH_pairs = self.find_vicinal_OH_pairs()
        self.viable_OH_pairs = self.find_viable_OH_pairs()
        self.minimal_clusters = self.carve_minimal_clusters()

        return

    def load_slab(self, file_path, file_type):
        slab = read(file_path, 0, file_type)
        return slab

    def find_OH_groups(self, slab=None, bond_cutoffs=None, exclude_waters=True):

        if slab is None:
            slab = self.slab
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs

        atoms = slab.get_chemical_symbols()
        bonds = neighbor_list('ij', slab, bond_cutoffs)

        OH_groups = []
        for i, X in enumerate(atoms):
            if X == 'H':
                H_neighbors = bonds[1][bonds[0] == i]
                if H_neighbors.shape[0] != 1:
                    print('find_OH_groups(): H {:d} is bonded to {:d} atoms'.format(i+1, H_neighbors.shape[0]))
                for j in H_neighbors:
                    if atoms[j] != 'O':
                        print('find_OH_groups(): H {:d} is bonded to {:s} {:d}'.format(i+1, atoms[j], j+1))
                for j in H_neighbors:
                    if atoms[j] == 'O':
                        O_neighbors = bonds[1][bonds[0] == j]
                        if O_neighbors.shape[0] != 2:
                            print('find_OH_groups(): O {:d} is bonded to {:d} atoms'.format(j+1, O_neighbors.shape[0]))
                        for k in O_neighbors:
                            if atoms[k] not in ['Si', 'H']:
                                print('find_OH_groups(): O {:d} is bonded to {:s} {:d}'.format(j+1, atoms[k], k+1))
                        if exclude_waters:
                            if 'Si' not in [atoms[i] for i in O_neighbors]:
                                continue
                            else:
                                OH_groups.append([i, j])
                        else:
                            if 'Si' not in [atoms[i] for i in O_neighbors]:
                                print('find_OH_groups(): O {:d} is not bonded to Si'.format(j+1))
                            OH_groups.append([i, j])

        return OH_groups

    def find_geminal_OH_pairs(self, slab=None, bond_cutoffs=None, OH_groups=None):

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
                if n > m:
                    O1_neighbors = bonds[1][bonds[0] == OH1_group[1]]
                    O2_neighbors = bonds[1][bonds[0] == OH2_group[1]]
                    if O1_neighbors.shape[0] != 2:
                        print('find_geminal_OH_pairs(): O {:d} is bonded to {:d} atoms'.format(OH1_group[1]+1, O1_neighbors.shape[0]))
                    for i in O1_neighbors:
                        if atoms[i] not in ['Si', 'H']:
                            print('find_geminal_OH_pairs(): O {:d} is bonded to {:s} {:d}'.format(OH1_group[1]+1, atoms[i], i+1))
                    if 'Si' not in [atoms[i] for i in O1_neighbors]:
                        print('find_geminal_OH_pairs(): O {:d} is not bonded to Si'.format(OH1_group[1]+1))
                    if O2_neighbors.shape[0] != 2:
                        print('find_geminal_OH_pairs(): O {:d} is bonded to {:d} atoms'.format(OH2_group[1]+1, O2_neighbors.shape[0]))
                    for i in O2_neighbors:
                        if atoms[i] not in ['Si', 'H']:
                            print('find_geminal_OH_pairs(): O {:d} is bonded to {:s} {:d}'.format(OH2_group[1]+1, atoms[i], i+1))
                    if 'Si' not in [atoms[i] for i in O2_neighbors]:
                        print('find_geminal_OH_pairs(): O {:d} is not bonded to Si'.format(OH2_group[1]+1))
                    for i in numpy.intersect1d(O1_neighbors, O2_neighbors):
                        if atoms[i] == 'Si':
                            geminal_OH_pairs.append([OH1_group, OH2_group])

        return geminal_OH_pairs

    def find_vicinal_OH_pairs(self, slab=None, bond_cutoffs=None, OH_groups=None):

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
                if n > m:
                    O1_neighbors = bonds[1][bonds[0] == OH1_group[1]]
                    O2_neighbors = bonds[1][bonds[0] == OH2_group[1]]
                    if O1_neighbors.shape[0] != 2:
                        print('find_vicinal_OH_pairs(): O {:d} is bonded to {:d} atoms'.format(OH1_group[1]+1, O1_neighbors.shape[0]))
                    for i in O1_neighbors:
                        if atoms[i] not in ['Si', 'H']:
                            print('find_vicinal_OH_pairs(): O {:d} is bonded to {:s} {:d}'.format(OH1_group[1]+1, atoms[i], i+1))
                    if 'Si' not in [atoms[i] for i in O1_neighbors]:
                        print('find_vicinal_OH_pairs(): O {:d} is not bonded to Si'.format(OH1_group[1]+1))
                    if O2_neighbors.shape[0] != 2:
                        print('find_vicinal_OH_pairs(): O {:d} is bonded to {:d} atoms'.format(OH2_group[1]+1, O2_neighbors.shape[0]))
                    for i in O2_neighbors:
                        if atoms[i] not in ['Si', 'H']:
                            print('find_vicinal_OH_pairs(): O {:d} is bonded to {:s} {:d}'.format(OH2_group[1]+1, atoms[i], i+1))
                    if 'Si' not in [atoms[i] for i in O2_neighbors]:
                        print('find_vicinal_OH_pairs(): O {:d} is not bonded to Si'.format(OH2_group[1]+1))
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
                    if len(Si1_candidates) != 1:
                        print('find_vicinal_OH_pairs(): O {:d} is bonded to {:d} Si atoms'.format(OH1_group[1]+1, len(Si1_candidates)))
                    for i, Si1_neighbors in zip(Si1_candidates, Si1_candidate_neighbors):
                        if Si1_neighbors.shape[0] != 4:
                            print('find_vicinal_OH_pairs(): Si {:d} is bonded to {:d} atoms'.format(i+1, Si1_neighbors.shape[0]))
                        for j in Si1_neighbors:
                            if atoms[j] != 'O':
                                print('find_vicinal_OH_pairs(): Si {:d} is bonded to {:s} {:d}'.format(i+1, atoms[j], j+1))
                    if len(Si2_candidates) != 1:
                        print('find_vicinal_OH_pairs(): O {:d} is bonded to {:d} Si atoms'.format(OH2_group[1]+1, len(Si2_candidates)))
                    for i, Si2_neighbors in zip(Si2_candidates, Si2_candidate_neighbors):
                        if Si2_neighbors.shape[0] != 4:
                            print('find_vicinal_OH_pairs(): Si {:d} is bonded to {:d} atoms'.format(i+1, Si2_neighbors.shape[0]))
                        for j in Si2_neighbors:
                            if atoms[j] != 'O':
                                print('find_vicinal_OH_pairs(): Si {:d} is bonded to {:s} {:d}'.format(i+1, atoms[j], j+1))
                    for i in numpy.intersect1d(numpy.concatenate(Si1_candidate_neighbors), numpy.concatenate(Si2_candidate_neighbors)):
                        if atoms[i] == 'O':
                            vicinal_OH_pairs.append([OH1_group, OH2_group])

        return vicinal_OH_pairs

    def find_viable_OH_pairs(self, slab=None, OH_groups=None, viable_cutoff=None,
            exclude_geminals=True, geminal_OH_pairs=None, exclude_vicinals=False, vicinal_OH_pairs=None):

        if slab is None:
            slab = self.slab
        if OH_groups is None:
            OH_groups = self.OH_groups
        if viable_cutoff is None:
            viable_cutoff = self.viable_cutoff
        if geminal_OH_pairs is None:
            geminal_OH_pairs = self.geminal_OH_pairs
        if vicinal_OH_pairs is None:
            vicinal_OH_pairs = self.vicinal_OH_pairs

        viable_OH_pairs = []
        for n, OH1_group in enumerate(OH_groups):
            for m, OH2_group in enumerate(OH_groups):
                if n > m:
                    if slab.get_distance(OH1_group[1], OH2_group[1], mic=True) < viable_cutoff:
                        if exclude_geminals:
                            geminal = False
                            for (OH3_group, OH4_group) in geminal_OH_pairs:
                                if OH1_group[1] in [OH3_group[1], OH4_group[1]] or OH2_group[1] in [OH3_group[1], OH4_group[1]]:
                                    geminal = True
                                    break
                            if geminal:
                                continue
                        if exclude_vicinals:
                            vicinal = False
                            for (OH3_group, OH4_group) in vicinal_OH_pairs:
                                if OH1_group[1] in [OH3_group[1], OH4_group[1]] and OH2_group[1] in [OH3_group[1], OH4_group[1]]:
                                    vicinal = True
                                    break
                            if vicinal:
                                continue
                        viable_OH_pairs.append([OH1_group, OH2_group])

        return viable_OH_pairs

    def carve_minimal_clusters(self, slab=None, bond_cutoffs=None, viable_OH_pairs=None, OH_bond_length=None, reorder_podals=True):

        if slab is None:
            slab = self.slab
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if viable_OH_pairs is None:
            viable_OH_pairs = self.viable_OH_pairs
        if OH_bond_length is None:
            OH_bond_length = self.OH_bond_length

        atoms = slab.get_chemical_symbols()
        bonds = neighbor_list('ij', slab, bond_cutoffs)

        minimal_clusters = []
        for (OH1_group, OH2_group) in viable_OH_pairs:

            cluster_atoms = []
            cluster_coords = []

            peripheral_hydrogens = [OH1_group[0], OH2_group[0]]
            cluster_atoms.append('H')
            cluster_coords.append(numpy.zeros((3,)))
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
                if i_neighbors.shape[0] != 2:
                    print('carve_minimal_clusters(): O {:d} is bonded to {:d} atoms'.format(i+1, i_neighbors.shape[0]))
                for j in i_neighbors:
                    if atoms[j] not in ['Si', 'H']:
                        print('carve_minimal_clusters(): O {:d} is bonded to {:s} {:d}'.format(i+1, atoms[j], j+1))
                if 'Si' not in [atoms[j] for j in i_neighbors]:
                    print('carve_minimal_clusters(): O {:d} is not bonded to Si'.format(i+1))
                for j in i_neighbors:
                    if atoms[j] == 'Si' and j not in chasis_silicons:
                        chasis_silicons.append(j)
                        cluster_atoms.append('Si')
                        cluster_coords.append(cluster_coords[len(peripheral_hydrogens) + n] + slab.get_distance(i, j, mic=True, vector=True))

            chasis_oxygens = []
            podal_oxygens = []
            for n, i in enumerate(chasis_silicons):
                i_neighbors = bonds[1][bonds[0] == i]
                if i_neighbors.shape[0] != 4:
                    print('carve_minimal_clusters(): Si {:d} is bonded to {:d} atoms'.format(i+1, i_neighbors.shape[0]))
                for j in i_neighbors:
                    if atoms[j] != 'O':
                        print('carve_minimal_clusters(): Si {:d} is bonded to {:s} {:d}'.format(i+1, atoms[j], j+1))
                for j in i_neighbors:
                    if atoms[j] == 'O' and j not in peripheral_oxygens + chasis_oxygens + podal_oxygens:
                        j_neighbors = bonds[1][bonds[0] == j]
                        m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(chasis_oxygens)
                        if numpy.intersect1d(chasis_silicons, j_neighbors).shape[0] >= 2:
                            chasis_oxygens.append(j)
                            cluster_atoms.insert(m, 'O')
                            cluster_coords.insert(m, cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))
                        else:
                            podal_oxygens.append(j)
                            cluster_atoms.append('O')
                            cluster_coords.append(cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))

            if reorder_podals:
                n = len(peripheral_hydrogens)
                m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(chasis_oxygens)
                p = m + len(chasis_silicons)
                q = m + len(chasis_silicons) + len(podal_oxygens)
                reordered = self.reorder_podal_oxygens(cluster_coords[p:q], cluster_coords[n+0], cluster_coords[n+1], cluster_coords[m+0], cluster_coords[m+1])
                old_podals = podal_oxygens
                old_coords = cluster_coords
                podal_oxygens = []
                cluster_coords = []
                for coord in old_coords[:p]:
                    cluster_coords.append(coord)
                for i in reordered:
                    podal_oxygens.append(old_podals[i])
                    cluster_coords.append(old_coords[p:q][i])
                for coord in old_coords[q:]:
                    cluster_coords.append(coord)

            podal_hydrogens = []
            for n, i in enumerate(podal_oxygens):
                i_neighbors = bonds[1][bonds[0] == i]
                if i_neighbors.shape[0] != 2:
                    print('carve_minimal_clusters(): O {:d} is bonded to {:d} atoms'.format(i+1, i_neighbors.shape[0]))
                for j in i_neighbors:
                    if atoms[j] not in ['Si', 'H']:
                        print('carve_minimal_clusters(): O {:d} is bonded to {:s} {:d}'.format(i+1, atoms[j], j+1))
                for j in i_neighbors:
                    if j not in peripheral_hydrogens + chasis_silicons:
                        m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(chasis_oxygens) + len(chasis_silicons)
                        if atoms[j] == 'H':
                            podal_hydrogens.append(j)
                            cluster_atoms.append('H')
                            cluster_coords.append(cluster_coords[m + n] + slab.get_distance(i, j, mic=True, vector=True))
                        elif atoms[j] == 'Si':
                            podal_hydrogens.append(j)
                            cluster_atoms.append('H')
                            axis = slab.get_distance(i, j, mic=True, vector=True)
                            axis = axis / numpy.linalg.norm(axis)
                            cluster_coords.append(cluster_coords[m + n] + axis * OH_bond_length)

            cluster = Atoms(cluster_atoms, cluster_coords)
            minimal_clusters.append(cluster)

        return minimal_clusters

    def reorder_podal_oxygens(self, coords, O1_coord, O2_coord, Si1_coord, Si2_coord, max_iter=50):

        origin = 0.5 * (Si1_coord + Si2_coord)
        centered = coords - origin
        xaxis = Si2_coord - Si1_coord
        xaxis = xaxis / numpy.linalg.norm(xaxis)
        zaxis = 0.5 * (O1_coord + O2_coord - Si1_coord - Si2_coord)
        zaxis = zaxis / numpy.linalg.norm(zaxis)
        yaxis = numpy.cross(zaxis, xaxis)

        if len(coords) not in [2, 4, 6]:
            print('number of podal oxygens is {:d}'.format(len(coords)))

        centered1 = coords - Si1_coord
        centered2 = coords - Si2_coord
        reordered1 = []
        reordered2 = []
        for i, (coord1, coord2) in enumerate(zip(centered1, centered2)):
            if numpy.linalg.norm(coord1) < numpy.linalg.norm(coord2):
                reordered1.append(i)
            else:
                reordered2.append(i)

        zaxis1 = O1_coord - Si1_coord
        zaxis1 = zaxis1 / numpy.linalg.norm(zaxis1)
        status = -1
        for i in range(max_iter):
            if status == 0:
                break
            else:
                status = 0
                for i, _ in enumerate(reordered1[:-1]):
                    if numpy.dot(numpy.cross(zaxis1, centered1[reordered1[i]]), centered1[reordered1[i+1]]) < 0.0:
                        reordered1[i], reordered1[i+1] = reordered1[i+1], reordered1[i]
                        status = -1
                        break

        zaxis2 = O2_coord - Si2_coord
        zaxis2 = zaxis2 / numpy.linalg.norm(zaxis2)
        status = -1
        for i in range(max_iter):
            if status == 0:
                break
            else:
                status = 0
                for i, _ in enumerate(reordered2[:-1]):
                    if numpy.dot(numpy.cross(zaxis2, centered2[reordered2[i]]), centered2[reordered2[i+1]]) < 0.0:
                        reordered2[i], reordered2[i+1] = reordered2[i+1], reordered2[i]
                        status = -1
                        break

        if len(reordered1) > 2:
            if len(reordered2) > 2:
                nm = numpy.argmin([numpy.linalg.norm(centered[i]-centered[j]) for i in reordered1 for j in reordered2])
                n = nm//len(reordered2)
                m = nm%len(reordered2)
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
                reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
            else:
                n = numpy.argmin([numpy.linalg.norm(centered[i]-centered[0]) for i in reordered1])
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
        elif len(reordered2) > 2:
            m = numpy.argmin([numpy.linalg.norm(centered[reordered1[-1]]-centered[j]) for j in reordered2])
            reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
        reordered = reordered1 + reordered2

        return reordered

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
                    pyplog.close()

        return

    def save_clusters(self, file_path, file_type, minimal_clusters=None):
        if minimal_clusters is None:
            minimal_clusters = self.minimal_clusters
        for i, cluster in enumerate(minimal_clusters):
            write(file_path.format(i), cluster, file_type)
        return


if __name__ == '__main__':


    clusters = Silanols('tests/A_117SiO2_35H2O', 'vasp')
    print('--- MAIN ---')
    print('atoms')
    print(clusters.slab.get_chemical_symbols())
    print('bonds')
    print(list(zip(*neighbor_list('ij', clusters.slab, clusters.bond_cutoffs))))
    print('OH_groups')
    print(len(clusters.OH_groups))
    print(clusters.OH_groups)
    print('geminal_OH_pairs')
    print(len(clusters.geminal_OH_pairs))
    print(clusters.geminal_OH_pairs)
    print('vicinal_OH_pairs')
    print(len(clusters.vicinal_OH_pairs))
    print(clusters.vicinal_OH_pairs)
    print('viable_OH_pairs')
    print(len(clusters.viable_OH_pairs))
    print(clusters.viable_OH_pairs)
    clusters.analyze_distances('A_d{:s}{:s}.png')
    clusters.save_clusters('A_{:04d}.xyz', 'xyz')



