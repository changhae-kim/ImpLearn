import numpy


class Silanols():

    def __init__(self, file_path, file_type, bond_cutoffs=None, vicinal_cutoff=None, OH_bond_length=None):

        if bond_cutoffs is not None:
            self.bond_cutoffs = bond_cutoffs
        else:
            # self.bond_cutoffs = {('Si', 'Si'): 2.0, ('O', 'O'): 1.5, ('Si', 'O'): 2.3, ('O', 'H'): 1.15}
            self.bond_cutoffs = {('Si', 'Si'): 2.0, ('O', 'O'): 2.0, ('Si', 'O'): 2.3, ('O', 'H'): 1.2}

        if vicinal_cutoff is not None:
            self.vicinal_cutoff = vicinal_cutoff
        else:
            self.vicinal_cutoff = 4.5

        if OH_bond_length is not None:
            self.OH_bond_length = OH_bond_length
        else:
            self.OH_bond_length = 0.96

        self.slab, self.bonds = self.load_slab(file_path, file_type, self.bond_cutoffs)

        self.atoms = self.slab.get_chemical_symbols()
        self.coords = self.slab.get_positions()
        self.cell = self.slab.get_cell()
        self.pbc = self.slab.get_pbc()

        self.OH_groups = self.find_OH_groups()
        self.geminal_OH_pairs = self.find_geminal_OH_pairs()
        self.vicinal_OH_pairs = self.find_vicinal_OH_pairs(self.vicinal_cutoff)
        self.minimal_clusters = self.carve_minimal_clusters(self.OH_bond_length)

        return

    def load_slab(self, file_path, file_type, bond_cutoffs):
        from ase.io import read
        from ase.neighborlist import neighbor_list
        slab = read(file_path, 0, file_type)
        bonds = neighbor_list('ij', slab, bond_cutoffs)
        return slab, bonds

    def find_OH_groups(self, exclude_waters=True):
        OH_groups = []
        for i, X in enumerate(self.atoms):
            if X == 'H':
                H_neighbors = self.bonds[1][self.bonds[0] == i]
                if H_neighbors.shape[0] != 1:
                    print('find_OH_groups(): H {:d} is bonded to {:d} atoms'.format(i+1, H_neighbors.shape[0]))
                elif self.atoms[H_neighbors[0]] != 'O':
                    print('find_OH_groups(): H {:d} is bonded to {:s} {:d}'.format(i+1, self.atoms[H_neighbors[0]], H_neighbors[0]+1))
                for j in H_neighbors:
                    if self.atoms[j] == 'O':
                        O_neighbors = self.bonds[1][self.bonds[0] == j]
                        if O_neighbors.shape[0] != 2:
                            print('find_OH_groups(): O {:d} is bonded to {:d} atoms'.format(j+1, O_neighbors.shape[0]))
                        if 'Si' not in [self.atoms[i] for i in O_neighbors]:
                            if exclude_waters:
                                continue
                            else:
                                print('find_OH_groups(): O {:d} is not bonded to Si'.format(j+1))
                        OH_groups.append([i, j])
        return OH_groups

    def find_geminal_OH_pairs(self):
        geminal_OH_pairs = []
        for n, OH1_group in enumerate(self.OH_groups):
            for m, OH2_group in enumerate(self.OH_groups):
                if n > m:
                    O1_neighbors = self.bonds[1][self.bonds[0] == OH1_group[1]]
                    O2_neighbors = self.bonds[1][self.bonds[0] == OH2_group[1]]
                    if O1_neighbors.shape[0] != 2:
                        print('find_geminal_OH_pairs(): O {:d} is bonded to {:d} atoms'.format(OH1_group[1]+1, O1_neighbors.shape[0]))
                    elif 'Si' not in [self.atoms[i] for i in O1_neighbors]:
                        print('find_geminal_OH_pairs(): O {:d} is not bonded to Si'.format(OH1_group[1]+1))
                    if O2_neighbors.shape[0] != 2:
                        print('find_geminal_OH_pairs(): O {:d} is bonded to {:d} atoms'.format(OH2_group[1]+1, O2_neighbors.shape[0]))
                    elif 'Si' not in [self.atoms[i] for i in O2_neighbors]:
                        print('find_geminal_OH_pairs(): O {:d} is not bonded to Si'.format(OH2_group[1]+1))
                    for i in numpy.intersect1d(O1_neighbors, O2_neighbors):
                        if self.atoms[i] == 'Si':
                            geminal_OH_pairs.append([OH1_group, OH2_group])
        return geminal_OH_pairs

    def find_vicinal_OH_pairs(self, vicinal_cutoff, exclude_geminals=True):
        vicinal_OH_pairs = []
        for n, OH1_group in enumerate(self.OH_groups):
            for m, OH2_group in enumerate(self.OH_groups):
                if n > m:
                    if self.slab.get_distance(OH1_group[1], OH2_group[1], mic=True) < vicinal_cutoff:
                        if exclude_geminals:
                            geminal = False
                            for (OH3_group, OH4_group) in self.geminal_OH_pairs:
                                if OH1_group[1] in [OH3_group[1], OH4_group[1]] or OH2_group[1] in [OH3_group[1], OH4_group[1]]:
                                    geminal = True
                                    break
                            if geminal:
                                continue
                        vicinal_OH_pairs.append([OH1_group, OH2_group])
        return vicinal_OH_pairs

    def carve_minimal_clusters(self, OH_bond_length, reorder_podal_atoms=True):

        from ase import Atoms
        minimal_clusters = []
        for (OH1_group, OH2_group) in self.vicinal_OH_pairs:

            atoms = []
            coords = []

            peripheral_hydrogens = [OH1_group[0], OH2_group[0]]
            atoms.append('H')
            coords.append(numpy.zeros((3,)))
            atoms.append('H')
            coords.append(coords[0] + self.slab.get_distance(OH1_group[0], OH2_group[0], mic=True, vector=True))

            peripheral_oxygens = [OH1_group[1], OH2_group[1]]
            atoms.append('O')
            coords.append(coords[0] + self.slab.get_distance(OH1_group[0], OH1_group[1], mic=True, vector=True))
            atoms.append('O')
            coords.append(coords[1] + self.slab.get_distance(OH2_group[0], OH2_group[1], mic=True, vector=True))

            chasis_silicons = []
            for n, i in enumerate(peripheral_oxygens):
                i_neighbors = self.bonds[1][self.bonds[0] == i]
                if i_neighbors.shape[0] != 2:
                    print('carve_minimal_clusters(): O {:d} is bonded to {:d} atoms'.format(i+1, i_neighbors.shape[0]))
                elif 'Si' not in [self.atoms[j] for j in i_neighbors]:
                    print('carve_minimal_clusters(): O {:d} is not bonded to Si'.format(i+1))
                for j in i_neighbors:
                    if self.atoms[j] == 'Si' and j not in chasis_silicons:
                        chasis_silicons.append(j)
                        atoms.append('Si')
                        coords.append(coords[len(peripheral_hydrogens) + n] + self.slab.get_distance(i, j, mic=True, vector=True))

            chasis_oxygens = []
            podal_oxygens = []
            for n, i in enumerate(chasis_silicons):
                i_neighbors = self.bonds[1][self.bonds[0] == i]
                if i_neighbors.shape[0] != 4:
                    print('carve_minimal_clusters(): Si {:d} is bonded to {:d} atoms'.format(i+1, i_neighbors.shape[0]))
                else:
                    for j in i_neighbors:
                        if self.atoms[j] != 'O':
                            print('carve_minimal_clusters(): Si {:d} is bonded to {:s} {:d}'.format(i+1, self.atoms[j], j+1))
                for j in i_neighbors:
                    if self.atoms[j] == 'O' and j not in peripheral_oxygens + chasis_oxygens + podal_oxygens:
                        j_neighbors = self.bonds[1][self.bonds[0] == j]
                        if j_neighbors.shape[0] == 2:
                            m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(chasis_oxygens)
                            if numpy.intersect1d(chasis_silicons, j_neighbors).shape[0] >= 2:
                                chasis_oxygens.append(j)
                                atoms.insert(m, 'O')
                                coords.insert(m, coords[m + n] + self.slab.get_distance(i, j, mic=True, vector=True))
                            else:
                                podal_oxygens.append(j)
                                atoms.append('O')
                                coords.append(coords[m + n] + self.slab.get_distance(i, j, mic=True, vector=True))

            if reorder_podal_atoms:
                n = len(peripheral_hydrogens)
                m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(chasis_oxygens)
                p = m + len(chasis_silicons)
                q = m + len(chasis_silicons) + len(podal_oxygens)
                ordered = self.reorder_right_hand_rule(coords[p:q], coords[n+0], coords[n+1], coords[m+0], coords[m+1])
                old_podals = podal_oxygens
                old_coords = coords
                podal_oxygens = []
                coords = []
                for coord in old_coords[:p]:
                    coords.append(coord)
                for i in ordered:
                    podal_oxygens.append(old_podals[i])
                    coords.append(old_coords[p:q][i])
                for coord in old_coords[q:]:
                    coords.append(coord)

            podal_hydrogens = []
            for n, i in enumerate(podal_oxygens):
                i_neighbors = self.bonds[1][self.bonds[0] == i]
                if i_neighbors.shape[0] != 2:
                    print('carve_minimal_clusters(): O {:d} is bonded to {:d} atoms'.format(i+1, i_neighbors.shape[0]))
                else:
                    for j in i_neighbors:
                        if self.atoms[j] == 'O':
                            print('carve_minimal_clusters(): O {:d} is bonded to O {:d}'.format(i+1, self.atoms[j], j+1))
                for j in i_neighbors:
                    if j not in peripheral_hydrogens + chasis_silicons:
                        m = len(peripheral_hydrogens) + len(peripheral_oxygens) + len(chasis_oxygens) + len(chasis_silicons)
                        if self.atoms[j] == 'H':
                            podal_hydrogens.append(j)
                            atoms.append('H')
                            coords.append(coords[m + n] + self.slab.get_distance(i, j, mic=True, vector=True))
                        elif self.atoms[j] == 'Si':
                            podal_hydrogens.append(j)
                            atoms.append('H')
                            axis = self.slab.get_distance(i, j, mic=True, vector=True)
                            axis = axis / numpy.linalg.norm(axis)
                            coords.append(coords[m + n] + axis * OH_bond_length)

            cluster = Atoms(atoms, coords)
            minimal_clusters.append(cluster)

        return minimal_clusters

    def reorder_right_hand_rule(self, coords, O1_coord, O2_coord, Si1_coord, Si2_coord):

        origin = 0.5 * (Si1_coord + Si2_coord)
        centered = coords - origin
        xaxis = Si2_coord - Si1_coord
        xaxis = xaxis / numpy.linalg.norm(xaxis)
        zaxis = 0.5 * (O1_coord + O2_coord - Si1_coord - Si2_coord)
        zaxis = zaxis / numpy.linalg.norm(zaxis)
        yaxis = numpy.cross(zaxis, xaxis)

        if len(coords) == 2:
            ordered = []
            ranking = numpy.argsort(numpy.einsum('ij,j->i', centered, + xaxis)),
            for i in ranking:
                ordered.append(i)

        elif len(coords) == 4:
            ordered = []
            rankings = [
                    numpy.argsort(numpy.einsum('ij,j->i', centered, + xaxis - yaxis)),
                    numpy.argsort(numpy.einsum('ij,j->i', centered, + xaxis + yaxis)),
                    numpy.argsort(numpy.einsum('ij,j->i', centered, - xaxis + yaxis)),
                    numpy.argsort(numpy.einsum('ij,j->i', centered, - xaxis - yaxis)),
                    ]
            for ranking in rankings:
                for i in ranking:
                    if i not in ordered:
                        ordered.append(i)
                        break

        elif len(coords) == 6:
            centered1 = coords - Si1_coord
            zaxis1 = O1_coord - Si1_coord
            zaxis1 = zaxis1 / numpy.linalg.norm(zaxis1)
            ordered1 = [i for i in range(0, 3)]
            if numpy.dot(numpy.cross(zaxis1, centered1[ordered1[1]]), centered1[ordered1[2]]) < 0.0:
                ordered1[1], ordered1[2] = ordered1[2], ordered1[1]
            centered2 = coords - Si2_coord
            zaxis2 = O2_coord - Si2_coord
            zaxis2 = zaxis2 / numpy.linalg.norm(zaxis2)
            ordered2 = [i for i in range(3, 6)]
            if numpy.dot(numpy.cross(zaxis2, centered2[ordered2[1]]), centered2[ordered2[2]]) < 0.0:
                ordered2[1], ordered2[2] = ordered2[2], ordered2[1]
            dists = [[numpy.linalg.norm(centered[i]-centered[j]) for j in ordered2] for i in ordered1]
            nm = numpy.argmin(dists)
            n = nm//3
            m = nm%3
            ordered = [ordered1[(n+1+i)%3] for i in range(3)] + [ordered2[(m+i)%3] for i in range(3)]

        return ordered

    def export_clusters(self, file_path, file_type):
        from ase.io import write
        for i, cluster in enumerate(self.minimal_clusters):
            write(file_path.format(i), cluster, format=file_type)
        return

    def analyze_distances(self, file_path):
        from matplotlib import pyplot
        distances = self.slab.get_all_distances(mic=True)
        elements = ['Si', 'O', 'H']
        for n, A in enumerate(elements):
            for m, B in enumerate(elements):
                if n <= m:
                    if n == m:
                        AB_distances = [distances[i, j] for i, X in enumerate(self.atoms) for j, Y in enumerate(self.atoms) if i < j and X == A and Y == B]
                    else:
                        AB_distances = [distances[i, j] for i, X in enumerate(self.atoms) for j, Y in enumerate(self.atoms) if X == A and Y == B]
                    pyplot.figure()
                    pyplot.hist(AB_distances, bins=[i*0.1 for i in range(51)])
                    pyplot.xlabel(r'{:s}-{:s} Distances (\AA)'.format(A, B))
                    pyplot.ylabel('Frequency')
                    pyplot.tight_layout()
                    pyplot.savefig(file_path.format(A, B))
        return


if __name__ == '__main__':

    clusters = Silanols('/mnt/c/Users/changhae/Documents/UIUC/Amorphous Catalysts/Slabs/Tielens/A_117SiO2_35H2O', 'vasp')
    print('--- MAIN ---')
    print('clusters.atoms')
    print(clusters.atoms)
    print('clusters.bonds')
    print(clusters.bonds)
    print('clusters.coords')
    print(clusters.coords)
    print('clusters.OH_groups')
    print(len(clusters.OH_groups))
    print(clusters.OH_groups)
    print('clusters.geminal_OH_pairs')
    print(len(clusters.geminal_OH_pairs))
    print(clusters.geminal_OH_pairs)
    print('clusters.vicinal_OH_pairs')
    print(len(clusters.vicinal_OH_pairs))
    print(clusters.vicinal_OH_pairs)
    clusters.analyze_distances('A_d{:s}{:s}.png')
    clusters.export_clusters('A_s{:04d}.xyz', 'xyz')

