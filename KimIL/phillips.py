import numpy

from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list


def rotate_vector(vector, axis, angle, degrees=True):
    unit = axis / numpy.linalg.norm(axis)
    parallel = numpy.inner(vector, unit) * unit
    perpend1 = vector - parallel
    perpend2 = numpy.cross(unit, perpend1)
    if degrees:
        rotated = parallel + perpend1 * numpy.cos(numpy.pi*angle/180.0) + perpend2 * numpy.sin(numpy.pi*angle/180.0)
    else:
        rotated = parallel + perpend1 * numpy.cos(angle) + perpend2 * numpy.sin(angle)
    return rotated


class Phillips():

    def __init__(self, file_path, file_type,
            peripheral_oxygens=[2, 3],
            alkyl_lengths=[4, 6],
            bond_cutoffs={
                ('Si', 'Si'): 2.0, ('O', 'O'): 2.0, ('Si', 'O'): 2.3, ('O', 'H'): 1.2,
                ('Cr', 'O'): 2.3, ('Cr', 'C'): 2.3, ('C', 'C'): 2.0, ('C', 'H'): 1.2
                },
            bond_lengths={('Cr', 'O'): 1.82, ('Cr', 'C'): 2.02, ('C', 'C'): 1.53, ('C', 'H'): 1.09},
            ethylene_bond_lengths={('Cr', 'C'): 2.5, ('C', 'C'): 1.34, ('C', 'H'): 1.09},
            transition_state_lengths={('Cr', 'C'): 2.1, ('C', 'C'): 2.2, ('C', 'Cr'): 2.1}
            ):

        self.bond_cutoffs = bond_cutoffs
        self.bond_lengths = bond_lengths
        self.ethylene_bond_lengths = ethylene_bond_lengths
        self.transition_state_lengths = transition_state_lengths

        self.cluster = self.load_cluster(file_path, file_type)

        self.alkyl_lengths = alkyl_lengths
        self.peripheral_oxygens = []
        for i, n in enumerate(peripheral_oxygens):
            if n < 0:
                self.peripheral_oxygens.append(len(self.slab.get_chemical_symbols()) + n)
            else:
                self.peripheral_oxygens.append(n)

        self.axes = self.define_axes(self.cluster)
        self.chromium_cluster = self.attach_chromium(self.cluster)
        self.L_butyl_cluster = self.attach_alkyl(self.chromium_cluster, self.alkyl_lengths[0], point_y=True, rotate_2=False)
        self.L_butyl_R_ethylene_cluster = self.attach_ethylene(self.L_butyl_cluster, point_y=False)
        self.LR_transition_state_cluster = self.attach_transition_state(self.chromium_cluster, self.alkyl_lengths[1], point_y=False)
        self.R_hexyl_cluster = self.attach_alkyl(self.chromium_cluster, self.alkyl_lengths[1], point_y=False, rotate_2=True)
        self.R_butyl_cluster = self.attach_alkyl(self.chromium_cluster, self.alkyl_lengths[0], point_y=False, rotate_2=False)
        self.R_butyl_L_ethylene_cluster = self.attach_ethylene(self.R_butyl_cluster, point_y=True)
        self.RL_transition_state_cluster = self.attach_transition_state(self.chromium_cluster, self.alkyl_lengths[1], point_y=True)
        self.L_hexyl_cluster = self.attach_alkyl(self.chromium_cluster, self.alkyl_lengths[1], point_y=True, rotate_2=True)

        return

    def load_cluster(self, file_path, file_type):
        cluster = read(file_path, 0, file_type)
        return cluster

    def define_axes(self, cluster, peripheral_oxygens=None, bond_cutoffs=None):

        if peripheral_oxygens is None:
            peripheral_oxygens = self.peripheral_oxygens
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        chasis_silicons = []
        for i in peripheral_oxygens:
            i_neighbors = bonds[1][bonds[0] == i]
            for j in i_neighbors:
                if atoms[j] == 'Si':
                    chasis_silicons.append(j)

        n, m = peripheral_oxygens
        p, q = chasis_silicons
        axes = numpy.empty((3, 3))
        axes[0] = coords[p] - coords[q]
        axes[0] = axes[0] / numpy.linalg.norm(axes[0])
        axes[2] = coords[n] + coords[m] - coords[p] - coords[q]
        axes[2] = axes[2] - numpy.inner(axes[2], axes[0]) * axes[0]
        axes[2] = axes[2] / numpy.linalg.norm(axes[2])
        axes[1] = numpy.cross(axes[2], axes[0])

        return axes

    def attach_chromium(self, cluster, peripheral_oxygens=None, bond_cutoffs=None, bond_lengths=None, axes=None, max_iter=50):

        if peripheral_oxygens is None:
            peripheral_oxygens = self.peripheral_oxygens
        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if bond_lengths is None:
            bond_lengths = self.bond_lengths
        if axes is None:
            axes = self.axes

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        peripheral_hydrogens = []
        chasis_silicons = []
        for i in peripheral_oxygens:
            i_neighbors = bonds[1][bonds[0] == i]
            for j in i_neighbors:
                if atoms[j] == 'H':
                    peripheral_hydrogens.append(j)
                elif atoms[j] == 'Si':
                    chasis_silicons.append(j)

        n, m = peripheral_oxygens
        p, q = chasis_silicons
        r, s = peripheral_hydrogens
        SiO1_bond_length = cluster.get_distance(n, p)
        SiO2_bond_length = cluster.get_distance(m, q)
        O1_coord = coords[p] + axes[2] * SiO1_bond_length
        O2_coord = coords[q] + axes[2] * SiO2_bond_length

        need_new_axes = False
        status = -1
        for i in range(max_iter):
            if status == 0:
                continue
            else:
                status = 0
                for j, (X, coord) in enumerate(zip(atoms, coords)):
                    if j not in [n, p, r]:
                        if X == 'Si':
                            OX_bond_cutoff = bond_cutoffs[('Si', 'O')]
                        elif X == 'O':
                            OX_bond_cutoff = bond_cutoffs[('O', 'O')]
                        elif X == 'H':
                            OX_bond_cutoff = bond_cutoffs[('O', 'H')]
                        OX_vec = O1_coord - coord
                        OX_dist = numpy.linalg.norm(OX_vec)
                        if OX_dist < OX_bond_cutoff:
                            O1_coord = O1_coord + OX_vec * 1.05 * OX_bond_cutoff / OX_dist
                            SiO_vec = O1_coord - coords[p]
                            SiO_dist = numpy.linalg.norm(SiO_vec)
                            O1_coord = coords[p] + SiO_vec * SiO1_bond_length / SiO_dist
                            need_new_axes = True
                            status = -1
                            break
        status = -1
        for i in range(max_iter):
            if status == 0:
                continue
            else:
                status = 0
                for j, (X, coord) in enumerate(zip(atoms, coords)):
                    if j not in [m, q, s]:
                        if X == 'Si':
                            OX_bond_cutoff = bond_cutoffs[('Si', 'O')]
                        elif X == 'O':
                            OX_bond_cutoff = bond_cutoffs[('O', 'O')]
                        elif X == 'H':
                            OX_bond_cutoff = bond_cutoffs[('O', 'H')]
                        OX_vec = O2_coord - coord
                        OX_dist = numpy.linalg.norm(OX_vec)
                        if OX_dist < OX_bond_cutoff:
                            O2_coord = O2_coord + OX_vec * 1.05 * OX_bond_cutoff / OX_dist
                            SiO_vec = O2_coord - coords[q]
                            SiO_dist = numpy.linalg.norm(SiO_vec)
                            O2_coord = coords[q] + SiO_vec * SiO2_bond_length / SiO_dist
                            need_new_axes = True
                            status = -1
                            break

        if need_new_axes:
            axes[2] = O1_coord + O2_coord - coords[p] - coords[q]
            axes[2] = axes[2] - numpy.inner(axes[2], axes[0]) * axes[0]
            axes[2] = axes[2] / numpy.linalg.norm(axes[2])
            axes[1] = numpy.cross(axes[2], axes[0])
            self.axes = axes

        OO_distance = numpy.linalg.norm(O1_coord - O2_coord)
        if 0.5 * OO_distance < bond_lengths[('Cr', 'O')]:
            Cr_coord = 0.5 * (O1_coord + O2_coord) + axes[2] * ((bond_lengths[('Cr', 'O')])**2.0 - (0.5 * OO_distance)**2.0)**0.5
        else:
            Cr_coord = 0.5 * (O1_coord + O2_coord)

        chromium_atoms = []
        chromium_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if i in peripheral_hydrogens:
                if i == peripheral_hydrogens[0]:
                    chromium_atoms.append('Cr')
                    chromium_coords.append(Cr_coord)
            elif i in peripheral_oxygens:
                if i == n:
                    chromium_atoms.append('O')
                    chromium_coords.append(O1_coord)
                elif i == m:
                    chromium_atoms.append('O')
                    chromium_coords.append(O2_coord)
            else:
                chromium_atoms.append(X)
                chromium_coords.append(coord)

        chromium_cluster = Atoms(chromium_atoms, chromium_coords)

        return chromium_cluster

    def attach_alkyl(self, cluster, alkyl_length, bond_cutoffs=None, bond_lengths=None, axes=None, point_y=True, rotate_2=False):

        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if bond_lengths is None:
            bond_lengths = self.bond_lengths
        if axes is None:
            axes = self.axes

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        Cr_index = -1
        Cr_coord = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'Cr':
                Cr_index = i
                Cr_coord = coord

        if point_y:
            tilts = [
                    axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) + axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) + axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0)
                    ]
        else:
            tilts = [
                    axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) - axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) - axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0)
                    ]

        C_coords = [Cr_coord + tilts[0] * bond_lengths[('Cr', 'C')]]
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
                C_coords[i] = C_coords[1] + rotate_vector(C_coords[i]-C_coords[1], -tilts[1], 180.0)
            for i in range(2, 2*alkyl_length+1):
                H_coords[i] = C_coords[1] + rotate_vector(H_coords[i]-C_coords[1], -tilts[1], 180.0)

        n = len(atoms)+1
        alkyl_atoms = []
        alkyl_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'C':
                n = i
                break
            else:
                alkyl_atoms.append(X)
                alkyl_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
            if X == 'C':
                alkyl_atoms.append('C')
                alkyl_coords.append(coord)
        for coord in C_coords:
            alkyl_atoms.append('C')
            alkyl_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
            if X == 'H':
                alkyl_atoms.append('H')
                alkyl_coords.append(coord)
        for coord in H_coords:
            alkyl_atoms.append('H')
            alkyl_coords.append(coord)

        alkyl_cluster = Atoms(alkyl_atoms, alkyl_coords)

        return alkyl_cluster

    def attach_ethylene(self, cluster, bond_cutoffs=None, bond_lengths=None, ethylene_bond_lengths=None, axes=None, point_y=False):

        if bond_cutoffs is None:
            bond_cutoffs = self.bond_cutoffs
        if bond_lengths is None:
            bond_lengths = self.bond_lengths
        if ethylene_bond_lengths is None:
            ethylene_bond_lengths = self.ethylene_bond_lengths
        if axes is None:
            axes = self.axes

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        Cr_index = -1
        Cr_coord = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'Cr':
                Cr_index = i
                Cr_coord = coord

        #angle = numpy.arctan(0.5 * ethylene_bond_lengths[('C', 'C')] / ethylene_bond_lengths[('Cr', 'C')])
        if point_y:
            tilt0 = axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) + axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0)
            tilt1 = -axes[0]
            #tilt1 = tilt0 * numpy.cos(angle) + axes[0] * numpy.sin(angle)
            #tilt2 = tilt0 * numpy.sin(angle) - axes[0] * numpy.cos(angle)
        else:
            tilt0 = axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) - axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0)
            tilt1 = +axes[0]
            #tilt1 = tilt0 * numpy.cos(angle) - axes[0] * numpy.sin(angle)
            #tilt2 = tilt0 * numpy.sin(angle) + axes[0] * numpy.cos(angle)
        C1_coord = Cr_coord + tilt0 * ethylene_bond_lengths[('Cr', 'C')]
        C2_coord = C1_coord + tilt1 * ethylene_bond_lengths[('C', 'C')]
        C_coords = [C1_coord, C2_coord]
        H1_coord = C1_coord + rotate_vector(tilt1, tilt0, -120.0) * bond_lengths[('C', 'H')]
        H2_coord = C1_coord + rotate_vector(tilt1, tilt0, +120.0) * bond_lengths[('C', 'H')]
        H3_coord = C2_coord + rotate_vector(tilt1, tilt0, -60.0) * bond_lengths[('C', 'H')]
        H4_coord = C2_coord + rotate_vector(tilt1, tilt0, +60.0) * bond_lengths[('C', 'H')]
        H_coords = [H1_coord, H2_coord, H3_coord, H4_coord]

        n = len(atoms)+1
        ethylene_atoms = []
        ethylene_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'C':
                n = i
                break
            else:
                ethylene_atoms.append(X)
                ethylene_coords.append(coord)
        for coord in C_coords:
            ethylene_atoms.append('C')
            ethylene_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
            if X == 'C':
                ethylene_atoms.append('C')
                ethylene_coords.append(coord)
        for coord in H_coords:
            ethylene_atoms.append('H')
            ethylene_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
            if X == 'H':
                ethylene_atoms.append('H')
                ethylene_coords.append(coord)

        ethylene_cluster = Atoms(ethylene_atoms, ethylene_coords)

        return ethylene_cluster

    def attach_transition_state(self, cluster, alkyl_length, bond_cutoffs=None, bond_lengths=None, ethylene_bond_lengths=None, transition_state_lengths=None, axes=None, point_y=False):

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

        atoms = cluster.get_chemical_symbols()
        coords = cluster.get_positions()
        bonds = neighbor_list('ij', cluster, bond_cutoffs)

        Cr_index = -1
        Cr_coord = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'Cr':
                Cr_index = i
                Cr_coord = coord

        if point_y:
            tilts = [
                    axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) + axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) + axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0)
                    ]
        else:
            tilts = [
                    axes[2] * numpy.cos(numpy.pi*0.5*109.5/180.0) - axes[1] * numpy.sin(numpy.pi*0.5*109.5/180.0),
                    axes[2] * numpy.cos(numpy.pi*(1.5*109.5-180.0)/180.0) - axes[1] * numpy.sin(numpy.pi*(1.5*109.5-180.0)/180.0)
                    ]

        C1_coord = Cr_coord + tilts[0] * transition_state_lengths[('Cr', 'C')]
        C2_coord = C1_coord + tilts[1] * ethylene_bond_lengths[('C', 'C')]
        C3_coord = C2_coord + tilts[0] * transition_state_lengths[('C', 'C')]
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
            C_coords[i] = C_coords[1] + rotate_vector(C_coords[i]-C_coords[1], -tilts[1], 180.0)
        for i in range(2, 2*alkyl_length+1):
            H_coords[i] = C_coords[1] + rotate_vector(H_coords[i]-C_coords[1], -tilts[1], 180.0)

        if point_y:
            axis = +axes[0]
        else:
            axis = -axes[0]

        x = numpy.array([1.0, 0.0])
        y = numpy.array([0.0, 1.0])
        ab = 1.34 * x
        bc = 2.2 * (x * numpy.cos((180.0-109.5)/180.0*numpy.pi) + y * numpy.sin((180.0-109.5)/180.0*numpy.pi))
        ac = ab + bc
        ph3 = numpy.arccos(numpy.dot(ab, ac)/(numpy.linalg.norm(ab)*numpy.linalg.norm(ac)))
        l1 = 2.1
        l2 = numpy.linalg.norm(ac)
        th1 = numpy.arccos(0.5*l2/l1)
        th2 = 2.0 * numpy.arcsin(0.5*l2/l1)

        angle = 0.5*(109.5-th2*180.0/numpy.pi)
        for i in range(0, alkyl_length):
            C_coords[i] = Cr_coord + rotate_vector(C_coords[i]-Cr_coord, axis, angle)
        for i in range(0, 2*alkyl_length+1):
            H_coords[i] = Cr_coord + rotate_vector(H_coords[i]-Cr_coord, axis, angle)

        angle = 109.5-(th1+ph3)*180.0/numpy.pi
        for i in range(1, alkyl_length):
            C_coords[i] = C_coords[0] + rotate_vector(C_coords[i]-C_coords[0], axis, angle)
        for i in range(1, 2*alkyl_length+1):
            H_coords[i] = C_coords[0] + rotate_vector(H_coords[i]-C_coords[0], axis, angle)

        n = len(atoms)+1
        transition_state_atoms = []
        transition_state_coords = []
        for i, (X, coord) in enumerate(zip(atoms, coords)):
            if X == 'C':
                n = i
                break
            else:
                transition_state_atoms.append(X)
                transition_state_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
            if X == 'C':
                transition_state_atoms.append('C')
                transition_state_coords.append(coord)
        for coord in C_coords:
            transition_state_atoms.append('C')
            transition_state_coords.append(coord)
        for X, coord in zip(atoms[n:], coords[n:]):
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
            labels = ['L-butyl', 'L-butyl-R-ethylene', 'LR-transition-state', 'R-hexyl', 'R-butyl', 'R-butyl-L-ethylene', 'RL-transition-state', 'L-hexyl']
        if clusters is None:
            clusters = [self.L_butyl_cluster, self.L_butyl_R_ethylene_cluster, self.LR_transition_state_cluster, self.R_hexyl_cluster, self.R_butyl_cluster, self.R_butyl_L_ethylene_cluster, self.RL_transition_state_cluster, self.L_hexyl_cluster]
        for label, cluster in zip(labels, clusters):
            write(file_path.format(label), cluster, file_type)
        return


if __name__ == '__main__':


    clusters = Phillips('tests/A_0001.xyz', 'xyz', [2, 3])
    clusters.save_clusters('A_0001_{:s}.xyz', 'xyz')


