import numpy
import os

from ase import Atoms
from ase.io import read, write
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize


class MatchCoords():

    def __init__(self, match_coords, target_coords, params=None):
        self.match_coords = match_coords
        self.target_coords = target_coords
        self.params = params
        if params is None:
            self.params = numpy.zeros(6)
        return

    def transform(self, coords, params=None):

        if params is None:
            params = self.params

        trans = params[0:3]
        rotat = params[3:6]

        rot = Rotation.from_euler('zyx', rotat)

        return rot.apply(coords + trans)

    def objective(self, params, match_coords, target_coords):

        diff_coords = self.transform(match_coords, params=params) - target_coords
        diff = numpy.linalg.norm(diff_coords)

        return diff

    def fit(self, params=None, match_coords=None, target_coords=None):

        if params is None:
            params = self.params
        if match_coords is None:
            match_coords = self.match_coords
        if target_coords is None:
            target_coords = self.target_coords

        res = minimize(self.objective, params, (match_coords, target_coords), method='Nelder-Mead')

        self.params = res.x

        return res.x


class Graft():

    def __init__(self, input_cluster, ref_cluster,
            podal_atoms=None, input_podal_atoms=None, ref_podal_atoms=None,
            match_atoms=None, input_match_atoms=None, ref_match_atoms=None,
            params=None
            ):

        if input_podal_atoms is None:
            input_podal_atoms = podal_atoms
        if ref_podal_atoms is None:
            ref_podal_atoms = podal_atoms

        if match_atoms is None:
            match_atoms = podal_atoms
        if input_match_atoms is None:
            input_match_atoms = match_atoms
            if match_atoms is None:
                input_match_atoms = input_podal_atoms
        if ref_match_atoms is None:
            ref_match_atoms = match_atoms
            if match_atoms is None:
                ref_match_atoms = ref_podal_atoms

        self.podal_atoms = podal_atoms
        self.input_podal_atoms = input_podal_atoms
        self.ref_podal_atoms = ref_podal_atoms
        self.match_atoms = match_atoms
        self.input_match_atoms = input_match_atoms
        self.ref_match_atoms = ref_match_atoms

        self.params=params

        self.input_cluster = input_cluster
        self.ref_cluster = ref_cluster

        return

    def run(self):

        input_atoms = self.input_cluster.get_chemical_symbols()
        input_coords = self.input_cluster.get_positions()

        ref_atoms = self.ref_cluster.get_chemical_symbols()
        ref_coords = self.ref_cluster.get_positions()

        match = MatchCoords(ref_coords[self.ref_match_atoms], input_coords[self.input_match_atoms], params=self.params)
        match.fit()

        if len(self.input_podal_atoms) == len(self.ref_podal_atoms):
            output_atoms = ref_atoms
            output_coords = match.transform(ref_coords)
            output_coords[self.ref_podal_atoms] = input_coords[self.input_podal_atoms]
        else:
            output_atoms = [X for i, X in enumerate(input_atoms) if i in self.input_podal_atoms] + [X for i, X in enumerate(ref_atoms) if i not in self.ref_podal_atoms]
            output_coords = [coords for i, coords in enumerate(input_coords) if i in self.input_podal_atoms] + [coords for i, coords in enumerate(match.transform(ref_coords)) if i not in self.ref_podal_atoms]

        self.output_cluster = Atoms(output_atoms, output_coords)

        return self.output_cluster

    def save_cluster(self, file_path, file_type='xyz', cluster=None):
        if cluster is None:
            cluster = self.output_cluster
        write(file_path, cluster, file_type)
        return

