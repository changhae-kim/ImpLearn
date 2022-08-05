import numpy


class Sampler():

    def __init__(self, n_points, initial_batch=[], initial_batch_size=50, batch_size=5, replace=False, random_state=None):

        self.n_points = n_points
        self.replace = replace
        self.rng = numpy.random.RandomState(random_state)

        if initial_batch == []:
            self.initial_batch_size = initial_batch_size
            candidates = [i for i in range(self.n_points)]
            self.samples = list(self.rng.choice(candidates, size=initial_batch_size, replace=replace))
        else:
            self.initial_batch_size = len(initial_batch)
            self.samples = initial_batch

        self.batch_size = batch_size

        return

    def sample(self, weights, batch=[], batch_size=None, replace=None, random_state=None):

        if batch == []:

            if batch_size is None:
                batch_size = self.batch_size
            if replace is None:
                replace = self.replace
            if random_state is not None:
                self.rng.seed(random_state)

            if replace:
                candidates = [i for i in range(self.n_points)]
                samples = list(self.rng.choice(candidates, size=batch_size, replace=True, p=weights))
            else:
                candidates = [i for i in range(self.n_points) if i not in self.samples]
                reweights = numpy.array(weights)[candidates]
                reweights /= numpy.sum(reweights)
                samples = list(self.rng.choice(candidates, size=batch_size, replace=False, p=reweights))

        else:

            samples = batch

        self.samples.extend(samples)

        return samples


if __name__ == '__main__':

    n_points = 500

    sampler = Sampler(n_points)

    print(sampler.samples)

    weights = numpy.exp(-5.0 * numpy.arange(n_points))
    weights /= numpy.sum(weights)
    samples = sampler.sample(weights)

    print(samples)
    print(sampler.samples)

