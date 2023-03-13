import numpy


class Sampler():

    def __init__(self, n_points, initial_batch=None, initial_batch_size=20, batch_size=10, replace=False, exclude=[], random_state=None):

        self.n_points = n_points
        self.replace = replace
        self.exclude = exclude
        self.rng = numpy.random.RandomState(random_state)

        self.samples = []
        if initial_batch is None:
            self.initial_batch_size = initial_batch_size
            self.sample(None, batch_size=initial_batch_size)
        else:
            self.initial_batch_size = len(initial_batch)
            self.samples = initial_batch

        self.batch_size = batch_size

        return

    def sample(self, weights=None, batch=None, batch_size=None, replace=None, exclude=None, random_state=None):

        if batch is None:

            if batch_size is None:
                batch_size = self.batch_size
            if replace is None:
                replace = self.replace
            if exclude is None:
                exclude = self.exclude
            if random_state is not None:
                self.rng.seed(random_state)

            if replace:
                candidates = [i for i in range(self.n_points) if i not in exclude]
                samples = list(self.rng.choice(candidates, size=batch_size, replace=True, p=weights))
            else:
                candidates = [i for i in range(self.n_points) if i not in self.samples and i not in exclude]
                if weights is None:
                    reweights = None
                else:
                    reweights = numpy.array(weights)[candidates]
                    reweights = reweights / numpy.sum(reweights)
                samples = list(self.rng.choice(candidates, size=batch_size, replace=False, p=reweights))

            self.samples.extend(samples)

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

