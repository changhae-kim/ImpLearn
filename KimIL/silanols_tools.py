import numpy


def reorder_podal_oxygens(podal_O_coords, O1_coord, O2_coord, Si1_coord, Si2_coord, max_iter=50):

    origin = 0.5 * (Si1_coord + Si2_coord)
    centered = podal_O_coords - origin
    xaxis = Si2_coord - Si1_coord
    xaxis = xaxis / numpy.linalg.norm(xaxis)
    zaxis = 0.5 * (O1_coord + O2_coord - Si1_coord - Si2_coord)
    zaxis = zaxis / numpy.linalg.norm(zaxis)
    yaxis = numpy.cross(zaxis, xaxis)

    centered1 = podal_O_coords - Si1_coord
    centered2 = podal_O_coords - Si2_coord
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
        m = numpy.argmin([numpy.linalg.norm(centered[0]-centered[j]) for j in reordered2])
        reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
    reordered = reordered1 + reordered2

    return reordered


def permute_podal_atoms(podal_coords, silanol_type='vicinal', right_hand_only=True):

    permutes = []

    if silanols_type == 'vicinal':
        permutes += [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [2, 3, 0, 1, 6, 7, 4, 5],
                ]
        if not right_hand_only:
            permutes += [
                    [3, 2, 1, 0, 7, 6, 5, 4],
                    [1, 0, 3, 2, 5, 4, 7, 6],
                    ]

    elif silanols_type == 'nonvicinal':
        permutes += [
                ]

    return permutes

