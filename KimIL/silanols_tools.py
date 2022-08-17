import numpy


def reorder_podal_oxygens(O_coords, O1_coord, O2_coord, Si1_coord, Si2_coord, O3_coord=None, max_iter=50):

    origin = 0.5 * (Si1_coord + Si2_coord)
    xaxis = Si2_coord - Si1_coord
    xaxis = xaxis / numpy.linalg.norm(xaxis)
    zaxis = 0.5 * (O1_coord + O2_coord - Si1_coord - Si2_coord)
    zaxis = zaxis / numpy.linalg.norm(zaxis)
    yaxis = numpy.cross(zaxis, xaxis)

    centered1 = O_coords - Si1_coord
    centered2 = O_coords - Si2_coord
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
            nm = numpy.argmin([numpy.linalg.norm(O_coords[i] - O_coords[j]) for i in reordered1 for j in reordered2])
            n, m = numpy.unravel_index(nm, [len(reordered1), len(reordered2)])
            reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
            reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
        elif len(reordered2) > 0:
            if O3_coord is None:
                nm = numpy.argmin([numpy.linalg.norm(O_coords[i] - O_coords[j]) for i in reordered1 for j in reordered2])
                n, m = numpy.unravel_index(nm, [len(reordered1), len(reordered2)])
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
                reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
            else:
                n = numpy.argmin([numpy.linalg.norm(O_coords[i] - O_coords[reordered2[0]]) for i in reordered1])
                if numpy.dot(numpy.cross(zaxis1, (O3_coord - Si1_coord)), centered1[reordered1[n]]) > 0.0:
                    n = (n-1)%len(reordered1)
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
        else:
            n = numpy.argmin([numpy.linalg.norm(O_coords[i] - Si2_coord) for i in reordered1])
            reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
    elif len(reordered2) > 2:
        if len(reordered1) > 0:
            if O3_coord is None:
                nm = numpy.argmin([numpy.linalg.norm(O_coords[i] - O_coords[j]) for i in reordered1 for j in reordered2])
                n, m = numpy.unravel_index(nm, [len(reordered1), len(reordered2)])
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
                reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
            else:
                m = numpy.argmin([numpy.linalg.norm(O_coords[reordered1[-1]] - O_coords[j]) for j in reordered2])
                if numpy.dot(numpy.cross(zaxis2, (O3_coord - Si2_coord)), centered2[reordered2[m]]) < 0.0:
                    m = (m+1)%len(reordered2)
                reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
        else:
            m = numpy.argmin([numpy.linalg.norm(Si1_coord - O_coords[j]) for j in reordered2])
            reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
    reordered = reordered1 + reordered2

    return reordered

def permute_podal_atoms(pair_type, podal_coords=None, right_hand_only=True):

    permutes = []

    if pair_type == 'vicinal':
        permute0 = list(range(8))
        permute1 = [2, 3, 0, 1]
        permute1 = permute1 + [i + 4 for i in permute1]
        permutes += [permute0, permute1]
        if not right_hand_only:
            permutes += [list(reversed(permute[:4])) + list(reversed(permute[4:])) for permute in permutes]

    elif pair_type == 'nonvicinal':
        permute0 = list(range(12))
        nm = numpy.argsort([numpy.linalg.norm(podal_coords[i] - podal_coords[j]) for i in range(0, 3) for j in range(3, 6)])
        n, m = numpy.unravel_index(nm[0], [3, 3])
        permute1a = [(j+1)%3+3 for j in range(m, m+3)]
        permute1b = [i%3 for i in range(n, n+3)]
        permute1 = permute1a + permute1b + [i + 6 for i in permute1a + permute1b]
        n, m = numpy.unravel_index(nm[1], [3, 3])
        permute2a = [(i+1)%3 for i in range(n, n+3)]
        permute2b = [j%3+3 for j in range(m, m+3)]
        permute2 = permute2a + permute2b + [i + 6 for i in permute2a + permute2b]
        n, m = numpy.unravel_index(nm[1], [3, 3])
        permute3a = [(j+1)%3+3 for j in range(m, m+3)]
        permute3b = [i%3 for i in range(n, n+3)]
        permute3 = permute3a + permute3b + [i + 6 for i in permute3a + permute3b]
        permutes += [permute0, permute1, permute2, permute3]
        if not right_hand_only:
            permutes += [list(reversed(permute[:6])) + list(reversed(permute[6:])) for permute in permutes]

    return permutes

