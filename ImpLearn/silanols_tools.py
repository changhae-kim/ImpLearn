import numpy


def reorder_podal_oxygens_v2(Op_coords, Oh_coords, Si_coords, Ob_coords=None, max_iter=50):

    Op_coords = numpy.array(Op_coords)
    Oh_coords = numpy.array(Oh_coords)
    Si_coords = numpy.array(Si_coords)

    if Ob_coords is None:
        Ox_coords = Op_coords
    else:
        Ox_coords = numpy.concatenate([Op_coords, Ob_coords])

    reordered = [[] for _ in Si_coords]
    centered = Ox_coords[numpy.newaxis, :, :] - Si_coords[:, numpy.newaxis, :]
    for i, _ in enumerate(Ox_coords):
        if i < len(Op_coords):
            norms = numpy.linalg.norm(centered[:, i, :], axis=1)
            n = numpy.argmin(norms)
            reordered[n].append(i)
        else:
            norms = numpy.linalg.norm(centered[:, i, :], axis=1)
            n, m = numpy.argsort(norms, kind='stable')[:2]
            reordered[n].append(i)
            reordered[m].append(i)

    indices = []
    zaxes = Oh_coords[numpy.newaxis, :, :] - Si_coords[:, numpy.newaxis, :]
    for i, _ in enumerate(Si_coords):
        norms = numpy.linalg.norm(zaxes[i, :, :], axis=1)
        n = numpy.argmin(norms)
        indices.append(n)
    zaxes = Oh_coords[indices] - Si_coords
    zaxes = zaxes / numpy.linalg.norm(zaxes, axis=1)[:, numpy.newaxis]

    for i, _ in enumerate(Si_coords):
        status = -1
        for n in range(max_iter):
            if status == 0:
                break
            else:
                status = 0
                for j, _ in enumerate(reordered[i][:-1]):
                    if numpy.dot(numpy.cross(zaxes[i], centered[i][reordered[i][j]]), centered[i][reordered[i][j+1]]) < 0.0:
                        reordered[i][j], reordered[i][j+1] = reordered[i][j+1], reordered[i][j]
                        status = -1
                        break

    norms = [ numpy.min([ numpy.linalg.norm(Ox_coords[i]-Ox_coords[j]) for i in reordered[p] for j in reordered[(p+1)%len(Si_coords)] ]) for p, _ in enumerate(Si_coords) ]
    indices = numpy.argsort(norms, kind='stable')
    for r in indices[::-1]:
        p = (r)%len(Si_coords)
        q = (r+1)%len(Si_coords)
        if len(reordered[p]) > 0:
            if len(reordered[q]) > 0:
                nm = numpy.argmin([numpy.linalg.norm(Ox_coords[i]-Ox_coords[j]) for i in reordered[p] for j in reordered[q]])
                n, m = numpy.unravel_index(nm, [len(reordered[p]), len(reordered[q])])
                reordered[p] = [reordered[p][(n+i+1)%len(reordered[p])] for i, _ in enumerate(reordered[p])]
                reordered[q] = [reordered[q][(m+i)%len(reordered[q])] for i, _ in enumerate(reordered[q])]
            else:
                n = numpy.argmin([numpy.linalg.norm(Ox_coords[i]-Si_coords[q]) for i in reordered[p]])
                reordered[p] = [reordered[p][(n+i+1)%len(reordered[p])] for i, _ in enumerate(reordered[p])]
        else:
            if len(reordered[q]) > 0:
                m = numpy.argmin([numpy.linalg.norm(Ox_coords[i]-Si_coords[p]) for i in reordered[q]])
                reordered[q] = [reordered[q][(m+i)%len(reordered[q])] for i, _ in enumerate(reordered[q])]
            else:
                continue

    recombined = [i for indices in reordered for i in indices if i < len(Op_coords)]

    return recombined

def reorder_podal_oxygens(podal_coords, O1_coord, O2_coord, Si1_coord, Si2_coord, O3_coord=None, max_iter=50):

    centered1 = podal_coords - Si1_coord
    centered2 = podal_coords - Si2_coord
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
            for j, _ in enumerate(reordered1[:-1]):
                if numpy.dot(numpy.cross(zaxis1, centered1[reordered1[j]]), centered1[reordered1[j+1]]) < 0.0:
                    reordered1[j], reordered1[j+1] = reordered1[j+1], reordered1[j]
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
            for j, _ in enumerate(reordered2[:-1]):
                if numpy.dot(numpy.cross(zaxis2, centered2[reordered2[j]]), centered2[reordered2[j+1]]) < 0.0:
                    reordered2[j], reordered2[j+1] = reordered2[j+1], reordered2[j]
                    status = -1
                    break

    if len(reordered1) > 2:
        if len(reordered2) > 2:
            nm = numpy.argmin([numpy.linalg.norm(podal_coords[i] - podal_coords[j]) for i in reordered1 for j in reordered2])
            n, m = numpy.unravel_index(nm, [len(reordered1), len(reordered2)])
            reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
            reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
        elif len(reordered2) > 0:
            if O3_coord is None:
                nm = numpy.argmin([numpy.linalg.norm(podal_coords[i] - podal_coords[j]) for i in reordered1 for j in reordered2])
                n, m = numpy.unravel_index(nm, [len(reordered1), len(reordered2)])
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
                reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
            else:
                n = numpy.argmin([numpy.linalg.norm(podal_coords[i] - podal_coords[reordered2[0]]) for i in reordered1])
                if numpy.dot(numpy.cross(zaxis1, (O3_coord - Si1_coord)), centered1[reordered1[n]]) > 0.0:
                    n = (n-1)%len(reordered1)
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
        else:
            n = numpy.argmin([numpy.linalg.norm(podal_coords[i] - Si2_coord) for i in reordered1])
            reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]

    elif len(reordered2) > 2:
        if len(reordered1) > 0:
            if O3_coord is None:
                nm = numpy.argmin([numpy.linalg.norm(podal_coords[i] - podal_coords[j]) for i in reordered1 for j in reordered2])
                n, m = numpy.unravel_index(nm, [len(reordered1), len(reordered2)])
                reordered1 = [reordered1[(n+1+i)%len(reordered1)] for i, _ in enumerate(reordered1)]
                reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
            else:
                m = numpy.argmin([numpy.linalg.norm(podal_coords[reordered1[-1]] - podal_coords[j]) for j in reordered2])
                if numpy.dot(numpy.cross(zaxis2, (O3_coord - Si2_coord)), centered2[reordered2[m]]) < 0.0:
                    m = (m+1)%len(reordered2)
                reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]
        else:
            m = numpy.argmin([numpy.linalg.norm(Si1_coord - podal_coords[j]) for j in reordered2])
            reordered2 = [reordered2[(m+i)%len(reordered2)] for i, _ in enumerate(reordered2)]

    reordered = reordered1 + reordered2

    return reordered

def permute_podal_atoms(pair_type, podal_coords=None, F_capping=True):

    permutes = []

    if pair_type == 'vicinal':
        permute0 = list(range(4))
        permute1 = [2, 3, 0, 1]
        if not F_capping:
            permute0 += [i + 4 for i in permute0]
            permute1 += [i + 4 for i in permute1]
        permutes += [permute0, permute1]

    elif pair_type == 'nonvicinal':
        permute0 = list(range(6))
        nm = numpy.argsort([ numpy.linalg.norm(podal_coords[i] - podal_coords[j]) for i in range(0, 3) for j in range(3, 6) ], kind='stable')
        n, m = numpy.unravel_index(nm[0], [3, 3])
        permute1a = [(j+1)%3+3 for j in range(m, m+3)]
        permute1b = [i%3 for i in range(n, n+3)]
        permute1 = permute1a + permute1b
        n, m = numpy.unravel_index(nm[1], [3, 3])
        permute2a = [(i+1)%3 for i in range(n, n+3)]
        permute2b = [j%3+3 for j in range(m, m+3)]
        permute2 = permute2a + permute2b
        n, m = numpy.unravel_index(nm[1], [3, 3])
        permute3a = [(j+1)%3+3 for j in range(m, m+3)]
        permute3b = [i%3 for i in range(n, n+3)]
        permute3 = permute3a + permute3b
        if not F_capping:
            permute0 += [i + 6 for i in permute0]
            permute1 += [i + 6 for i in permute1]
            permute2 += [i + 6 for i in permute2]
            permute3 += [i + 6 for i in permute3]
        permutes += [permute0, permute1, permute2, permute3]

    return permutes

