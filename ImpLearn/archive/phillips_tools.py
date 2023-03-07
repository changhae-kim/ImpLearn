import numpy


def rotate_vector(vector, axis, angle, degrees=True):
    unit = numpy.array(axis) / numpy.linalg.norm(axis)
    parallel = numpy.dot(vector, unit) * unit
    perpend1 = vector - parallel
    perpend2 = numpy.cross(unit, perpend1)
    if degrees:
        rotated = parallel + perpend1 * numpy.cos(numpy.pi*angle/180.0) + perpend2 * numpy.sin(numpy.pi*angle/180.0)
    else:
        rotated = parallel + perpend1 * numpy.cos(angle) + perpend2 * numpy.sin(angle)
    return rotated

def step_attractive(coord1, coord2, pivot1, pivot2, radius, max_angle=5.0):
    vec12 = coord2 - coord1
    dist12 = numpy.linalg.norm(vec12)
    if dist12 > radius:
        vec1 = coord1 - pivot1
        vec2 = coord2 - pivot2
        dist1 = numpy.linalg.norm(vec1)
        dist2 = numpy.linalg.norm(vec2)
        axis1 = numpy.cross(vec1, +vec12) / (dist1 * dist12)
        axis2 = numpy.cross(vec2, -vec12) / (dist2 * dist12)
        dot1 = numpy.dot(vec1, +vec12) / (dist1 * dist12)
        dot2 = numpy.dot(vec2, -vec12) / (dist2 * dist12)
        if dot1 > +1.0:
            dot1 = +1.0
        elif dot1 < -1.0:
            dot1 = -1.0
        if dot2 > +1.0:
            dot2 = +1.0
        elif dot2 < -1.0:
            dot2 = -1.0
        weight1 = numpy.arccos(dot1)
        weight2 = numpy.arccos(dot2)
        angle1 = 180.0 * numpy.arctan((dist12 - radius) / dist1) / numpy.pi
        angle2 = 180.0 * numpy.arctan((dist12 - radius) / dist2) / numpy.pi
        if (weight1 * angle1 + weight2 * angle2) / (weight1 + weight2) > max_angle:
            angle_1 = max_angle
            angle_2 = max_angle
        new_coord1 = pivot1 + rotate_vector(vec1, axis1, angle1 * weight1 / (weight1 + weight2))
        new_coord2 = pivot2 + rotate_vector(vec2, axis2, angle2 * weight2 / (weight1 + weight2))
        return new_coord1, new_coord2
    else:
        return coord1, coord2

def step_repulsive(coords, pivot, other_coords, radii, axis=None, max_angle=5.0):
    vecs = coords[numpy.newaxis, :, :] - other_coords[:, numpy.newaxis, :]
    dists = numpy.linalg.norm(vecs, axis=-1)
    n, m = numpy.unravel_index(numpy.argmin(dists - radii), numpy.shape(dists))
    if dists[n, m] < radii[n, m]:
        vec = coords[m] - pivot
        dist = numpy.linalg.norm(vec)
        cross = numpy.cross(vec, vecs[n, m]) / (dist * dists[n, m])
        if axis is None:
            axis = cross
        angle = 180.0 * numpy.arctan((radii[n, m] - dists[n, m]) * numpy.dot(axis, cross) / dist) / numpy.pi
        if angle > max_angle:
            angle = max_angle
        new_coords = [pivot + rotate_vector(coord - pivot, axis, angle) for coord in coords]
        return new_coords
    else:
        return coords

