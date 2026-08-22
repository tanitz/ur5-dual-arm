"""
Where the four corners of an open box come from, and what judges them.

Each sensor does the one thing it is good at. The colour image finds the four
corners of the opening: it has no depth discontinuity to fall into, so it
answers for the far rim as readily as the near one, to the pixel. Depth is
never asked where the corners are — it is asked, afterwards, how far away the
near rim actually is, so that an answer built entirely out of RGB can be
checked against a measurement that shares none of its assumptions.

That check is the whole point. Four corners plus an assumed opening size
always produce *a* pose; only the agreement between what that pose predicts
for the near rim and what the sensor measured there says whether the size was
the right one. On this cell a correct size agrees to a few millimetres and a
wrong one is out by hundreds, so the same number both accepts an answer and
picks the box size when it is not known in advance.

Plane segmentation is deliberately absent. It was tried, and the assumption it
rests on — that the largest plane in view is the surface the box stands on —
is false in a real workshop: RANSAC finds the floor, and once the view is
cropped to the box it finds the far inner wall, because a steep camera sees
more of that wall than of the box's floor.

Corner order everywhere in this package is front-left, front-right,
back-right, back-left, where "front" is the bottom of the picture. A cell
camera looks down into the box from above and behind, so the bottom edge of
the opening in the image is the rim nearest the lens. `measure_near_edge`
depends on that, and so does the object frame `solve_opening_pnp` builds.
"""

import numpy as np

import cv2


# Euro/KLT stacking crates, in millimetres. Only used when nobody has said
# what size the box is: the depth cross-check tells these apart easily,
# because the wrong one of a pair is out by more than a hundred millimetres.
STANDARD_SIZES = [
    (600, 400), (400, 300), (600, 800), (300, 200),
    (594, 396), (396, 297), (800, 600), (500, 300),
    # Small cartons commonly placed on top of a crate.  These mirror the
    # candidates used by box.py; depth still has to confirm the chosen size.
    (195, 100), (200, 150), (250, 150), (150, 100),
]


# ── the colour image: four corners ────────────────────────────────────────
def find_rim_quad(color, roi=None, min_area=5000, min_cover=0.45):
    """The opening's four corners in the picture, or None.

    Edges, not colour. A grey crate on a grey floor is one reading of the same
    number twice — measured on this cell, BGR 126,118,111 against 116,117,114 —
    while the rim itself is a hard bright line that Canny answers for every
    time.

    The bilateral filter earns its cost: an ordinary blur moves corners by
    several pixels, and every millimetre of the final pose is a pixel of
    corner error multiplied by range.

    `roi` masks the edge image rather than cropping the picture, so corners
    come back in full-frame coordinates and the caller has nothing to add
    back on.
    """
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 30, 90)
    if roi is not None:
        x1, y1, x2, y2 = (int(v) for v in roi)
        mask = np.zeros_like(edges)
        mask[y1:y2 + 1, x1:x2 + 1] = 255
        edges = cv2.bitwise_and(edges, mask)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for area, hull, contour in sorted(
            ((cv2.contourArea(h), h, c)
             for c, h in ((c, cv2.convexHull(c)) for c in contours)),
            key=lambda item: -item[0])[:15]:
        if area < min_area:
            break                       # sorted, so nothing after this is big
        quad = _quad_from_hull(hull)
        if quad is None:
            continue
        if _outline_coverage(contour, quad) < min_cover:
            continue
        quads.append(quad)

    if not quads:
        return None, edges

    # Canny reports the rim's outer and inner edge separately. Their average
    # is the middle of the line, which is what the rim actually is; either one
    # alone is half a line-width out, and at working range that is millimetres
    # of pose. The corners must be ordered before they are averaged, or the
    # mean is taken between corners that do not correspond.
    if len(quads) >= 2 and _quads_similar(quads[0], quads[1]):
        quad = np.mean(quads[:2], axis=0)
    else:
        quad = quads[0]
    return quad, edges


def find_bright_quads(color, roi=None, min_area=1500):
    """Small pale rectangular faces, ordered like an opening's corners.

    A small carton often has too few edge pixels to compete with the crate
    underneath it in :func:`find_rim_quad`.  Thresholding several upper
    brightness percentiles exposes its top face directly.  This is the
    ``bright`` strategy from ``box.py``; callers must still use depth to
    reject highlights and other pale rectangles.
    """
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    bounds = (0, 0, width - 1, height - 1) if roi is None else roi
    x1, y1, x2, y2 = (int(v) for v in bounds)
    candidates = []

    for percentile in (88, 92, 96, 98, 99):
        threshold = float(np.percentile(gray[y1:y2 + 1, x1:x2 + 1],
                                        percentile))
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        outside = np.zeros_like(mask)
        outside[y1:y2 + 1, x1:x2 + 1] = 255
        mask = cv2.bitwise_and(mask, outside)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if (area < min_area or x <= x1 + 1 or y <= y1 + 1 or
                    x + w >= x2 - 1 or y + h >= y2 - 1):
                continue
            component = (labels == index).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            quad = _quad_from_hull(cv2.convexHull(contours[0]),
                                   sweep=np.arange(0.01, 0.10, 0.004))
            if quad is None:
                continue
            if min(np.linalg.norm(quad[(i + 1) % 4] - quad[i])
                   for i in range(4)) < 20:
                continue
            duplicate = next((i for i, old in enumerate(candidates)
                              if np.max(np.linalg.norm(quad - old,
                                                       axis=1)) < 20), None)
            if duplicate is None:
                candidates.append(quad)
            else:
                # Several percentiles usually trace the same pale face a
                # pixel or two apart.  Their mean is noticeably steadier on
                # a face only 20 px tall (the small box in the reference
                # capture), and matches box.py's candidate merging.
                candidates[duplicate] = (candidates[duplicate] + quad) / 2.0
    return candidates


def _quad_from_hull(hull, sweep=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06)):
    """A hull reduced to four corners, or None if it will not reduce.

    The hull, rather than the contour, is what makes this survive a box with
    work standing proud of the rim. Anything crossing the rim breaks the edge
    Canny drew, and a broken loop is an *open* curve: measured on this cell, a
    rim traced over 2769 pixels came back with a contour area of 127, so every
    test that a closed quadrilateral would pass — area, four vertices,
    convexity — rejected it. The break lies along a straight side, and the
    hull's own edge is that same straight line, so taking the hull repairs it
    exactly rather than approximately.

    Closing the gap with a morphological close was tried instead and is worse:
    it welds the intruding object's outline to the rim, and then the hull
    swells to contain the object as well.

    Epsilon is swept from fine to coarse because one value does not fit both
    cases: a hull with several breaks in it carries extra corners and needs a
    coarser tolerance, while a clean rim already has four at the finest, and
    starting coarse would round off a real corner.
    """
    peri = cv2.arcLength(hull, True)
    for eps in sweep:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_corners(approx.reshape(4, 2).astype(np.float64))
    return None


def _outline_coverage(contour, quad, band=4.0):
    """How much of the contour actually lies along the quad's outline.

    Accepting open curves costs the shape test that "a closed four-sided
    contour" used to perform for free. A sprawl of edge in a cluttered scene
    can have a large hull, but its points do not sit on the sides of the
    rectangle that hull reduces to. Measured on this cell, an opening scores
    0.75 to 1.00 and clutter scores 0.36.
    """
    points = contour.reshape(-1, 2).astype(np.float64)
    nearest = np.full(len(points), np.inf)
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        along = b - a
        length = np.linalg.norm(along)
        if length < 1e-6:
            continue
        along = along / length
        relative = points - a
        # clipped to the side itself: a point beyond a corner is not on it
        t = np.clip(relative @ along, 0.0, length)
        nearest = np.minimum(nearest, np.linalg.norm(
            relative - t[:, None] * along, axis=1))
    return float(np.mean(nearest < band))


def order_corners(points):
    """Front-left, front-right, back-right, back-left.

    Sorted by angle about the centroid so the four are in ring order whatever
    order they arrived in, then rolled so the top-left of the picture leads,
    then reversed — which turns the clockwise-from-top-left ring into this
    package's front-first convention.
    """
    points = np.asarray(points, dtype=np.float64)
    centre = points.mean(0)
    ring = points[np.argsort(np.arctan2(points[:, 1] - centre[1],
                                        points[:, 0] - centre[0]))]
    ring = np.roll(ring, -int(np.argmin(ring[:, 0] + ring[:, 1])), axis=0)
    return ring[::-1].copy()


def _quads_similar(a, b, tol=25.0):
    """Are these two the outer and inner edge of one drawn line?"""
    return float(np.max(np.linalg.norm(a - b, axis=1))) < tol


# ── depth: how far the near rim really is ─────────────────────────────────
def measure_near_edge(depth, intrinsics, a, b, samples=90, scan=14,
                      z_min=0.3, z_max=5.0):
    """The near rim as 3D points, measured without reference to the corners.

    Only the near rim. The far one carries no usable depth at all: it is a
    depth discontinuity, and the sensor reads straight past the few
    millimetres of plastic to the background behind it — on this cell, 1.9 m
    for a rim that was at 1.05 m. The near rim has the box's own wall standing
    behind it, so what comes back is the rim.

    At each sample the scan runs perpendicular to the rim and keeps the
    *nearest* depth in that line, because the rim is the top of a wall and so
    the closest surface anywhere across it. That is what makes this tolerant
    of corners that are a few pixels out, which is the only way it can serve
    as an independent check on the corners themselves.

    The whole sample grid is read at once. The obvious loop is 2610 iterations
    of scalar indexing per frame, which measured 40 ms on this Jetson and put
    a 30 fps camera out of reach on its own; this is the same arithmetic in
    1.3 ms, down to the rounding — np.rint breaks halves to even exactly as
    round() does, and argmin keeps the first minimum exactly as `<` did.
    """
    depth = np.asarray(depth, dtype=float)
    height, width = depth.shape
    a = np.asarray(a, dtype=float)
    along = np.asarray(b, dtype=float) - a
    length = np.linalg.norm(along)
    if length < 1e-6:
        return None
    across = np.array([-along[1], along[0]]) / length

    base = a + np.outer(np.linspace(0.02, 0.98, samples), along)
    grid = base[:, None, :] + across * np.arange(-scan, scan + 1)[:, None]
    u = np.rint(grid[:, :, 0]).astype(np.intp)
    v = np.rint(grid[:, :, 1]).astype(np.intp)

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    z = depth[np.clip(v, 0, height - 1), np.clip(u, 0, width - 1)]
    # anything unusable becomes infinite, so the nearest-surface rule cannot
    # choose it and the sample simply has no answer
    z = np.where(inside & (z > z_min) & (z < z_max), z, np.inf)

    pick = np.argmin(z, axis=1)
    rows = np.arange(len(base))
    z, u, v = z[rows, pick], u[rows, pick], v[rows, pick]
    keep = np.isfinite(z)
    if keep.sum() < 20:
        return None
    z, u, v = z[keep], u[keep], v[keep]

    points = np.stack([(u - intrinsics.cx) * z / intrinsics.fx,
                       (v - intrinsics.cy) * z / intrinsics.fy, z], axis=1)

    # a sample that read through the rim lands metres away from the rest;
    # a median absolute deviation cut removes it without assuming how many
    median = np.median(points[:, 2])
    spread = np.median(np.abs(points[:, 2] - median))
    keep = np.abs(points[:, 2] - median) < 4 * (spread + 1e-6)
    if keep.sum() >= 20:
        points = points[keep]

    centre = points.mean(0)
    direction = np.linalg.svd(points - centre, full_matrices=False)[2][0]
    projected = (points - centre) @ direction
    residual = np.linalg.norm(
        (points - centre) - np.outer(projected, direction), axis=1)
    low, high = np.percentile(projected, 1), np.percentile(projected, 99)
    return {
        "length": float(high - low),
        "centre": centre,
        "direction": direction,
        "straightness": float(residual.std()),   # low means it really is a line
        "count": int(len(points)),
        "points": points,
    }


def predicted_near_edge(transform, width):
    """How far the solved pose puts the middle of the near rim from the lens.

    The one number a pose built from corners and an assumed width can be
    challenged on, because depth can measure the same thing without ever
    being told where the corners are.
    """
    transform = np.asarray(transform, dtype=float)
    middle = transform[:3, :3] @ np.array([0.0, -float(width) / 2.0, 0.0])
    return float(np.linalg.norm(middle + transform[:3, 3]))


def depth_disagreement(frame, corners, transform, width):
    """Predicted near-rim distance minus the measured one, in metres.

    None when depth had nothing to say there, which is an answer of its own:
    it means this frame cannot vouch for the size it was given, not that the
    size is wrong.
    """
    corners = np.asarray(corners, dtype=float)
    edge = measure_near_edge(frame.depth, frame.intrinsics,
                             corners[0], corners[1])
    if edge is None:
        return None, None
    return (predicted_near_edge(transform, width)
            - float(np.linalg.norm(edge["centre"]))), edge


def choose_size(corners, frame, solve, candidates=None, tolerance=0.030):
    """The candidate opening size whose pose agrees with measured depth.

    `solve` is the pose solver to try each size with, kept a parameter so this
    does not have to import the module that owns it. Returns
    (size, disagreement, edge) with size None when nothing agrees closely
    enough to be believed.
    """
    edge = measure_near_edge(frame.depth, frame.intrinsics,
                             corners[0], corners[1])
    if edge is None:
        return None, None, None
    measured = float(np.linalg.norm(edge["centre"]))
    sizes = candidates or [(a / 1000.0, b / 1000.0) for a, b in STANDARD_SIZES]

    scored = []
    for length, width in sizes:
        try:
            transform, _ = solve(corners, frame.intrinsics, length, width)
        except Exception:
            continue
        scored.append(((length, width),
                       predicted_near_edge(transform, width) - measured))
    if not scored:
        return None, None, edge

    size, disagreement = min(scored, key=lambda item: abs(item[1]))
    if abs(disagreement) > tolerance:
        return None, disagreement, edge
    return size, disagreement, edge
