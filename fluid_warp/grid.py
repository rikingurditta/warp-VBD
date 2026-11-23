import warp as wp


@wp.struct
class Grid:
    # using square grid with (width, height) points at corners
    width: int  # horizontally, grid has width points, aka width-1 cells
    height: int  # vertically, grid has height points, aka height-1 cells
    dx: float  # cell width
    q: wp.array2d[float]  # dim=(width, height)


def make_grid(width: int, height: int, dx: float) -> Grid:
    out = Grid()
    out.width = width
    out.height = height
    out.dx = dx
    out.q = wp.zeros((width, height), dtype=float)
    return out


def grid_like(g: Grid) -> Grid:
    return make_grid(g.width, g.height, g.dx)


def copy_grid(g: Grid) -> Grid:
    out = Grid()
    out.width = g.width
    out.height = g.height
    out.dx = g.dx
    out.q = wp.clone(g.q)
    return out


@wp.func
def query_grid(g: Grid, p: wp.vec2f) -> float:
    """
    query a grid that is treated as NON-STAGGERED
    aka, the bottom-left point is at (0, 0) and the top-right point is at ((width - 1) * dx, (height - 1) * dx)
    :param g: grid to query
    :param p: point to query at
    :return: linearly interpolated result  # TODO: use cubic interpolation
    """
    # transform point to mesh coordinates
    x = wp.clamp(p[0] / g.dx, 0., float(g.width))
    y = wp.clamp(p[1] / g.dx, 0., float(g.height))

    # get indices of cell corners around point
    x0 = wp.clamp(int(wp.floor(x)), 0, g.width)
    x1 = wp.clamp(int(wp.ceil(x)), 0, g.width)
    y0 = wp.clamp(int(wp.floor(y)), 0, g.height)
    y1 = wp.clamp(int(wp.ceil(y)), 0, g.height)

    t_x = x - float(x0)
    t_y = y - float(y0)

    a0 = wp.lerp(g.q[x0, y0], g.q[x1, y0], t_x)
    a1 = wp.lerp(g.q[x0, y1], g.q[x1, y1], t_x)
    return wp.lerp(a0, a1, t_y)


@wp.func
def query_staggered_grid(u: Grid, v: Grid, p: wp.vec2f) -> wp.vec2f:
    """
    query a staggered (MAC) grid (u, v)
    u has its bottom-left corner at (0, 0.5 * dx), v has its bottom-left corner at (0.5 * dx, 0)
    :param u: x-component grid, offset by 0.5 * dx in y dir
    :param v: y-component grid, offset by 0.5 * dx in x dir
    :param p: point to query at
    :return: linearly interpolated result
    """
    p_u = p
    p_u[1] -= 0.5 * u.dx
    p_v = p
    p_v[0] -= 0.5 * v.dx

    uu = query_grid(u, p_u)
    vv = query_grid(v, p_v)
    return wp.vec2f(uu, vv)


@wp.kernel
def advect_grid(q: Grid, u_a: Grid, v_a: Grid, h: float, out: Grid):
    """
    use semi-lagrangian advection to flow the scalar field q through the vector field (u_a, v_a) for time h
    :param q: scalar field to advect
    :param u_a: x-component of velocity
    :param v_a: y-component of velocity
    :param h: time to advect over
    :param out: advected result
    """
    i, j = wp.tid()
    p = wp.vec2f(float(i) * q.dx, float(j) * q.dx)
    p_backtrack = p - h * query_staggered_grid(u_a, v_a, p)  # TODO: replace with rk backtracking
    out.q[i, j] = query_grid(q, p_backtrack)


@wp.kernel
def advect_staggered_x(u: Grid, u_a: Grid, v_a: Grid, h: float, out: Grid):
    """
    use semi-lagrangian advection to flow the x-component of (u, v) through vector field (u_a, v_a) for time h
    :param u: field to advect
    :param u_a: x-component of velocity
    :param v_a: y-component of velocity
    :param h: time to advect over
    :param out: advected result
    """
    i, j = wp.tid()  # these are indices of u array, so 0 <= i <= width, 0 <= j < height
    p = wp.vec2f(float(i) * u.dx, (float(j) + 0.5) * u.dx)
    p_backtrack = p - h * query_staggered_grid(u_a, v_a, p)  # TODO: replace with rk backtracking
    uv_backtrack = query_staggered_grid(u, v_a, p_backtrack)
    out.q[i, j] = uv_backtrack[0]


@wp.kernel
def advect_staggered_y(v: Grid, u_a: Grid, v_a: Grid, h: float, out: Grid):
    """
    use semi-lagrangian advection to flow the y-component of (u, v) through vector field (u_a, v_a) for time h
    :param v: field to advect
    :param u_a: x-component of velocity
    :param v_a: y-component of velocity
    :param h: time to advect over
    :param out: advected result
    """
    i, j = wp.tid()  # these are indices of u array, so 0 <= i <= width, 0 <= j < height
    p = wp.vec2f((float(i) + 0.5) * v.dx, float(j) * v.dx)
    p_backtrack = p - h * query_staggered_grid(u_a, v_a, p)  # TODO: replace with rk backtracking
    uv_backtrack = query_staggered_grid(u_a, v, p_backtrack)
    out.q[i, j] = uv_backtrack[1]


@wp.kernel
def divergence(u: Grid, v: Grid, out: Grid):
    """
    :param u: has shape (width + 1, height)
    :param v: has shape (width, height + 1)
    :param out: has shape (width, height)
    """
    i, j = wp.tid()
    out.q[i, j] = (u.q[i + 1, j] - u.q[i, j] + v.q[i, j + 1] - v.q[i, j]) / out.dx

