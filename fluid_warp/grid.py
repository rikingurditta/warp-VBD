import numpy as np
import warp as wp
import matplotlib.pyplot
from matplotlib import pyplot as plt


@wp.struct
class Grid:
    # using square grid with (width, height) points at corners
    width: int  # horizontally, grid has width points, aka width-1 cells
    height: int  # vertically, grid has height points, aka height-1 cells
    dx: float  # cell width
    q: wp.array2d[float]  # dim=(width, height)


@wp.func
def serialize_coord(i: int, j: int, g: Grid) -> int:
    """
    :param i: row
    :param j: col
    :param g: grid
    :return: index so that g.q[i, j] = g.q.flatten()[index]
    """
    return i * g.height + j


@wp.kernel
def random_init(seed: int, g: Grid):
    i, j = wp.tid()
    rng = wp.rand_init(seed, serialize_coord(i, j, g))
    g.q[i, j] = wp.randf(rng)


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
    x = wp.clamp(p[0] / g.dx, 0., float(g.width - 1))
    y = wp.clamp(p[1] / g.dx, 0., float(g.height - 1))

    # get indices of cell corners around point
    x0 = wp.clamp(int(wp.floor(x)), 0, g.width - 1)
    x1 = wp.clamp(int(wp.ceil(x)), 0, g.width - 1)
    y0 = wp.clamp(int(wp.floor(y)), 0, g.height - 1)
    y1 = wp.clamp(int(wp.ceil(y)), 0, g.height - 1)

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

def compute_divergence(u: Grid, v: Grid) -> Grid:
    out = make_grid(v.width, u.height, u.dx)
    wp.launch(kernel=divergence,
              dim=out.q.shape,
              inputs=[u, v, out],
              device='cuda')
    return out


@wp.kernel
def sample_vector_field(u: Grid, v: Grid, u_out: Grid, v_out: Grid):
    """
    uniformly samples staggered grid vector field onto non-staggered vector field
    u_out and v_out should have same shape
    :param u: shape (width + 1, height)
    :param v: shape (width, height + 1)
    :param u_out: shape (X, Y)
    :param v_out: shape (X, Y)
    :return:
    """
    i, j = wp.tid()
    x0 = float(i) / float(u_out.width)
    y0 = float(j) / float(u_out.height)
    x = x0 * u.dx * float(v.width)
    y = y0 * v.dx * float(u.height)
    vec = query_staggered_grid(u, v, wp.vec2f(x, y))
    u_out.q[i, j] = vec[0]
    v_out.q[i, j] = vec[1]


def visualize_vector_field(u: Grid, v: Grid, ax, samples: tuple[int, int] = (10, 10), scale=1):
    W = samples[0]
    H = samples[1]
    u_out = make_grid(W, H, 1)
    v_out = copy_grid(u_out)
    wp.launch(kernel=sample_vector_field,
              dim=u_out.q.shape,
              inputs=[u, v, u_out, v_out],
              device='cuda')
    X = np.linspace(0, v.width * u.dx, W)
    Y = np.linspace(0, u.height * v.dx, H)
    ax.quiver(X, Y, u_out.q.numpy().transpose(), v_out.q.numpy().transpose(), scale=scale)
