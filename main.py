import numpy as np
import warp as wp
import warp.render

from PIL import Image
import subprocess  # basic system calls + gifski for creating gifs
from datetime import datetime

from fem_numpy.potential_energy import *
from fem_numpy.mesh import Mesh
from fem_numpy.simulation import Simulation as Simulation_np

from fem_warp.mesh import WarpMesh
from fem_warp.potential_energy import MaterialParams
from fem_warp.simulation import Simulation as Simulation_wp


def run_numpy_fem():
    save_gif = False
    headless = False
    screen_width = 1024
    screen_height = 1024
    max_frames = 1e10
    h = 0.01
    wp.init()
    renderer = warp.render.OpenGLRenderer(draw_sky=False,
                                          draw_grid=False,
                                          draw_axis=False,
                                          camera_pos=(2, 0, 10),
                                          render_wireframe=True,
                                          background_color=(1, 1, 1),
                                          screen_width=screen_width,
                                          screen_height=screen_height,
                                          headless=headless)

    m0 = Mesh(filename='test_mesh.txt')
    m0.centre_at_zero()

    m_curr = m0.copy()
    m_curr.V[:, 0] += 0.001
    m_curr.V[:, 1] *= 1.01

    m_prev = m0.copy()

    s = Simulation_np(m0, LinearElasticity(1, 1))

    if save_gif:
        subprocess.run(['mkdir', 'run'])
        subprocess.run('rm run/*', shell=True)

    frame = 0
    while renderer.is_running() and frame < max_frames:
        # print(f'x={m_curr.x()}')
        m_next = s.timestep(m_curr, m_prev, h)
        m_prev = m_curr.copy()
        m_curr = m_next.copy()

        renderer.begin_frame()
        renderer.render_mesh('undeformed', m0.V_3d(), m0.E, colors=(1., 0., 0.))
        renderer.render_mesh('deformed', m_curr.V_3d(), m_curr.E, colors=(0., 0., 1.))
        renderer.end_frame()

        print(s.E_total(m_curr, (m_curr.x() - m_prev.x()) / h))

        if save_gif:
            image = wp.empty((screen_height, screen_width, 3), dtype=wp.uint8, device='cuda')
            renderer.get_pixels(image, split_up_tiles=False, use_uint8=True)
            im = Image.fromarray(image.numpy())
            im.save(f'run/frame{frame:03d}.png')

        print(f'frame: {frame}')
        frame += 1

    if save_gif:
        subprocess.run(['mkdir', 'animations'])
        print('making gif...')
        time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        subprocess.run(f'gifski --output animations/ani-{time}.gif --fps 60 run/*.png', shell=True)
        print('done!')


def run_warp_fem():
    h = 0.01
    max_frames = 1e10

    wp.init()
    renderer = warp.render.OpenGLRenderer(draw_sky=False,
                                     draw_grid=False,
                                     draw_axis=False,
                                     camera_pos=(2, 0, 10),
                                     render_wireframe=True,
                                     background_color=(1, 1, 1),
                                     screen_width=1024,
                                     screen_height=1024,
                                     headless=False)
    m0 = Mesh(filename='test_mesh.txt')
    m0.centre_at_zero()
    m_def = m0.copy()
    m_def.V[:, 0] += 0.001
    m_def.V[:, 1] *= 1.01

    p = MaterialParams()
    p.mu_lame = 1.
    p.lambda_lame = 1.

    wm0 = WarpMesh(m0)
    m_curr = WarpMesh(m_def)
    m_prev = wm0

    s = Simulation_wp(wm0, p)

    frame = 0
    while renderer.is_running() and frame < max_frames:
        m_next = s.timestep(m_curr, m_prev, h)
        m_prev = m_curr.copy()
        m_curr = m_next.copy()

        renderer.begin_frame()
        renderer.render_mesh('undeformed', m0.V_3d(), m0.E, colors=(1., 0., 0.))
        renderer.render_mesh('deformed', m_curr.V_3d(), m0.E, colors=(0., 0., 1.))
        renderer.end_frame()
        print(f'frame: {frame}')
        frame += 1

if __name__ == "__main__":
    # run_numpy_fem()
    run_warp_fem()