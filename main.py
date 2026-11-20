from numpy_sim.potential_energy import *
from numpy_sim.mesh import Mesh
from numpy_sim.simulation import Simulation

import numpy as np

import warp as wp
import warp.render

from PIL import Image
import subprocess  # basic system calls + gifski for creating gifs
from datetime import datetime

if __name__ == "__main__":
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

    s = Simulation(m0, LinearElasticity(1, 1))

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
