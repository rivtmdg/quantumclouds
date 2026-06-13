import numpy as np
import matplotlib.pyplot as plt
from .constants import Z_EFF


class AtomicOrbital:
    def __init__(self, element="H", grid_size=65, scale=None, l=1, m=0):
        """Initializes a digitized 3D space dynamically optimized for the orbital scale."""
        self.element = element.upper()
        self.grid_size = grid_size
        self.l = l
        self.m = m

        self.Z_eff = Z_EFF.get(self.element, 1.0)

        # WORKSPACE BOX CALIBRATION
        if scale is None:
            if self.l == 0:
                self.scale = 3.5 / self.Z_eff
            elif self.l >= 3:
                self.scale = (5.0 * (self.l + 1)) / self.Z_eff
            else:
                self.scale = 6.0 / self.Z_eff
        else:
            self.scale = scale

        self.axis = np.linspace(-self.scale, self.scale, self.grid_size)
        self.x, self.y, self.z = np.meshgrid(self.axis, self.axis, self.axis, indexing='ij')

        self.r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        self.r[self.r == 0] = 1e-10

        self.probability_density = None

    def compute(self):
        """Computes pure math wavefunctions cleanly across quantum numbers l=0 to l=4."""
        if self.l == 0:
            self.m = 0

            # l = 0: s-orbital
        if self.l == 0:
            psi = np.exp(-self.Z_eff * self.r / 1.0)

        # l = 1: p-orbitals
        elif self.l == 1:
            if self.m == 0:
                psi = self.z * np.exp(-self.Z_eff * self.r / 2.0)
            else:
                psi = (self.x + 1j * self.y) * np.exp(-self.Z_eff * self.r / 2.0)

        # l = 2: d-orbitals
        elif self.l == 2:
            if self.m == 0:
                psi = (2 * self.z ** 2 - self.x ** 2 - self.y ** 2) * np.exp(-self.Z_eff * self.r / 3.0)
            elif abs(self.m) == 1:
                psi = self.x * self.z * np.exp(-self.Z_eff * self.r / 3.0)
            else:
                psi = (self.x ** 2 - self.y ** 2) * np.exp(-self.Z_eff * self.r / 3.0)

        # l = 3: f-orbitals
        elif self.l == 3:
            if self.m == 0:
                psi = self.z * (5 * self.z ** 2 - 3 * self.r ** 2) * np.exp(-self.Z_eff * self.r / 4.0)
            else:
                psi = self.x * (self.x ** 2 - 3 * self.y ** 2) * np.exp(-self.Z_eff * self.r / 4.0)

        # l = 4: g-orbitals
        elif self.l == 4:
            psi = (35 * self.z ** 4 - 30 * self.z ** 2 * self.r ** 2 + 3 * self.r ** 4) * np.exp(
                -self.Z_eff * self.r / 5.0)

        else:
            psi = self.z * self.r ** (self.l - 1) * np.exp(-self.Z_eff * self.r / float(self.l + 1))

        self.probability_density = np.real(psi * np.conj(psi))
        return self

    def plot2d(self, cmap='magma'):
        """Renders a flawlessly smooth 2D cross-section slice within the 3D space."""
        if self.probability_density is None:
            raise ValueError("Execution Error: Run compute() before plotting!")

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=0, azim=-90)

        mid_idx = self.grid_size // 2

        contour = ax.contourf(
            self.x[:, mid_idx, :],
            self.probability_density[:, mid_idx, :],
            self.z[:, mid_idx, :],
            zdir='y', offset=0, levels=100, cmap=cmap
        )
        fig.colorbar(contour, ax=ax, label='Relative Electron Probability Density')

        ax.set_xlim(-self.scale, self.scale)
        ax.set_ylim(-self.scale, self.scale)
        ax.set_zlim(-self.scale, self.scale)
        plt.show()

    def plot3d(self, cmap='magma'):
        """
        Renders a beautifully glowing, crash-proof VisPy engine.
        Perfectly zoomed out with the magma color theme active by default.
        """
        import vispy
        vispy.use(app='pyqt6')

        from vispy import app, scene
        from vispy.scene import visuals

        if self.probability_density is None:
            raise ValueError("Execution Error: Run compute() before plotting!")

        x_flat = self.x.flatten()
        y_flat = self.y.flatten()
        z_flat = self.z.flatten()
        prob_flat = self.probability_density.flatten()

        # Balance density dynamically so the canvas pops perfectly
        threshold_ratio = 0.005 if self.l == 0 else (0.02 if self.l >= 3 else 0.08)

        mask = prob_flat > (np.max(prob_flat) * threshold_ratio)
        x_filt = x_flat[mask]
        y_filt = y_flat[mask]
        z_filt = z_flat[mask]
        prob_filt = prob_flat[mask]

        # Map color values cleanly
        norm_colors = prob_filt / np.max(prob_filt)
        rgba_colors = plt.get_cmap(cmap)(norm_colors)

        # GLOW ENGINE TUNING: Additive blending core
        rgba_colors[:, 3] = 0.35

        canvas = scene.SceneCanvas(
            keys='interactive',
            show=True,
            size=(900, 900),
            title=f"quantumpy Engine [VisPy Core] | {self.element} State (l={self.l})"
        )

        view = canvas.central_widget.add_view()
        view.bgcolor = '#09090d'  # Space charcoal background

        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.center = (0, 0, 0)

        # CAMERA DISTANCE CALIBRATION: Optimized to capture multi-lobed structures beautifully
        view.camera.distance = self.scale * 4.5

        # Jitter implementation to organically scatter grid vectors
        grid_spacing = (2 * self.scale) / self.grid_size
        x_jit = x_filt + np.random.uniform(-0.5, 0.5, size=x_filt.shape) * grid_spacing
        y_jit = y_filt + np.random.uniform(-0.5, 0.5, size=y_filt.shape) * grid_spacing
        z_jit = z_filt + np.random.uniform(-0.5, 0.5, size=z_filt.shape) * grid_spacing

        # Create the Electron Cloud Markers
        markers = visuals.Markers()
        initial_pos = np.column_stack((x_jit, y_jit, z_jit)).astype(np.float32)
        marker_size = 4.5 if self.l == 0 else 3.0
        markers.set_data(initial_pos, face_color=rgba_colors, edge_color=None, size=marker_size, symbol='disc')

        # CRASH BYPASS & GLOW EQUATION
        markers.set_gl_state(
            blend=True,
            depth_test=False,
            blend_func=('src_alpha', 'one')
        )
        view.add(markers)

        # Precompute radial system attributes for frame transitions
        r_spatial = np.sqrt(x_filt ** 2 + y_filt ** 2 + z_filt ** 2)
        theta_spatial = np.arccos(np.clip(z_filt / (r_spatial + 1e-15), -1.0, 1.0))
        phi_base = np.arctan2(y_filt, x_filt)

        anim_state = {'frame': 0}

        def on_timer_update(event):
            anim_state['frame'] += 1
            velocity = self.m if self.m != 0 else 0.5
            time_phase = anim_state['frame'] * 0.025 * velocity

            phi_current = phi_base + time_phase
            x_curr = r_spatial * np.sin(theta_spatial) * np.cos(phi_current)
            y_curr = r_spatial * np.sin(theta_spatial) * np.sin(phi_current)

            x_curr_jit = x_curr + np.random.uniform(-0.2, 0.2, size=x_curr.shape) * grid_spacing
            y_curr_jit = y_curr + np.random.uniform(-0.2, 0.2, size=y_curr.shape) * grid_spacing
            z_curr_jit = z_jit + np.random.uniform(-0.2, 0.2, size=z_jit.shape) * grid_spacing

            new_pos = np.column_stack((x_curr_jit, y_curr_jit, z_curr_jit)).astype(np.float32)
            markers.set_data(new_pos, face_color=rgba_colors, edge_color=None, size=marker_size, symbol='disc')
            canvas.update()

        timer = app.Timer(interval=0.033, connect=on_timer_update, start=True)
        app.run()
