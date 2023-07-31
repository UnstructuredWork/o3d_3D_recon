import time
import threading
import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from kinect import Kinect
from ketisdk.sensor.realsense_sensor import RSSensor
from o3d_recon import RealtimeRecon


def set_enabled(widget, enable):
    widget.enabled = enable
    for child in widget.get_children():
        child.enabled = enable

class ReconstructionWindow:
    def __init__(self):

        self.window = gui.Application.instance.create_window(
            'Open3D - Reconstruction', 1280, 800)

        w = self.window
        em = w.theme.font_size

        spacing = int(np.round(0.25 * em))
        vspacing = int(np.round(0.5 * em))

        margins = gui.Margins(vspacing)

        # First panel
        self.panel = gui.Vert(spacing, margins)

        ## Items in fixed props
        self.fixed_prop_grid = gui.VGrid(2, spacing, gui.Margins(em, 0, em, 0))

        ### Depth scale slider
        scale_label = gui.Label('Depth scale')
        self.scale_slider = gui.Slider(gui.Slider.INT)
        self.scale_slider.set_limits(1000, 5000)
        self.scale_slider.int_value = 1000
        self.fixed_prop_grid.add_child(scale_label)
        self.fixed_prop_grid.add_child(self.scale_slider)

        voxel_size_label = gui.Label('Voxel size')
        self.voxel_size_slider = gui.Slider(gui.Slider.DOUBLE)
        self.voxel_size_slider.set_limits(0.003, 0.01)
        self.voxel_size_slider.double_value = 0.006
        self.fixed_prop_grid.add_child(voxel_size_label)
        self.fixed_prop_grid.add_child(self.voxel_size_slider)

        trunc_multiplier_label = gui.Label('Trunc multiplier')
        self.trunc_multiplier_slider = gui.Slider(gui.Slider.DOUBLE)
        self.trunc_multiplier_slider.set_limits(1.0, 20.0)
        self.trunc_multiplier_slider.double_value = 8.0
        self.fixed_prop_grid.add_child(trunc_multiplier_label)
        self.fixed_prop_grid.add_child(self.trunc_multiplier_slider)

        est_block_count_label = gui.Label('Est. blocks')
        self.est_block_count_slider = gui.Slider(gui.Slider.INT)
        self.est_block_count_slider.set_limits(40000, 100000)
        self.est_block_count_slider.int_value = 100000
        self.fixed_prop_grid.add_child(est_block_count_label)
        self.fixed_prop_grid.add_child(self.est_block_count_slider)

        est_point_count_label = gui.Label('Est. points')
        self.est_point_count_slider = gui.Slider(gui.Slider.INT)
        self.est_point_count_slider.set_limits(500000, 8000000)
        self.est_point_count_slider.int_value = 8000000
        self.fixed_prop_grid.add_child(est_point_count_label)
        self.fixed_prop_grid.add_child(self.est_point_count_slider)

        ## Items in adjustable props
        self.adjustable_prop_grid = gui.VGrid(2, spacing,
                                              gui.Margins(em, 0, em, 0))

        ### Reconstruction interval
        interval_label = gui.Label('Recon. interval')
        self.interval_slider = gui.Slider(gui.Slider.INT)
        self.interval_slider.set_limits(1, 500)
        self.interval_slider.int_value = 50
        self.adjustable_prop_grid.add_child(interval_label)
        self.adjustable_prop_grid.add_child(self.interval_slider)

        ### Depth max slider
        max_label = gui.Label('Depth max')
        self.max_slider = gui.Slider(gui.Slider.DOUBLE)
        self.max_slider.set_limits(3.0, 6.0)
        self.max_slider.double_value = 3.0
        self.adjustable_prop_grid.add_child(max_label)
        self.adjustable_prop_grid.add_child(self.max_slider)

        ### Depth diff slider
        diff_label = gui.Label('Depth diff')
        self.diff_slider = gui.Slider(gui.Slider.DOUBLE)
        self.diff_slider.set_limits(0.07, 0.5)
        self.adjustable_prop_grid.add_child(diff_label)
        self.adjustable_prop_grid.add_child(self.diff_slider)

        ### Update surface?
        update_label = gui.Label('Update surface?')
        self.update_box = gui.Checkbox('')
        self.update_box.checked = True
        self.adjustable_prop_grid.add_child(update_label)
        self.adjustable_prop_grid.add_child(self.update_box)

        ### Ray cast color?
        raycast_label = gui.Label('Raycast color?')
        self.raycast_box = gui.Checkbox('')
        self.raycast_box.checked = True
        self.adjustable_prop_grid.add_child(raycast_label)
        self.adjustable_prop_grid.add_child(self.raycast_box)

        set_enabled(self.fixed_prop_grid, True)

        ## Application control
        b = gui.ToggleSwitch('Resume/Pause')
        # b.set_on_clicked(self._on_switch)
        self._on_switch(True)

        ## Tabs
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        ### Input image tab
        tab1 = gui.Vert(0, tab_margins)
        self.input_color_image = gui.ImageWidget()
        self.input_depth_image = gui.ImageWidget()
        tab1.add_child(self.input_color_image)
        tab1.add_fixed(vspacing)
        tab1.add_child(self.input_depth_image)
        tabs.add_tab('Input images', tab1)

        ### Rendered image tab
        tab2 = gui.Vert(0, tab_margins)
        self.raycast_color_image = gui.ImageWidget()
        self.raycast_depth_image = gui.ImageWidget()
        tab2.add_child(self.raycast_color_image)
        tab2.add_fixed(vspacing)
        tab2.add_child(self.raycast_depth_image)
        tabs.add_tab('Raycast images', tab2)

        ### Info tab
        tab3 = gui.Vert(0, tab_margins)
        self.output_info = gui.Label('Output info')
        self.output_info.font_id = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
        tab3.add_child(self.output_info)
        tabs.add_tab('Info', tab3)

        self.panel.add_child(gui.Label('Starting settings'))
        self.panel.add_child(self.fixed_prop_grid)
        self.panel.add_fixed(vspacing)
        self.panel.add_child(gui.Label('Reconstruction settings'))
        self.panel.add_child(self.adjustable_prop_grid)
        self.panel.add_child(b)
        self.panel.add_stretch()
        self.panel.add_child(tabs)

        # Scene widget
        self.widget3d = gui.SceneWidget()

        # FPS panel
        self.fps_panel = gui.Vert(spacing, margins)
        self.output_fps = gui.Label('FPS: 0.0')
        self.fps_panel.add_child(self.output_fps)

        # Now add all the complex panels
        w.add_child(self.panel)
        w.add_child(self.widget3d)
        w.add_child(self.fps_panel)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])

        w.set_on_layout(self._on_layout)
        w.set_on_close(self._on_close)

        self.is_done = False

        self.is_started = False
        self.is_running = False
        self.is_surface_updated = False

        self.idx = 0
        self.poses = []

        # Start running
        threading.Thread(name='UpdateMain', target=self.run_recon).start()

    def _on_layout(self, ctx):
        em = ctx.theme.font_size

        panel_width = 20 * em
        rect = self.window.content_rect

        self.panel.frame = gui.Rect(rect.x, rect.y, panel_width, rect.height)

        x = self.panel.frame.get_right()
        self.widget3d.frame = gui.Rect(x, rect.y,
                                       rect.get_right() - x, rect.height)

        fps_panel_width = 7 * em
        fps_panel_height = 2 * em
        self.fps_panel.frame = gui.Rect(rect.get_right() - fps_panel_width,
                                        rect.y, fps_panel_width,
                                        fps_panel_height)

        # Toggle callback: application's main controller

    def _on_switch(self, is_on):
        # if not self.is_started:
        gui.Application.instance.post_to_main_thread(
            self.window, self._on_start)
        # self.is_running = not self.is_running

        # On start: point cloud buffer and model initialization.

    def _on_start(self):
        max_points = self.est_point_count_slider.int_value

        pcd_placeholder = o3d.t.geometry.PointCloud(
            o3c.Tensor(np.zeros((max_points, 3), dtype=np.float32)))
        pcd_placeholder.point.colors = o3c.Tensor(
            np.zeros((max_points, 3), dtype=np.float32))
        mat = rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.sRGB_color = True
        self.widget3d.scene.scene.add_geometry('points', pcd_placeholder, mat)

        # self.model = o3d.t.pipelines.slam.Model(
        #     self.voxel_size_slider.double_value, 16,
        #     self.est_block_count_slider.int_value, o3c.Tensor(np.eye(4)),
        #     o3c.Device('CUDA:0'))
        self.is_started = True

        set_enabled(self.fixed_prop_grid, False)
        set_enabled(self.adjustable_prop_grid, True)

    def _on_close(self):
        self.is_done = True

        # if self.is_started:
        #     print('Saving model to {}...'.format(config.path_npz))
        #     self.model.voxel_grid.save(config.path_npz)
        #     print('Finished.')
        #
        #     mesh_fname = '.'.join(config.path_npz.split('.')[:-1]) + '.ply'
        #     print('Extracting and saving mesh to {}...'.format(mesh_fname))
        #     mesh = extract_trianglemesh(self.model.voxel_grid, config,
        #                                 mesh_fname)
        #     print('Finished.')
        #
        #     log_fname = '.'.join(config.path_npz.split('.')[:-1]) + '.log'
        #     print('Saving trajectory to {}...'.format(log_fname))
        #     save_poses(log_fname, self.poses)
        #     print('Finished.')

        return True

    def init_render(self, depth_ref, color_ref):
        # self.input_depth_image.update_image(
        #     depth_ref.colorize_depth(float(self.scale_slider.int_value),
        #                              0.2,
        #                              self.max_slider.double_value).to_legacy())
        # self.input_color_image.update_image(color_ref.to_legacy())
        #
        # self.raycast_depth_image.update_image(
        #     depth_ref.colorize_depth(float(self.scale_slider.int_value),
        #                              3.0,
        #                              self.max_slider.double_value).to_legacy())
        # self.raycast_color_image.update_image(color_ref.to_legacy())
        self.window.set_needs_layout()

        bbox = o3d.geometry.AxisAlignedBoundingBox([-5, -5, -5], [5, 5, 5])
        self.widget3d.setup_camera(60, bbox, [0, 0, 0])
        self.widget3d.look_at([0, 0, 0], [0, -1, -3], [0, -1, 0])

    def update_render(self, input_depth, input_color, raycast_depth,
                      raycast_color, pcd, frustum):
        # self.input_depth_image.update_image(
        #     input_depth.colorize_depth(
        #         float(self.scale_slider.int_value), 0.2,
        #         self.max_slider.double_value).to_legacy())
        # self.input_color_image.update_image(input_color.to_legacy())
        #
        # self.raycast_depth_image.update_image(
        #     raycast_depth.colorize_depth(
        #         float(self.scale_slider.int_value), 3.0,
        #         self.max_slider.double_value).to_legacy())
        # self.raycast_color_image.update_image(
        #     (raycast_color).to(o3c.uint8, False, 255.0).to_legacy())

        # if self.is_scene_updated:
        if pcd is not None and pcd.point.positions.shape[0] > 0:
            self.widget3d.scene.scene.update_geometry(
                'points', pcd, rendering.Scene.UPDATE_POINTS_FLAG |
                               rendering.Scene.UPDATE_COLORS_FLAG)

        self.widget3d.scene.remove_geometry("frustum")
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 5.0
        self.widget3d.scene.add_geometry("frustum", frustum, mat)

    def run_recon(self):
        # sensor = Kinect()
        # sensor.start(1536)
        # intrinsic = sensor.intrinsic_color

        sensor = RSSensor()
        sensor.start()
        intrinsic = np.array([[sensor.info.fx, 0, sensor.info.cx],
                              [0, sensor.info.fy, sensor.info.cy],
                              [0, 0, 1]])

        color, depth = sensor.get_data()

        time.sleep(1)

        slam = RealtimeRecon(voxel_size=0.006, intrinsic=intrinsic, send_ros=False)

        if slam.is_started:

            while True:
                start = time.time()
                t11 = time.time()
                color, depth = sensor.get_data()
                # imu = sensor.get_imu()
                t21 = time.time()
                print((t21 - t11) * 1000)

                t1 = time.time() * 1000
                pcd, curr_points, curr_colors, prev_points, prev_colors = slam(color, depth)

                # print(curr_colors.shape)
                # print(curr_points.shape)
                # print()
                # print(curr_colors.shape, curr_colors.shape)
                t2 = time.time() * 1000
                t = t2 - t1
                # print(t)

                ################
                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.init_render(slam.depth_ref, slam.color_ref))

                frustum = o3d.geometry.LineSet.create_camera_visualization(
                    slam.color_ref.columns, slam.color_ref.rows, slam.intrinsic.numpy(),
                    np.linalg.inv(slam.T_frame_to_model.cpu().numpy()), 0.2)
                frustum.paint_uniform_color([0.961, 0.475, 0.000])

                # Output FPS
                if (self.idx % 30 == 0):
                    end = time.time()
                    elapsed = end - start
                    start = time.time()
                    self.output_fps.text = 'FPS: {:.3f}'.format(30 /
                                                                elapsed)

                # Output info
                info = 'Frame {}\n\n'.format(self.idx)
                info += 'Transformation:\n{}\n'.format(
                    np.array2string(slam.T_frame_to_model.numpy(),
                                    precision=3,
                                    max_line_width=40,
                                    suppress_small=True))
                # info += 'Active voxel blocks: {}/{}\n'.format(
                #     self.model.voxel_grid.hashmap().size(),
                #     self.model.voxel_grid.hashmap().capacity())
                info += 'Surface points: {}/{}\n'.format(
                    0 if pcd is None else pcd.point.positions.shape[0],
                    self.est_point_count_slider.int_value)

                self.output_info.text = info

                gui.Application.instance.post_to_main_thread(
                    self.window, lambda: self.update_render(
                        slam.input_frame.get_data_as_image('depth'),
                        slam.input_frame.get_data_as_image('color'),
                        slam.raycast_frame.get_data_as_image('depth'),
                        slam.raycast_frame.get_data_as_image('color'), pcd, frustum))

                self.idx += 1

                # time.sleep(0.5)


if __name__ == '__main__':
    app = gui.Application.instance
    app.initialize()
    w = ReconstructionWindow()
    app.run()
