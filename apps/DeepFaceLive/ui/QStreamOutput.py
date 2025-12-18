from localization import L
from resources.fonts import QXFontDB
from xlib import qt as qtx
from xlib.qt.widgets.QXLabel import QXLabel

from ..backend import OutputTarget, StreamOutput, VirtualCameraOutput
from .widgets.QBackendPanel import QBackendPanel
from .widgets.QCheckBoxCSWFlag import QCheckBoxCSWFlag
from .widgets.QComboBoxCSWDynamicSingleSwitch import \
    QComboBoxCSWDynamicSingleSwitch
from .widgets.QErrorCSWError import QErrorCSWError
from .widgets.QLabelCSWNumber import QLabelCSWNumber
from .widgets.QLabelPopupInfo import QLabelPopupInfo
from .widgets.QLabelPopupInfoCSWInfoLabel import QLabelPopupInfoCSWInfoLabel
from .widgets.QLineEditCSWText import QLineEditCSWText
from .widgets.QPathEditCSWPaths import QPathEditCSWPaths
from .widgets.QSpinBoxCSWNumber import QSpinBoxCSWNumber
from .widgets.QXPushButtonCSWSignal import QXPushButtonCSWSignal


class QStreamOutput(QBackendPanel):
    def __init__(self, backend: StreamOutput, camera_backend: VirtualCameraOutput = None):
        self._camera_backend = camera_backend
        self._camera_state_label = qtx.QXLabel()
        self._camera_backend_state = None

        cs = backend.get_control_sheet()
        camera_cs = camera_backend.get_control_sheet() if camera_backend is not None else None

        cs.output_target.call_on_selected(self._on_output_target_selected)

        q_average_fps_label = QLabelPopupInfo(label=L('@QStreamOutput.avg_fps'), popup_info_text=L('@QStreamOutput.help.avg_fps'))
        q_average_fps = QLabelCSWNumber(cs.avg_fps, reflect_state_widgets=[q_average_fps_label])

        q_source_type_label = QLabelPopupInfo(label=L('@QStreamOutput.source_type'))
        q_source_type = QComboBoxCSWDynamicSingleSwitch(cs.source_type, reflect_state_widgets=[q_source_type_label])

        q_output_target_label = QLabelPopupInfo(label=L('@VirtualCameraOutput.output_target.label'),
                                                popup_info_text=L('@VirtualCameraOutput.output_target.help'))
        q_output_target = QComboBoxCSWDynamicSingleSwitch(cs.output_target, reflect_state_widgets=[q_output_target_label])

        q_show_hide_window = QXPushButtonCSWSignal(cs.show_hide_window, text=L('@QStreamOutput.show_hide_window'), button_size=(None, 22))

        q_aligned_face_id_label = QLabelPopupInfo(label=L('@QStreamOutput.aligned_face_id'), popup_info_text=L('@QStreamOutput.help.aligned_face_id'))
        q_aligned_face_id = QSpinBoxCSWNumber(cs.aligned_face_id, reflect_state_widgets=[q_aligned_face_id_label])

        q_target_delay_label = QLabelPopupInfo(label=L('@QStreamOutput.target_delay'), popup_info_text=L('@QStreamOutput.help.target_delay'))
        q_target_delay = QSpinBoxCSWNumber(cs.target_delay, reflect_state_widgets=[q_target_delay_label])

        q_save_sequence_path_label = QLabelPopupInfo(label=L('@QStreamOutput.save_sequence_path'), popup_info_text=L('@QStreamOutput.help.save_sequence_path'))
        q_save_sequence_path = QPathEditCSWPaths(cs.save_sequence_path, reflect_state_widgets=[q_target_delay_label, q_save_sequence_path_label])
        q_save_sequence_path_error = QErrorCSWError(cs.save_sequence_path_error)

        q_save_fill_frame_gap_label = QLabelPopupInfo(label=L('@QStreamOutput.save_fill_frame_gap'), popup_info_text=L('@QStreamOutput.help.save_fill_frame_gap'))
        q_save_fill_frame_gap = QCheckBoxCSWFlag(cs.save_fill_frame_gap, reflect_state_widgets=[q_save_fill_frame_gap_label])

        q_is_streaming_label = QLabelPopupInfo(label='mpegts udp://')
        q_is_streaming = QCheckBoxCSWFlag(cs.is_streaming, reflect_state_widgets=[q_is_streaming_label])

        q_stream_addr = QLineEditCSWText(cs.stream_addr, font=QXFontDB.get_fixedwidth_font())
        q_stream_port = QSpinBoxCSWNumber(cs.stream_port)

        grid_l = qtx.QXGridLayout(spacing=5)
        row = 0
        grid_l.addWidget(q_average_fps_label, row, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
        grid_l.addWidget(q_average_fps, row, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
        row += 1
        grid_l.addWidget(q_source_type_label, row, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
        grid_l.addWidget(q_source_type, row, 1, 1, 1, alignment=qtx.AlignLeft | qtx.AlignVCenter)
        grid_l.addWidget(q_show_hide_window, row, 2, 1, 1)
        row += 1
        grid_l.addWidget(q_output_target_label, row, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
        grid_l.addWidget(q_output_target, row, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
        row += 1
        grid_l.addWidget(q_aligned_face_id_label, row, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
        grid_l.addWidget(q_aligned_face_id, row, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
        row += 1
        grid_l.addWidget(q_target_delay_label, row, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
        grid_l.addWidget(q_target_delay, row, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
        row += 1

        grid_l.addWidget(q_save_sequence_path_label, row, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
        grid_l.addWidget(q_save_sequence_path, row, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
        row += 1
        grid_l.addLayout(qtx.QXHBoxLayout([q_save_fill_frame_gap, 4, q_save_fill_frame_gap_label]), row, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
        row += 1
        grid_l.addWidget(q_save_sequence_path_error, row, 0, 1, 3)
        row += 1
        grid_l.addLayout(qtx.QXHBoxLayout([q_is_streaming, 4, q_is_streaming_label]), row, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
        grid_l.addLayout(qtx.QXHBoxLayout([q_stream_addr, qtx.QXLabel(text=':'), q_stream_port]), row, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
        row += 1

        layouts = [grid_l]

        if camera_cs is not None:
            camera_backend.call_on_state_change(self._on_camera_backend_state_change)

            cam_status = QLabelPopupInfoCSWInfoLabel(camera_cs.status)
            cam_error = QErrorCSWError(camera_cs.error)

            cam_enabled_label = QLabelPopupInfo(label=L('@VirtualCameraOutput.output_enabled'))
            cam_enabled = QCheckBoxCSWFlag(camera_cs.output_enabled, reflect_state_widgets=[cam_enabled_label])

            cam_device_label = QLabelPopupInfo(label=L('@VirtualCameraOutput.device'),
                                               popup_info_text=L('@VirtualCameraOutput.device.help'))
            cam_device = QComboBoxCSWDynamicSingleSwitch(camera_cs.device, reflect_state_widgets=[cam_device_label])

            cam_resolution_label = QLabelPopupInfo(label=L('@VirtualCameraOutput.resolution'))
            cam_width = QSpinBoxCSWNumber(camera_cs.width)
            cam_height = QSpinBoxCSWNumber(camera_cs.height)

            cam_fps_label = QLabelPopupInfo(label=L('@VirtualCameraOutput.fps_limit'))
            cam_fps = QSpinBoxCSWNumber(camera_cs.fps_limit)

            cam_avg_fps_label = QLabelPopupInfo(label=L('@VirtualCameraOutput.avg_fps'))
            cam_avg_fps = QLabelCSWNumber(camera_cs.avg_fps, reflect_state_widgets=[cam_avg_fps_label])

            cam_start_button = qtx.QXPushButton(text=L('@QBackendPanel.start'), released=self._start_camera_backend, fixed_width=90)
            cam_stop_button = qtx.QXPushButton(text=L('@QBackendPanel.stop'), released=self._stop_camera_backend, fixed_width=90)

            cam_grid = qtx.QXGridLayout(spacing=5)
            crow = 0
            cam_grid.addWidget(self._camera_state_label, crow, 0, 1, 3, alignment=qtx.AlignLeft | qtx.AlignVCenter)
            crow += 1
            cam_grid.addWidget(cam_status, crow, 0, 1, 3, alignment=qtx.AlignLeft | qtx.AlignVCenter)
            crow += 1
            cam_grid.addLayout(qtx.QXHBoxLayout([cam_enabled, 4, cam_enabled_label]), crow, 0, 1, 3, alignment=qtx.AlignLeft | qtx.AlignVCenter)
            crow += 1
            cam_grid.addWidget(cam_device_label, crow, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
            cam_grid.addWidget(cam_device, crow, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
            crow += 1
            cam_grid.addWidget(cam_resolution_label, crow, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
            cam_grid.addLayout(qtx.QXHBoxLayout([cam_width, qtx.QXLabel(text='x'), cam_height]), crow, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
            crow += 1
            cam_grid.addWidget(cam_fps_label, crow, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
            cam_grid.addWidget(cam_fps, crow, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
            crow += 1
            cam_grid.addWidget(cam_avg_fps_label, crow, 0, 1, 1, alignment=qtx.AlignRight | qtx.AlignVCenter)
            cam_grid.addWidget(cam_avg_fps, crow, 1, 1, 2, alignment=qtx.AlignLeft | qtx.AlignVCenter)
            crow += 1
            cam_grid.addLayout(qtx.QXHBoxLayout([cam_start_button, cam_stop_button], spacing=6), crow, 0, 1, 3, alignment=qtx.AlignLeft | qtx.AlignVCenter)
            crow += 1
            cam_grid.addWidget(cam_error, crow, 0, 1, 3)

            layouts.append(qtx.QXFrame(fixed_height=1))
            layouts.append(cam_grid)

        super().__init__(backend, L('@QStreamOutput.module_title'),
                         layout=qtx.QXVBoxLayout(layouts))
        self._update_camera_state_label()

    def _start_camera_backend(self):
        if self._camera_backend is not None:
            self._camera_backend.start()

    def _stop_camera_backend(self):
        if self._camera_backend is not None:
            self._camera_backend.stop()

    def _on_output_target_selected(self, _, output_target: OutputTarget):
        if self._camera_backend is None:
            return
        camera_cs = self._camera_backend.get_control_sheet()
        enable_camera = output_target in [OutputTarget.VIRTUAL_CAMERA, OutputTarget.FILE_AND_VIRTUAL]
        camera_cs.output_enabled.set_flag(enable_camera)

    def _on_camera_backend_state_change(self, backend, started, starting, stopping, stopped, busy):
        self._camera_backend_state = (started, starting, stopping, stopped, busy)
        self._update_camera_state_label()

    def _update_camera_state_label(self):
        if self._camera_backend_state is None:
            self._camera_state_label.setText(L('@VirtualCameraOutput.state.unknown'))
            self._camera_state_label.set_color('gray')
            return
        started, starting, stopping, stopped, busy = self._camera_backend_state
        state_text = L('@VirtualCameraOutput.state.stopped')
        color = 'gray'
        if starting:
            state_text = L('@VirtualCameraOutput.state.starting')
            color = 'yellow'
        elif stopping:
            state_text = L('@VirtualCameraOutput.state.stopping')
            color = 'orange'
        elif busy:
            state_text = L('@VirtualCameraOutput.state.busy')
            color = 'orange'
        elif started:
            state_text = L('@VirtualCameraOutput.state.started')
            color = 'lime'
        self._camera_state_label.setText(state_text)
        self._camera_state_label.set_color(color)

    def _on_backend_state_change(self, backend, started, starting, stopping, stopped, busy):
        super()._on_backend_state_change(backend, started, starting, stopping, stopped, busy)
        qtx.show_and_enable([self._content_widget])
