import importlib.util
import platform
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from xlib import time as lib_time
from xlib.image import ImageProcessor
from xlib.mp import csw as lib_csw

from .BackendBase import (BackendConnection, BackendDB, BackendHost,
                          BackendSignal, BackendWeakHeap, BackendWorker,
                          BackendWorkerState)
from .StreamOutput import SourceType, ViewModeNames


class VirtualCameraOutput(BackendHost):
    """
    Pushes frames into a virtual camera device.
    """

    def __init__(self, weak_heap: BackendWeakHeap,
                       reemit_frame_signal: BackendSignal,
                       bc_in: BackendConnection,
                       bc_out: BackendConnection,
                       backend_db: BackendDB = None):

        super().__init__(backend_db=backend_db,
                         sheet_cls=Sheet,
                         worker_cls=VirtualCameraOutputWorker,
                         worker_state_cls=WorkerState,
                         worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out])

    def get_control_sheet(self) -> 'Sheet.Host': return super().get_control_sheet()


class VirtualCameraOutputWorker(BackendWorker):
    def get_state(self) -> 'WorkerState': return super().get_state()
    def get_control_sheet(self) -> 'Sheet.Worker': return super().get_control_sheet()

    def on_start(self, weak_heap: BackendWeakHeap,
                       reemit_frame_signal: BackendSignal,
                       bc_in: BackendConnection,
                       bc_out: BackendConnection):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out

        self._pyvirtualcam = None
        self._pyvirtualcam_checked = False
        self._camera = None
        self._camera_cfg: Tuple[Optional[str], Optional[int], Optional[int], Optional[float]] = (None, None, None, None)
        self._frame_queue = deque(maxlen=3)
        self._fps_counter = lib_time.FPSCounter()
        self._last_fps_value = 0.0
        self._pending_bcd = None
        self._auto_camera_attempted = False

        state, cs = self.get_state(), self.get_control_sheet()

        cs.source_type.call_on_selected(self.on_cs_source_type)
        cs.aligned_face_id.call_on_number(self.on_cs_aligned_face_id)
        cs.output_enabled.call_on_flag(self.on_cs_output_enabled)
        cs.device.call_on_selected(self.on_cs_device_selected)
        cs.width.call_on_number(self.on_cs_width)
        cs.height.call_on_number(self.on_cs_height)
        cs.fps_limit.call_on_number(self.on_cs_fps_limit)

        cs.status.enable()
        cs.error.enable()
        cs.avg_fps.enable()
        cs.avg_fps.set_config(lib_csw.Number.Config(min=0, max=240, decimals=1, read_only=True))
        cs.avg_fps.set_number(0)

        cs.source_type.enable()
        cs.source_type.set_choices(SourceType, ViewModeNames, none_choice_name='@misc.menu_select')
        cs.source_type.select(state.source_type if state.source_type is not None else SourceType.MERGED_FRAME_OR_SOURCE_FRAME)

        cs.aligned_face_id.disable()

        cs.output_enabled.enable()
        cs.output_enabled.set_flag(state.output_enabled if state.output_enabled is not None else False)

        cs.device.enable()
        devices = self._refresh_devices()
        if state.device_name is None and len(devices) > 0:
            state.device_name = devices[0]
        if state.device_name is not None:
            cs.device.select(state.device_name)

        cs.width.enable()
        cs.height.enable()
        cs.width.set_config(lib_csw.Number.Config(min=0, max=7680, step=2, decimals=0, zero_is_auto=True, allow_instant_update=True))
        cs.height.set_config(lib_csw.Number.Config(min=0, max=4320, step=2, decimals=0, zero_is_auto=True, allow_instant_update=True))
        cs.width.set_number(state.width if state.width is not None else 0)
        cs.height.set_number(state.height if state.height is not None else 0)

        cs.fps_limit.enable()
        cs.fps_limit.set_config(lib_csw.Number.Config(min=0, max=240, decimals=2, zero_is_auto=True, allow_instant_update=True))
        cs.fps_limit.set_number(state.fps_limit if state.fps_limit is not None else 30)

        self._set_status('@VirtualCameraOutput.status.stopped')
        cs.error.set_error(None)

    def on_stop(self):
        self._close_camera()

    def on_cs_source_type(self, idx, source_type):
        state, cs = self.get_state(), self.get_control_sheet()
        if source_type in [SourceType.ALIGNED_FACE, SourceType.ALIGNED_N_SWAPPED_FACE]:
            cs.aligned_face_id.enable()
            cs.aligned_face_id.set_config(lib_csw.Number.Config(min=0, max=16, step=1, allow_instant_update=True))
            cs.aligned_face_id.set_number(state.aligned_face_id or 0)
        else:
            cs.aligned_face_id.disable()
        state.source_type = source_type
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_aligned_face_id(self, aligned_face_id):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.aligned_face_id.get_config()
        aligned_face_id = state.aligned_face_id = int(np.clip(aligned_face_id, cfg.min, cfg.max))
        cs.aligned_face_id.set_number(aligned_face_id)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_output_enabled(self, output_enabled):
        state, _ = self.get_state(), self.get_control_sheet()
        state.output_enabled = output_enabled
        self.save_state()
        if not output_enabled:
            self._set_status('@VirtualCameraOutput.status.disabled')
            self._frame_queue.clear()
            self._close_camera()

    def on_cs_device_selected(self, idx, device_name):
        state, _ = self.get_state(), self.get_control_sheet()
        state.device_name = device_name
        self.save_state()
        self._close_camera()

    def on_cs_width(self, width):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.width.get_config()
        width = state.width = int(np.clip(width, cfg.min, cfg.max))
        cs.width.set_number(width)
        self.save_state()
        self._close_camera()

    def on_cs_height(self, height):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.height.get_config()
        height = state.height = int(np.clip(height, cfg.min, cfg.max))
        cs.height.set_number(height)
        self.save_state()
        self._close_camera()

    def on_cs_fps_limit(self, fps_limit):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.fps_limit.get_config()
        fps_limit = state.fps_limit = float(np.clip(fps_limit, cfg.min, cfg.max))
        cs.fps_limit.set_number(fps_limit)
        self.save_state()
        self._close_camera()

    def on_tick(self):
        cs, state = self.get_control_sheet(), self.get_state()

        bcd = self.bc_in.read(timeout=0.005)
        if bcd is not None:
            bcd.assign_weak_heap(self.weak_heap)
            frame = self._extract_frame(bcd, state)
            if frame is not None and state.output_enabled:
                self._frame_queue.append((frame, bcd.get_frame_fps()))
            self._pending_bcd = bcd

        self._drain_queue()
        self._forward_pending_bcd()

    def _drain_queue(self):
        state, cs = self.get_state(), self.get_control_sheet()

        if not state.output_enabled:
            return

        if not self._ensure_pyvirtualcam():
            return

        frame_fps = None
        if len(self._frame_queue) > 0:
            frame, frame_fps = self._frame_queue.pop()
            self._frame_queue.clear()
        else:
            return

        target_w, target_h = self._resolve_resolution(state, frame.shape[1], frame.shape[0])
        target_fps = self._resolve_fps(state, frame_fps)

        if target_w is None or target_h is None:
            cs.error.set_error('@VirtualCameraOutput.error.no_resolution')
            return

        if not self._ensure_camera(state.device_name, target_w, target_h, target_fps):
            return

        prepared_frame = self._prepare_frame(frame, target_w, target_h)
        if prepared_frame is None:
            cs.error.set_error('@VirtualCameraOutput.error.empty_frame')
            return

        try:
            self._camera.send(prepared_frame)
            self._camera.sleep_until_next_frame()
            self._last_fps_value = self._fps_counter.step()
            cs.avg_fps.set_number(self._last_fps_value)
            cs.error.set_error(None)
            self._set_status('@VirtualCameraOutput.status.running',
                             info_lines=[f'{target_w}x{target_h} @ {target_fps:.2f} FPS',
                                         state.device_name or '@VirtualCameraOutput.status.default_device'])
        except Exception as e:
            self._set_status('@VirtualCameraOutput.status.error')
            cs.error.set_error(str(e))
            self._close_camera()

    def _forward_pending_bcd(self):
        if self._pending_bcd is None or self.bc_out is None:
            return

        if self.bc_out.is_full_read(1):
            self.bc_out.write(self._pending_bcd)
            self._pending_bcd = None

    def _ensure_pyvirtualcam(self) -> bool:
        if self._pyvirtualcam_checked:
            return self._pyvirtualcam is not None

        self._pyvirtualcam_checked = True
        spec = importlib.util.find_spec('pyvirtualcam')
        if spec is None:
            self._set_missing_dependency_error()
            return False

        if spec.loader is None:
            self._set_missing_dependency_error()
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._pyvirtualcam = module
        self._set_status('@VirtualCameraOutput.status.ready')
        return True

    def _set_missing_dependency_error(self):
        system = platform.system()
        msg = '@VirtualCameraOutput.error.missing_pyvirtualcam'
        if system == 'Windows':
            msg = '@VirtualCameraOutput.error.missing_pyvirtualcam_windows'
        elif system == 'Linux':
            msg = '@VirtualCameraOutput.error.missing_pyvirtualcam_linux'
        elif system == 'Darwin':
            msg = '@VirtualCameraOutput.error.missing_pyvirtualcam_macos'

        self.get_control_sheet().error.set_error(msg)
        self._set_status('@VirtualCameraOutput.status.error')

    def _ensure_camera(self, device_name: Optional[str], width: int, height: int, fps: float) -> bool:
        cs = self.get_control_sheet()

        if not self._check_platform_requirements():
            return False

        if device_name is None and not self._auto_camera_attempted:
            # Try default virtual cam creation (if driver present) once.
            self._auto_camera_attempted = True
            device_name = None

        cfg_device, cfg_w, cfg_h, cfg_fps = self._camera_cfg
        if self._camera is not None and (cfg_device != device_name or cfg_w != width or cfg_h != height or cfg_fps != fps):
            self._close_camera()

        if self._camera is None:
            try:
                self._camera = self._pyvirtualcam.Camera(width=width, height=height, fps=fps, device=device_name)
                self._camera_cfg = (device_name, width, height, fps)
                cs.error.set_error(None)
            except Exception as e:
                cs.error.set_error(str(e))
                self._set_status('@VirtualCameraOutput.status.error')
                self._camera_cfg = (None, None, None, None)
                return False

        return True

    def _check_platform_requirements(self) -> bool:
        cs = self.get_control_sheet()
        system = platform.system()
        if system == 'Linux':
            devices = list(Path('/dev').glob('video*'))
            if len(devices) == 0:
                cs.error.set_error('@VirtualCameraOutput.error.no_v4l2loopback')
                self._set_status('@VirtualCameraOutput.status.error')
                return False
        return True

    def _refresh_devices(self):
        cs, state = self.get_control_sheet(), self.get_state()

        device_names: List[str] = []
        if self._ensure_pyvirtualcam():
            list_cameras = getattr(self._pyvirtualcam, 'list_cameras', None)
            if callable(list_cameras):
                device_names = list(list_cameras() or [])

        if platform.system() == 'Linux' and len(device_names) == 0:
            device_names = [str(p) for p in Path('/dev').glob('video*')]

        cs.device.set_choices(device_names, none_choice_name='@misc.menu_select')
        if len(device_names) > 0:
            self._auto_camera_attempted = False

        if state.device_name is not None and state.device_name not in device_names:
            state.device_name = None
            self.save_state()
        return device_names

    def _resolve_resolution(self, state: 'WorkerState', source_w: int, source_h: int) -> Tuple[Optional[int], Optional[int]]:
        width = state.width if state.width not in [None, 0] else source_w
        height = state.height if state.height not in [None, 0] else source_h
        return width, height

    def _resolve_fps(self, state: 'WorkerState', frame_fps: Optional[float]) -> float:
        if state.fps_limit is None or state.fps_limit == 0:
            return frame_fps if frame_fps is not None and frame_fps > 0 else 30.0
        return state.fps_limit

    def _extract_frame(self, bcd, state: 'WorkerState') -> Optional[np.ndarray]:
        source_type = state.source_type
        if source_type is None:
            return None

        view_image = None

        if source_type == SourceType.SOURCE_FRAME:
            view_image = bcd.get_image(bcd.get_frame_image_name())
        elif source_type in [SourceType.MERGED_FRAME, SourceType.MERGED_FRAME_OR_SOURCE_FRAME]:
            view_image = bcd.get_image(bcd.get_merged_image_name())
            if view_image is None and source_type == SourceType.MERGED_FRAME_OR_SOURCE_FRAME:
                view_image = bcd.get_image(bcd.get_frame_image_name())
        elif source_type == SourceType.SWAPPED_FACE:
            for fsi in bcd.get_face_swap_info_list():
                view_image = bcd.get_image(fsi.face_swap_image_name)
                if view_image is not None:
                    break
        elif source_type == SourceType.ALIGNED_FACE:
            aligned_face_id = state.aligned_face_id or 0
            for i, fsi in enumerate(bcd.get_face_swap_info_list()):
                if aligned_face_id == i:
                    view_image = bcd.get_image(fsi.face_align_image_name)
                    break
        elif source_type in [SourceType.SOURCE_N_MERGED_FRAME, SourceType.SOURCE_N_MERGED_FRAME_OR_SOURCE_FRAME]:
            source_frame = bcd.get_image(bcd.get_frame_image_name())
            merged_frame = bcd.get_image(bcd.get_merged_image_name())
            if merged_frame is None and source_type == SourceType.SOURCE_N_MERGED_FRAME_OR_SOURCE_FRAME:
                merged_frame = source_frame
            if source_frame is not None and merged_frame is not None:
                source_frame = ImageProcessor(source_frame).to_ufloat32().get_image('HWC')
                view_image = np.concatenate((source_frame, merged_frame), 1)
        elif source_type == SourceType.ALIGNED_N_SWAPPED_FACE:
            aligned_face_id = state.aligned_face_id or 0
            aligned_face = None
            swapped_face = None
            for i, fsi in enumerate(bcd.get_face_swap_info_list()):
                if aligned_face_id == i:
                    aligned_face = bcd.get_image(fsi.face_align_image_name)
                    break
            for fsi in bcd.get_face_swap_info_list():
                swapped_face = bcd.get_image(fsi.face_swap_image_name)
                if swapped_face is not None:
                    break
            if aligned_face is not None and swapped_face is not None:
                view_image = np.concatenate((aligned_face, swapped_face), 1)

        return view_image

    def _close_camera(self):
        if self._camera is not None:
            close = getattr(self._camera, 'close', None)
            if callable(close):
                close()
            self._camera = None
            self._camera_cfg = (None, None, None, None)
            self._set_status('@VirtualCameraOutput.status.stopped')

    def _set_status(self, label: str, info_lines: Optional[List[str]] = None):
        cfg = lib_csw.InfoLabel.Config(label=label, info_lines=info_lines or [])
        self.get_control_sheet().status.set_config(cfg)

    @staticmethod
    def _prepare_frame(frame: np.ndarray, target_w: int, target_h: int) -> Optional[np.ndarray]:
        if frame is None:
            return None

        img = frame
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        if img.shape[-1] > 3:
            img = img[..., :3]

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.shape[1] != target_w or img.shape[0] != target_h:
            img = cv2.resize(img, (target_w, target_h))

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class Sheet:
    class Host(lib_csw.Sheet.Host):
        def __init__(self):
            super().__init__()
            self.source_type = lib_csw.DynamicSingleSwitch.Client()
            self.aligned_face_id = lib_csw.Number.Client()
            self.output_enabled = lib_csw.Flag.Client()
            self.device = lib_csw.DynamicSingleSwitch.Client()
            self.width = lib_csw.Number.Client()
            self.height = lib_csw.Number.Client()
            self.fps_limit = lib_csw.Number.Client()
            self.status = lib_csw.InfoLabel.Client()
            self.error = lib_csw.Error.Client()
            self.avg_fps = lib_csw.Number.Client()

    class Worker(lib_csw.Sheet.Worker):
        def __init__(self):
            super().__init__()
            self.source_type = lib_csw.DynamicSingleSwitch.Host()
            self.aligned_face_id = lib_csw.Number.Host()
            self.output_enabled = lib_csw.Flag.Host()
            self.device = lib_csw.DynamicSingleSwitch.Host()
            self.width = lib_csw.Number.Host()
            self.height = lib_csw.Number.Host()
            self.fps_limit = lib_csw.Number.Host()
            self.status = lib_csw.InfoLabel.Host()
            self.error = lib_csw.Error.Host()
            self.avg_fps = lib_csw.Number.Host()


class WorkerState(BackendWorkerState):
    source_type: SourceType = None
    aligned_face_id: int = None
    output_enabled: bool = None
    device_name: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps_limit: Optional[float] = None
