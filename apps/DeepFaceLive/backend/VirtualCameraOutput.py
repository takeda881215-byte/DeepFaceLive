import importlib.util
import platform
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from xlib import time as lib_time
from xlib.image import ImageProcessor
from xlib.mp import csw as lib_csw

from .BackendBase import (
    BackendConnection,
    BackendDB,
    BackendHost,
    BackendSignal,
    BackendWeakHeap,
    BackendWorker,
    BackendWorkerState,
)
from .StreamOutput import SourceType, ViewModeNames


# =========================
# 常量：统一在这里控制
# =========================
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30.0


class VirtualCameraOutput(BackendHost):
    """
    Pushes frames into a virtual camera device.
    """

    def __init__(
        self,
        weak_heap: BackendWeakHeap,
        reemit_frame_signal: BackendSignal,
        bc_in: BackendConnection,
        bc_out: BackendConnection,
        backend_db: BackendDB = None,
    ):
        super().__init__(
            backend_db=backend_db,
            sheet_cls=Sheet,
            worker_cls=VirtualCameraOutputWorker,
            worker_state_cls=WorkerState,
            worker_start_args=[weak_heap, reemit_frame_signal, bc_in, bc_out],
        )

    def get_control_sheet(self) -> "Sheet.Host":
        return super().get_control_sheet()


class VirtualCameraOutputWorker(BackendWorker):
    def get_state(self) -> "WorkerState":
        return super().get_state()

    def get_control_sheet(self) -> "Sheet.Worker":
        return super().get_control_sheet()

    # -------------------------
    # 生命周期
    # -------------------------
    def on_start(
        self,
        weak_heap: BackendWeakHeap,
        reemit_frame_signal: BackendSignal,
        bc_in: BackendConnection,
        bc_out: BackendConnection,
    ):
        self.weak_heap = weak_heap
        self.reemit_frame_signal = reemit_frame_signal
        self.bc_in = bc_in
        self.bc_out = bc_out

        self._pyvirtualcam = None
        self._pyvirtualcam_checked = False
        self._camera = None
        self._camera_cfg = (None, None, None, None)

        self._frame_queue = deque(maxlen=3)
        self._fps_counter = lib_time.FPSCounter()
        self._pending_bcd = None

        state, cs = self.get_state(), self.get_control_sheet()

        # ---------- UI 绑定 ----------
        cs.source_type.call_on_selected(self.on_cs_source_type)
        cs.aligned_face_id.call_on_number(self.on_cs_aligned_face_id)
        cs.output_enabled.call_on_flag(self.on_cs_output_enabled)
        cs.device.call_on_selected(self.on_cs_device_selected)
        cs.width.call_on_number(self.on_cs_width)
        cs.height.call_on_number(self.on_cs_height)
        cs.fps_limit.call_on_number(self.on_cs_fps_limit)

        # ---------- UI 初始化 ----------
        cs.status.enable()
        cs.error.enable()

        cs.avg_fps.enable()
        cs.avg_fps.set_config(
            lib_csw.Number.Config(min=0, max=240, decimals=1, read_only=True)
        )
        cs.avg_fps.set_number(0)

        cs.source_type.enable()
        cs.source_type.set_choices(
            SourceType,
            ViewModeNames,
            none_choice_name="@misc.menu_select",
        )
        cs.source_type.select(
            state.source_type
            if state.source_type is not None
            else SourceType.MERGED_FRAME_OR_SOURCE_FRAME
        )

        cs.aligned_face_id.disable()

        cs.output_enabled.enable()
        cs.output_enabled.set_flag(state.output_enabled or False)

        cs.device.enable()
        self._refresh_devices()
        if state.device_name:
            cs.device.select(state.device_name)

        # ---------- 分辨率 ----------
        cs.width.enable()
        cs.height.enable()
        cs.width.set_config(
            lib_csw.Number.Config(
                min=64, max=7680, step=2, decimals=0, zero_is_auto=False
            )
        )
        cs.height.set_config(
            lib_csw.Number.Config(
                min=64, max=4320, step=2, decimals=0, zero_is_auto=False
            )
        )

        cs.width.set_number(state.width or DEFAULT_WIDTH)
        cs.height.set_number(state.height or DEFAULT_HEIGHT)

        cs.fps_limit.enable()
        cs.fps_limit.set_config(
            lib_csw.Number.Config(min=1, max=240, decimals=2, zero_is_auto=False)
        )
        cs.fps_limit.set_number(state.fps_limit or DEFAULT_FPS)

        self._set_status("@VirtualCameraOutput.status.stopped")
        cs.error.set_error(None)

    def on_stop(self):
        self._close_camera()

    # -------------------------
    # UI 回调
    # -------------------------
    def on_cs_source_type(self, idx, source_type):
        state, cs = self.get_state(), self.get_control_sheet()
        if source_type in (
            SourceType.ALIGNED_FACE,
            SourceType.ALIGNED_N_SWAPPED_FACE,
        ):
            cs.aligned_face_id.enable()
            cs.aligned_face_id.set_config(
                lib_csw.Number.Config(min=0, max=16, step=1)
            )
            cs.aligned_face_id.set_number(state.aligned_face_id or 0)
        else:
            cs.aligned_face_id.disable()

        state.source_type = source_type
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_aligned_face_id(self, v):
        state, cs = self.get_state(), self.get_control_sheet()
        cfg = cs.aligned_face_id.get_config()
        v = int(np.clip(int(v), cfg.min, cfg.max))
        state.aligned_face_id = v
        cs.aligned_face_id.set_number(v)
        self.save_state()
        self.reemit_frame_signal.send()

    def on_cs_output_enabled(self, enabled):
        state = self.get_state()
        state.output_enabled = bool(enabled)
        self.save_state()
        if not enabled:
            self._frame_queue.clear()
            self._close_camera()
            self._set_status("@VirtualCameraOutput.status.disabled")

    def on_cs_device_selected(self, idx, name):
        state = self.get_state()
        state.device_name = name
        self.save_state()
        self._close_camera()

    def on_cs_width(self, w):
        state = self.get_state()
        state.width = self._sanitize_int(w, DEFAULT_WIDTH)
        self.save_state()
        self._close_camera()

    def on_cs_height(self, h):
        state = self.get_state()
        state.height = self._sanitize_int(h, DEFAULT_HEIGHT)
        self.save_state()
        self._close_camera()

    def on_cs_fps_limit(self, fps):
        state = self.get_state()
        state.fps_limit = float(max(1.0, min(240.0, float(fps))))
        self.save_state()
        self._close_camera()

    # -------------------------
    # 主循环
    # -------------------------
    def on_tick(self):
        bcd = self.bc_in.read(timeout=0.005)
        if bcd:
            bcd.assign_weak_heap(self.weak_heap)
            frame = self._extract_frame(bcd, self.get_state())
            if frame is not None and self.get_state().output_enabled:
                self._frame_queue.append(frame)
            self._pending_bcd = bcd

        self._drain_queue()
        self._forward_pending_bcd()

    # -------------------------
    # 设备枚举（不碰 frame queue）
    # -------------------------
    def _refresh_devices(self):
        """
        Refresh available virtual camera devices.
        This MUST NOT touch frame queue.
        """
        cs, state = self.get_control_sheet(), self.get_state()

        device_names = []

        # 1) pyvirtualcam 枚举
        if self._ensure_pyvirtualcam():
            list_cameras = getattr(self._pyvirtualcam, "list_cameras", None)
            if callable(list_cameras):
                try:
                    device_names = list(list_cameras() or [])
                except Exception as e:
                    cs.error.set_error(f"pyvirtualcam.list_cameras() failed: {e}")
                    device_names = []

        # 2) Linux fallback
        if platform.system() == "Linux" and not device_names:
            device_names = self._linux_video_devices_filtered()

        cs.device.set_choices(device_names, none_choice_name="@misc.menu_select")

        # 列表为空时不清空已选，避免短暂失败导致丢配置
        if device_names and (state.device_name not in device_names):
            state.device_name = None
            self.save_state()

    def _linux_video_devices_filtered(self):
        """
        Linux: return list of /dev/video* that can be opened.
        """
        out = []
        for p in sorted(Path("/dev").glob("video*")):
            dev = str(p)
            cap = None
            try:
                cap = cv2.VideoCapture(dev)
                if cap is not None and cap.isOpened():
                    out.append(dev)
            except Exception:
                pass
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
        return out

    # -------------------------
    # 帧提取（兜底版：不会崩）
    # -------------------------
    def _extract_frame(self, bcd, state):
        """
        Best-effort frame extraction.
        Returns BGR uint8 image or None.
        """

        def _get(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        st = getattr(state, "source_type", None)
        candidates = []

        if st == SourceType.MERGED_FRAME_OR_SOURCE_FRAME:
            candidates += ["merged_frame", "frame", "src_frame", "source_frame"]
        elif st == SourceType.ALIGNED_FACE:
            candidates += ["aligned_face", "aligned_face_frame", "face_aligned"]
        elif st == SourceType.ALIGNED_N_SWAPPED_FACE:
            candidates += ["swapped_face", "aligned_swapped_face", "face_swapped"]
        else:
            candidates += ["merged_frame", "frame", "src_frame", "source_frame"]

        candidates += ["image", "img", "bgr", "rgb"]

        img = None
        for k in candidates:
            v = _get(bcd, k, None)
            if v is not None:
                img = v
                break

        if img is None:
            return None

        if isinstance(img, ImageProcessor):
            try:
                img = img.get_image()
            except Exception:
                try:
                    img = img.image
                except Exception:
                    return None

        if not isinstance(img, np.ndarray):
            try:
                img = np.asarray(img)
            except Exception:
                return None

        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        if img.ndim != 3 or img.shape[2] < 3:
            return None

        img = img[..., :3]

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    # -------------------------
    # 输出核心
    # -------------------------
    def _drain_queue(self):
        state, cs = self.get_state(), self.get_control_sheet()
        if not state.output_enabled or not self._ensure_pyvirtualcam():
            return

        if not self._frame_queue:
            return

        frame = self._frame_queue.pop()
        self._frame_queue.clear()

        w, h = self._resolve_resolution(state)
        fps = self._resolve_fps(state)

        if not self._ensure_camera(state.device_name, w, h, fps):
            return

        frame = self._prepare_frame(frame, w, h)
        self._camera.send(frame)
        self._camera.sleep_until_next_frame()

        cs.avg_fps.set_number(self._fps_counter.step())
        self._set_status(
            "@VirtualCameraOutput.status.running",
            [f"{w}x{h} @ {fps:.2f} FPS", state.device_name or "default"],
        )

    # -------------------------
    # 工具方法
    # -------------------------
    def _resolve_resolution(self, state):
        w = state.width or DEFAULT_WIDTH
        h = state.height or DEFAULT_HEIGHT
        w = int(np.clip(w, 64, 7680)) // 2 * 2
        h = int(np.clip(h, 64, 4320)) // 2 * 2
        return w, h

    def _resolve_fps(self, state):
        return float(state.fps_limit or DEFAULT_FPS)

    def _sanitize_int(self, v, default):
        try:
            v = int(v)
            return v if v > 0 else default
        except Exception:
            return default

    def _ensure_pyvirtualcam(self):
        if self._pyvirtualcam_checked:
            return self._pyvirtualcam is not None
        self._pyvirtualcam_checked = True

        spec = importlib.util.find_spec("pyvirtualcam")
        if not spec:
            self.get_control_sheet().error.set_error("pyvirtualcam not installed")
            return False

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            self.get_control_sheet().error.set_error(f"pyvirtualcam import failed: {e}")
            return False

        self._pyvirtualcam = module
        return True

    def _ensure_camera(self, device, w, h, fps):
        cs = self.get_control_sheet()

        if self._camera:
            cfg = self._camera_cfg
            if cfg == (device, w, h, fps):
                return True
            self._close_camera()

        try:
            self._camera = self._pyvirtualcam.Camera(
                width=w, height=h, fps=fps, device=device
            )
            self._camera_cfg = (device, w, h, fps)
            cs.error.set_error(None)
            return True
        except Exception as e:
            cs.error.set_error(f"Virtual cam open failed: {e}")
            self._camera = None
            self._camera_cfg = (None, None, None, None)
            return False

    def _close_camera(self):
        if self._camera:
            try:
                self._camera.close()
            except Exception:
                pass
            self._camera = None
            self._camera_cfg = (None, None, None, None)

    def _forward_pending_bcd(self):
        if self._pending_bcd and self.bc_out and self.bc_out.is_full_read(1):
            self.bc_out.write(self._pending_bcd)
            self._pending_bcd = None

    def _set_status(self, label, info_lines=None):
        cfg = lib_csw.InfoLabel.Config(label=label, info_lines=info_lines or [])
        self.get_control_sheet().status.set_config(cfg)

    @staticmethod
    def _prepare_frame(frame, w, h):
        img = frame
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =========================
# UI Sheet
# =========================
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
    output_enabled: bool = False
    device_name: Optional[str] = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps_limit: float = DEFAULT_FPS
