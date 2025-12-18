import importlib.util
import unittest

_np_spec = importlib.util.find_spec('numpy')
_cv2_spec = importlib.util.find_spec('cv2')

if _np_spec and _cv2_spec:
    import numpy as np
    from apps.DeepFaceLive.backend.VirtualCameraOutput import VirtualCameraOutputWorker
else:
    np = None
    VirtualCameraOutputWorker = None


class VirtualCameraOutputWorkerTest(unittest.TestCase):
    @unittest.skipUnless(_np_spec and _cv2_spec, 'numpy and cv2 are required for virtual camera tests.')
    def test_prepare_frame_resizes_and_converts_to_rgb(self):
        frame = np.array(
            [
                [[0, 64, 128], [0, 64, 128]],
                [[0, 64, 128], [0, 64, 128]],
            ],
            dtype=np.float32,
        )

        prepared = VirtualCameraOutputWorker._prepare_frame(frame, target_w=4, target_h=4)

        self.assertEqual(prepared.shape, (4, 4, 3))
        self.assertEqual(prepared.dtype, np.uint8)
        # Confirm BGR to RGB conversion
        self.assertTrue(np.all(prepared[0, 0] == np.array([128, 64, 0], dtype=np.uint8)))

    @unittest.skipUnless(_np_spec and _cv2_spec, 'numpy and cv2 are required for virtual camera tests.')
    def test_prepare_frame_handles_grayscale(self):
        frame = np.ones((3, 3), dtype=np.uint8) * 255
        prepared = VirtualCameraOutputWorker._prepare_frame(frame, target_w=3, target_h=3)
        self.assertEqual(prepared.shape, (3, 3, 3))
        self.assertTrue(np.all(prepared == 255))


if __name__ == '__main__':
    unittest.main()
