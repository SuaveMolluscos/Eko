import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import unittest

import numpy as np
import pygame

from eko.backend import EventType, Key, PygameBackend


class TestPygameBackend(unittest.TestCase):

    def setUp(self):
        self.backend = PygameBackend(capture_mouse=False)
        self.backend.initialize(4, 3, "Eko backend tests")

    def tearDown(self):
        self.backend.shutdown()

    def test_pixel_buffer_is_transposed_and_blitted_as_a_frame(self):
        buffer = np.zeros((3, 4, 3), dtype=np.uint8)
        buffer[0, 0] = (255, 0, 0)
        buffer[2, 3] = (0, 255, 0)

        self.backend.blit_pixel_buffer(buffer)
        pixels = pygame.surfarray.array3d(self.backend._screen)
        np.testing.assert_array_equal(pixels, buffer.transpose(1, 0, 2))
        self.backend.present()

    def test_poll_events_translates_backend_events(self):
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w))
        pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_ESCAPE))
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, rel=(5, -2), pos=(0, 0), buttons=(0, 0, 0)))
        pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=1, y=-2))

        events = self.backend.poll_events()
        self.assertEqual(
            [(event.type, event.key, event.mouse_delta, event.scroll) for event in events],
            [
                (EventType.QUIT, None, (0.0, 0.0), 0.0),
                (EventType.KEY_DOWN, Key.W, (0.0, 0.0), 0.0),
                (EventType.KEY_UP, Key.ESC, (0.0, 0.0), 0.0),
                (EventType.MOUSE_MOVE, None, (5.0, -2.0), 0.0),
                (EventType.SCROLL, None, (0.0, 0.0), -2.0),
            ],
        )

    def test_unknown_keys_and_mouse_capture(self):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_z))
        events = self.backend.poll_events()
        self.assertEqual(events[0].key, Key.UNKNOWN)
        self.backend.set_mouse_capture(True)
        self.backend.set_mouse_capture(False)

    def test_buffer_and_drawing_input_validation(self):
        with self.assertRaises(ValueError):
            self.backend.blit_pixel_buffer(np.zeros((4, 3, 3), dtype=np.uint8))
        with self.assertRaises(TypeError):
            self.backend.blit_pixel_buffer(np.zeros((3, 4, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            self.backend.clear((256, 0, 0))
        with self.assertRaises(ValueError):
            self.backend.draw_line((0, 0), (1, float("inf")), (0, 0, 0))
        self.backend.clear((10, 20, 30))
        self.backend.draw_line((0, 0), (3, 2), (255, 255, 255))

    def test_initialize_validates_dimensions_and_lifecycle(self):
        with self.assertRaises(RuntimeError):
            self.backend.initialize(2, 2, "already initialized")
        with self.assertRaises(ValueError):
            PygameBackend().initialize(0, 2, "invalid")
        self.assertEqual(self.backend.size, (4, 3))


if __name__ == "__main__":
    unittest.main()
