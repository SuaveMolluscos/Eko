import unittest

from eko.backend import EkoEvent, EventType, IRenderer, Key


class TestEkoEvent(unittest.TestCase):

    def test_defaults_and_normalized_mouse_delta(self):
        event = EkoEvent(EventType.MOUSE_MOVE, mouse_delta=(2, -3))
        self.assertIsNone(event.key)
        self.assertEqual(event.mouse_delta, (2.0, -3.0))
        self.assertEqual(event.scroll, 0.0)

    def test_rejects_invalid_event_data(self):
        with self.assertRaises(TypeError):
            EkoEvent("QUIT")
        with self.assertRaises(TypeError):
            EkoEvent(EventType.KEY_DOWN, key="W")
        with self.assertRaises(ValueError):
            EkoEvent(EventType.MOUSE_MOVE, mouse_delta=(1, 2, 3))
        with self.assertRaises(ValueError):
            EkoEvent(EventType.SCROLL, scroll=float("nan"))

    def test_renderer_contract_is_abstract(self):
        with self.assertRaises(TypeError):
            IRenderer()

    def test_event_is_immutable(self):
        event = EkoEvent(EventType.KEY_DOWN, key=Key.W)
        with self.assertRaises(AttributeError):
            event.key = Key.A
        self.assertEqual(EventType.QUIT.name, "QUIT")
        self.assertEqual(Key.ESC.name, "ESC")


if __name__ == "__main__":
    unittest.main()
