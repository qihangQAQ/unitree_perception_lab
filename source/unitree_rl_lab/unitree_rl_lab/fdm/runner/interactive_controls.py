"""Isaac Sim keyboard subscription and status window for repeated local goals."""

from ..interactive_goals import SpaceRequest


class InteractiveControls:
    def __init__(self):
        import carb.input
        import omni.appwindow
        import omni.ui as ui

        self.request = SpaceRequest()
        self._carb = carb.input
        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Interactive goals require an Isaac Sim application window.")
        self._keyboard = app_window.get_keyboard()
        self._subscription = None
        self._window = ui.Window("FDM Interactive Goals", width=420, height=200)
        with self._window.frame:
            with ui.VStack(spacing=6):
                self._state = ui.Label("Settling...", height=30)
                self._goal = ui.Label("Goal: waiting for initialization", height=25)
                self._distance = ui.Label("Distance: --", height=25)
                self._hint = ui.Label("Focus the viewport. SPACE starts the next goal when waiting.",
                                      word_wrap=True, height=65)
        try:
            # Generic input subscriptions have an ordering parameter. Consume
            # SPACE before Kit shortcuts can use it to toggle the timeline.
            # API: docs.omniverse.nvidia.com/kit/docs/kit-manual/109.0.6/carb.input/carb.input.IInput.html
            self._subscription = self._input.subscribe_to_input_events(self._on_input, order=0)
        except Exception:
            self._window.destroy()
            raise

    def _on_input(self, event, *_):
        if event.deviceType != self._carb.DeviceType.KEYBOARD:
            return True
        key = event.event
        if key.keyboard != self._keyboard:
            return True
        if key.type == self._carb.KeyboardEventType.CHAR:
            return key.input != " "
        if key.input != self._carb.KeyboardInput.SPACE:
            return True
        if key.type == self._carb.KeyboardEventType.KEY_PRESS:
            self.request.key(True)
        elif key.type == self._carb.KeyboardEventType.KEY_RELEASE:
            self.request.key(False)
        # Repeats are consumed but cannot create another request.
        return False

    def set_waiting(self, waiting):
        self.request.enable(waiting)

    def consume_next(self):
        return self.request.consume()

    def update(self, phase, round_number, goal, distance, reason=""):
        labels = {"settling": "Settling", "navigating": "Walking", "waiting": "Waiting for SPACE",
                  "stopped": "Stopped", "paused": "Simulation paused"}
        self._state.text = f"Round {round_number} | {labels[phase]}"
        self._goal.text = "Goal: --" if goal is None else (
            f"Goal: X {goal[0]:.2f}, Y {goal[1]:.2f}, Z {goal[2]:.2f} m")
        self._distance.text = "Distance: --" if distance is None else f"Distance: {distance:.2f} m"
        self._hint.text = reason or "Focus the viewport. SPACE starts the next goal when waiting."

    def close(self):
        self.set_waiting(False)
        if self._subscription is not None:
            self._input.unsubscribe_to_input_events(self._subscription)
            self._subscription = None
        if self._window is not None:
            self._window.destroy()
            self._window = None
