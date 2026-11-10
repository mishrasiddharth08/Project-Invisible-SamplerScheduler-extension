import importlib.util
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

from tqdm import tqdm


spec = importlib.util.spec_from_file_location(
    "pi_console_progress", Path(__file__).parents[1] / "scripts/pi_console_progress.py"
)
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)


class ConsoleProgressTests(TestCase):
    def setUp(self):
        class State:
            job = "generate"
            sampling_step = 0
            sampling_steps = 4

        self.console = StringIO()
        self.state = State()
        self.shared = SimpleNamespace(
            state=self.state, progress_print_out=self.console,
            cmd_opts=SimpleNamespace(disable_console_progressbars=False),
        )
        self.addCleanup(progress._watch(self.shared))

    def test_direct_sampling_and_cancelled_next_image(self):
        for step in (1, 2, 3, 4, 0, 1):
            self.state.sampling_step = step
        self.state.job = ""
        self.assertIn("4/4", self.console.getvalue())
        self.assertIn("1/4", self.console.getvalue().split("4/4")[-1])

    def test_existing_console_bar_suppresses_fallback(self):
        with tqdm(total=4, desc="Native", file=self.console) as native:
            self.state.sampling_step = 1
            native.update()
        self.assertNotIn("Sampling:", self.console.getvalue())

    def test_console_disable_is_respected(self):
        self.shared.cmd_opts.disable_console_progressbars = True
        self.state.sampling_step = 1
        self.assertEqual(self.console.getvalue(), "")
