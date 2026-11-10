"""Show Forge's reported sampling steps when an engine has no console bar."""

import logging

from tqdm import tqdm


log = logging.getLogger(__name__)
_uninstall = None


def _watch(shared):
    state_class = type(shared.state)
    original = state_class.__setattr__
    had_setter = "__setattr__" in vars(state_class)
    bar = None

    def close():
        nonlocal bar
        if bar is not None:
            active, bar = bar, None
            active.close()

    def update(state, name, previous, value):
        nonlocal bar
        if name == "job":
            if previous != value:
                close()
            return
        if value <= 0 or value < previous:
            close()
        if (value <= 0 or value == previous or not state.job
                or shared.cmd_opts.disable_console_progressbars):
            return
        total = state.sampling_steps
        if total <= 0:
            return

        # Native samplers, Forge's total bar, and direct engines such as
        # FLUX.3 may already own a visible tqdm bar for this job.
        with tqdm.get_lock():
            other_bar = any(
                active is not bar and not getattr(active, "disable", True)
                for active in tuple(getattr(tqdm, "_instances", ()))
            )
        if other_bar:
            close()
            return
        if bar is None:
            bar = tqdm(total=total, desc="Sampling", file=shared.progress_print_out,
                       dynamic_ncols=True)
        bar.total = total
        bar.update(min(value, total) - bar.n)
        if value >= total:
            close()

    def set_attribute(state, name, value):
        watch = name in ("sampling_step", "job") and state is shared.state
        previous = getattr(state, name, None) if watch else None
        original(state, name, value)
        if watch:
            try:
                update(state, name, previous, value)
            except (OSError, ValueError):
                log.debug("Could not write sampling progress to the console", exc_info=True)

    def restore():
        if state_class.__setattr__ is set_attribute:
            if had_setter:
                state_class.__setattr__ = original
            else:
                del state_class.__setattr__
        close()

    state_class.__setattr__ = set_attribute
    return restore


def install(*_):
    global _uninstall
    if _uninstall is None:
        from modules import shared
        _uninstall = _watch(shared)


def uninstall():
    global _uninstall
    if _uninstall is not None:
        restore, _uninstall = _uninstall, None
        restore()


try:
    from modules import script_callbacks
except ImportError:
    pass
else:
    script_callbacks.on_app_started(install)
    script_callbacks.on_script_unloaded(uninstall)
