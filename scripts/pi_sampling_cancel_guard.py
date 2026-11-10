"""Keep Forge's native output finalization when sampling stops before a latent."""
import ast
import inspect
import logging
import textwrap
from functools import wraps

log = logging.getLogger('PI-SamplingCancel')
_installed = []


def _insert_after(source, anchor, lines):
    matches = [line for line in source.splitlines() if line.strip().startswith(anchor)]
    if len(matches) != 1:
        raise ValueError(f'Forge sampling anchor changed: {anchor}')
    line = matches[0]
    indent = line[:len(line) - len(line.lstrip())]
    return source.replace(line, line + '\n' + '\n'.join(indent + added for added in lines), 1)


def _inner_source(source):
    start = source.index('            samples_ddim = p.sample(')
    end = source.index('            if p.scripts is not None:', start)
    block = source[start:end]
    lines = [line for line in block.splitlines() if line.strip()]
    if len(lines) != 5 or not all(needle in block for needle in (
        'for x_sample in samples_ddim:', 'p.latents_after_sampling.append(x_sample)',
        'if sigmas_backup is not None:', 'predictor.set_sigmas(sigmas_backup)')):
        raise ValueError('Forge sample collection changed')
    indent = ' ' * 12
    replacement = [
        'try:',
        '    ' + lines[0].strip(),
        'finally:',
        '    ' + lines[3].strip(),
        '        ' + lines[4].strip(),
        'if samples_ddim is None:',
        '    if state.interrupted or state.stopping_generation:',
        '        break',
        '    if state.skipped:',
        '        state.nextjob()',
        '        continue',
        '    raise RuntimeError("The sampler produced no latent. Check the preceding sampler log.")',
        lines[1].strip(),
        '    ' + lines[2].strip(),
    ]
    source = source[:start] + '\n'.join(indent + line for line in replacement) + '\n\n' + source[end:]
    grid = 'if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid'
    if source.count(grid) != 1:
        raise ValueError('Forge grid finalization changed')
    return source.replace(grid, 'if output_images and (opts.return_grid or opts.grid_save) and not p.do_not_save_grid')


def _compile_native(function, transform):
    original = inspect.unwrap(function)
    source, start = inspect.getsourcelines(original)
    tree = ast.parse(transform(textwrap.dedent(''.join(source))))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Forge sampling function structure changed')
    node = tree.body[0]
    node.decorator_list = []
    # Compile only the function's body. Do not re-evaluate defaults, annotations,
    # decorators or any module-level code while installing the compatibility guard.
    ast.increment_lineno(tree, start - 1)
    compiled = compile(tree, original.__code__.co_filename, 'exec')
    replacement = next(c for c in compiled.co_consts
                       if inspect.iscode(c) and c.co_name == original.__name__)
    previous = original.__code__
    fields = ('co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_freevars', 'co_cellvars')
    if any(getattr(previous, key) != getattr(replacement, key) for key in fields):
        raise ValueError('Forge sampling function signature changed')
    return original, previous, replacement


def install(*_):
    if _installed:
        return
    try:
        from modules import processing, sd_samplers_common, shared
        edits = [
            _compile_native(processing.process_images_inner, _inner_source),
            _compile_native(processing.StableDiffusionProcessingTxt2Img.sample,
                lambda s: _insert_after(s, 'samples = self.sampler.sample(', ['if samples is None:', '    return None'])),
            _compile_native(processing.StableDiffusionProcessingTxt2Img.sample_hr_pass,
                lambda s: _insert_after(s, 'samples = self.sampler.sample_img2img(', [
                    'if samples is None:', '    self.sampler = None', '    self.is_hr_pass = False', '    return None'])),
            _compile_native(processing.StableDiffusionProcessingImg2Img.sample,
                lambda s: _insert_after(s, 'samples = self.sampler.sample_img2img(', ['if samples is None:', '    return None'])),
        ]
        owner = sd_samplers_common.Sampler
        original = owner.launch_sampling
        inspect.signature(original).bind(object(), 1, lambda: None)

        @wraps(original)
        def launch(self, steps, func):
            # The fallback must not reuse a previous batch or Hires pass's latent.
            shared.state.current_latent = None
            return original(self, steps, func)

        for function, old, new in edits:
            function.__code__ = new
            _installed.append((function, '__code__', old, new))
        owner.launch_sampling = launch
        _installed.append((owner, 'launch_sampling', original, launch))
        log.info('Early sampling cancellation guard active.')
    except (AttributeError, ImportError, OSError, SyntaxError, TypeError, ValueError, StopIteration) as error:
        uninstall()
        log.warning('Early sampling cancellation guard inactive: %s', error)


def uninstall(*_):
    while _installed:
        owner, name, old, new = _installed.pop()
        if getattr(owner, name) is new:
            setattr(owner, name, old)


try:
    from modules import script_callbacks
except ImportError:
    pass
else:
    script_callbacks.on_app_started(install)
    script_callbacks.on_script_unloaded(uninstall)
