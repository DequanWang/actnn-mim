import time
import warnings

import mmcv
from mmcv.runner import RUNNERS, IterBasedRunner, IterLoader

import actnn
import torch

@RUNNERS.register_module()
class ActnnIterRunner(IterBasedRunner):
    """Actnn Runner.
    This runner train models epoch by epoch.
    """

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1

        def backprop():
            self.optimizer.zero_grad()
            # By the way, self.optimizer is unused and reserved here.
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
            torch.cuda.synchronize()
            outputs['loss'].backward()
        self.controller.iterate(backprop)

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """Start running.
        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, work_dir: %s', work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        actnn_opt_level = 'L2'
        actnn.setup_config(actnn_opt_level, work_dir)
        self.logger.info('ActNN Opt Level: %s', actnn_opt_level)
        self.controller = actnn.controller.Controller(self.model, 100)

        def pack_hook(tensor):  # quantize hook
            return self.controller.quantize(tensor)

        def unpack_hook(tensor):  # dequantize hook
            return self.controller.dequantize(tensor)

        torch._C._autograd._register_saved_tensors_default_hooks(pack_hook, unpack_hook)

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

        torch._C._autograd._reset_saved_tensors_default_hooks()