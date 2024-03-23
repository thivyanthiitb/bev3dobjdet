from mmcv.runner import HOOKS, Hook

__all__ = ["AdjustReduceBeamsHook"]

@HOOKS.register_module()
class AdjustReduceBeamsHook(Hook):
    def __init__(self, milestones, beam_values):
        assert len(milestones) == len(beam_values), "Milestones and beam values must have the same length"
        self.milestones = milestones
        self.beam_values = beam_values

    def before_train_iter(self, runner):
        current_iter = runner.iter
        for milestone, value in zip(self.milestones, self.beam_values):
            if current_iter < milestone:
                runner.cfg.reduce_beams = value
                break
        else:
            runner.cfg.reduce_beams = self.beam_values[-1]  # Default to the last value if beyond all milestones
