from src.icmppo.curiosity.icm import ICMFactory
from .models import RIDE


class RIDEFactory(ICMFactory):

    def __init__(self, model_factory, policy_weight, reward_scale, weight,
                 intrinsic_reward_integration, reporter, count_norm, norm_rewards=False):
        super().__init__(model_factory, policy_weight, reward_scale, weight,
                         intrinsic_reward_integration, reporter, norm_rewards=norm_rewards)
        self.count_norm = count_norm

    def create(self, state_converter, action_converter):
        return RIDE(state_converter, action_converter, self.model_factory, self.policy_weight, self.scale, self.weight,
                    self.intrinsic_reward_integration, self.reporter, count_norm=self.count_norm,
                    norm_rewards=self.norm_rewards)
