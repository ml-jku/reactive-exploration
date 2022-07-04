from .models import RND, RNDModel, RNDMlpModel
from .noveld import NovelD


class RNDModelFactory:

    def __init__(self, latent_dim=512, kind="cnn"):
        self.latent_dim = latent_dim
        self.kind = kind

    def create(self, state_converter, action_converter):
        if self.kind == "cnn":
            return RNDModel(state_converter, action_converter, latent_dim=self.latent_dim)
        elif self.kind == "mlp":
            return RNDMlpModel(state_converter, action_converter, latent_dim=self.latent_dim)
        raise NotImplementedError()


class RNDFactory:

    def __init__(self, model_factory, policy_weight, reward_scale, weight, intrinsic_reward_integration,
                 reporter, norm_rewards=False, norm_obs=False):
        self.policy_weight = policy_weight
        self.model_factory = model_factory
        self.scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration
        self.reporter = reporter
        self.norm_rewards = norm_rewards
        self.norm_obs = norm_obs

    def create(self, state_converter, action_converter):
        return RND(state_converter, action_converter, self.model_factory, self.policy_weight, self.scale, self.weight,
                   self.intrinsic_reward_integration, self.reporter, self.norm_rewards, self.norm_obs)


class NoveldDFactory(RNDFactory):

    def __init__(self, model_factory, policy_weight, reward_scale, weight,
                 intrinsic_reward_integration, reporter, alpha, count_norm, norm_rewards=False, norm_obs=False):
        super().__init__(model_factory, policy_weight, reward_scale, weight,
                         intrinsic_reward_integration, reporter, norm_rewards, norm_obs=norm_obs)
        self.alpha = alpha
        self.count_norm = count_norm

    def create(self, state_converter, action_converter):
        return NovelD(state_converter, action_converter, self.model_factory, self.policy_weight, self.scale,
                      self.weight, self.intrinsic_reward_integration, self.reporter, self.alpha,
                      count_norm=self.count_norm, norm_rewards=self.norm_rewards, norm_obs=self.norm_obs)
