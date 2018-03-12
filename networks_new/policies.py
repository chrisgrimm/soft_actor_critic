from networks_new.policy_mixins import AbstractPolicy, GaussianPolicy, MLPPolicy


class GaussianMLPPolicy(AbstractPolicy, GaussianPolicy, MLPPolicy):
    pass




