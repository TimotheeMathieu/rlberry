from rlberry.envs import gym_make
from rlberry.manager import plot_writer_data, AgentManager, evaluate_agents
from rlberry.agents.torch import PPOAgent
import gym
from stable_baselines3 import PPO
from rlberry.agents.stable_baselines import StableBaselinesAgent
from gym.wrappers import TimeLimit
import torch


class PPO2(PPO):
    def __init__(self, **kwargs):
        PPO.__init__(self, **kwargs)

    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 1])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

        entr_coef = trial.suggest_loguniform("entr_coef", 1e-8, 0.1)

        eps_clip = trial.suggest_categorical("eps_clip", [0.1, 0.2, 0.3])

        k_epochs = trial.suggest_categorical("k_epochs", [1, 5, 10])

        gae_lambda = trial.suggest_categorical("gae_lambda", [0.94])

        n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024])

        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "entr_coef": entr_coef,
            "eps_clip": eps_clip,
            "k_epochs": k_epochs,
            "gae_lambda":gae_lambda,
            "n_steps":n_steps
        }


env_name = "Acrobot-v1"
if __name__ == "__main__":
    ppo = AgentManager(
        PPOAgent,
        (gym_make, dict(id=env_name)),
        fit_budget=int(1e5),
        eval_kwargs=dict(eval_horizon=500),
        n_fit=4,
        agent_name="RLB_PPO",
    )

    ppo.optimize_hyperparams(
        n_trials=64,
        timeout=None,
        n_fit=4,
        optuna_parallelization="process",
        n_optuna_workers=4,
    )
    ppo.fit()

    evaluate_agents([ppo], n_simulations=50)
