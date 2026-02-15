import os
from logging import config

from networkx import config

from experiment_manager.config_exp import ExperimentConfig
from experiment_manager.config_manager import ConfigManager
from experiment_manager.experiment import (
    NeuroevolutionExperiment,
    ReinforcementLearningExperiment,
)


def main():
    file = "config_2.yaml"

    # Option 1: Load configuration from JSON/YAML file
    config = ExperimentConfig.load_yaml(os.path.join("configurations", file))

    # Option 2: Use predefined configuration in the code
    # from config_manager import PresetConfigs
    # config = PresetConfigs.default()

    # Option 3: Load from configuration manager
    # manager = ConfigManager()
    # manager.load_and_add("default", "config_1.yaml")
    # config = manager.get_config("default")

    if config.algorithm in [
        "simple",
        "mu_plus_lambda",
        "mu_comma_lambda",
        "generate_update",
    ]:
        experiment = NeuroevolutionExperiment(config)
    elif config.algorithm in ["q_learning", "sarsa", "monte_carlo"]:
        experiment = ReinforcementLearningExperiment(config)
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    results = experiment.run()
    return results


if __name__ == "__main__":
    main()
