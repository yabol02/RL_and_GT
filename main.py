import os

from neuroevolution.config_exp import ExperimentConfig
from neuroevolution.config_manager import ConfigManager
from neuroevolution.experiment import MLPExperiment


def main():
    file = "config_fb.yaml"

    # Option 1: Load configuration from JSON/YAML file
    config = ExperimentConfig.load_yaml(os.path.join("configurations", file))

    # Option 2: Use predefined configuration in the code
    # from config_manager import PresetConfigs
    # config = PresetConfigs.default()

    # Option 3: Load from configuration manager
    # manager = ConfigManager()
    # manager.load_and_add("default", "config_1.yaml")
    # config = manager.get_config("default")

    experiment = MLPExperiment(config)
    results = experiment.run()

    return results


if __name__ == "__main__":
    main()
