from pathlib import Path
from typing import Dict, List, Optional

from config_exp import ExperimentConfig


class ConfigManager:
    """Managers experiment configurations, allowing to save, load, compare and list them."""

    def __init__(self, config_dir: Path = Path("configs")):
        """
        Initializes the configuration manager.

        :param config_dir: Directory where configurations will be saved and loaded from
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._configs: Dict[str, ExperimentConfig] = {}

    def add_config(self, name: str, config: ExperimentConfig) -> None:
        """
        Adds a configuration to the manager.

        :param name: Name of the configuration
        :param config: `ExperimentConfig` instance to add
        """
        self._configs[name] = config

    def get_config(self, name: str) -> Optional[ExperimentConfig]:
        """
        Gets a configuration by name.

        :param name: Name of the configuration to retrieve
        :return: `ExperimentConfig` instance if found, else None
        """
        return self._configs.get(name)

    def list_configs(self) -> List[str]:
        """Lists all available configurations."""
        return list(self._configs.keys())

    def save_config(self, name: str, format: str = "yaml") -> Path:
        """
        Saves a configuration to disk.

        :param name: Name of the configuration
        :param format: File format ('yaml' or 'json')
        :return: Path to the saved file
        """
        if name not in self._configs:
            raise ValueError(f"Configuration '{name}' not found")

        config = self._configs[name]
        filepath = self.config_dir / f"{name}.{format}"

        if format == "yaml":
            config.save_yaml(filepath)
        elif format == "json":
            config.save_json(filepath)
        else:
            raise ValueError(f"Format '{format}' not supported")

        return filepath

    def load_config(self, filepath: Path | str) -> ExperimentConfig:
        """
        Loads a configuration from a file.

        :param filepath: Path to the configuration file
        :return: Loaded `ExperimentConfig` instance
        """
        filepath = Path(filepath)

        if filepath.suffix == ".yaml" or filepath.suffix == ".yml":
            return ExperimentConfig.load_yaml(filepath)
        elif filepath.suffix == ".json":
            return ExperimentConfig.load_json(filepath)
        else:
            raise ValueError(f"File format '{filepath.suffix}' not supported")

    def load_and_add(self, name: str, filepath: Path | str) -> None:
        """
        Loads a configuration and adds it to the manager.

        :param name: Name to assign to the loaded configuration
        :param filepath: Path to the configuration file
        """
        config = self.load_config(filepath)
        self.add_config(name, config)

    def save_all(self, format: str = "yaml") -> List[Path]:
        """
        Saves all configurations.

        :param format: File format to save ('yaml' or 'json')
        :return: List of paths to the saved files
        """
        saved_paths = []
        for name in self._configs:
            path = self.save_config(name, format)
            saved_paths.append(path)
        return saved_paths

    def compare_configs(self, name1: str, name2: str) -> Dict:
        """
        Compares two configurations and returns a dictionary with the differences.

        :param name1: Name of the first configuration
        :param name2: Name of the second configuration
        :return: Dictionary with the differences found
        """
        if name1 not in self._configs or name2 not in self._configs:
            raise ValueError("One or both configurations do not exist")

        config1_dict = self._configs[name1].to_dict()
        config2_dict = self._configs[name2].to_dict()

        differences = {}
        self._compare_dicts(config1_dict, config2_dict, differences)

        return differences

    def _compare_dicts(
        self, dict1: Dict, dict2: Dict, differences: Dict, prefix: str = ""
    ):
        """
        Recursively compares two dictionaries.

        :param dict1: First dictionary to compare
        :param dict2: Second dictionary to compare
        :param differences: Dictionary to store the differences found
        :param prefix: Current path in the dictionary (used for nested keys)
        """
        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in all_keys:
            current_path = f"{prefix}.{key}" if prefix else key

            if key not in dict1:
                differences[current_path] = {"config1": None, "config2": dict2[key]}
            elif key not in dict2:
                differences[current_path] = {"config1": dict1[key], "config2": None}
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                self._compare_dicts(dict1[key], dict2[key], differences, current_path)
            elif dict1[key] != dict2[key]:
                differences[current_path] = {
                    "config1": dict1[key],
                    "config2": dict2[key],
                }
