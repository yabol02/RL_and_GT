import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from deap import algorithms, tools


class CrossoverMethod(Enum):
    """Crossover methods available in DEAP."""

    ONE_POINT = "one_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    PARTIALLY_MATCHED = "partially_matched"
    UNIFORM_PARTIALLY_MATCHED = "uniform_partially_matched"
    ORDERED = "ordered"
    BLEND = "blend"
    BLEND_ES = "blend_es"
    TWO_POINT_ES = "two_point_es"
    SIMULATED_BINARY = "simulated_binary"
    SIMULATED_BINARY_BOUNDED = "simulated_binary_bounded"
    MESSY_ONE_POINT = "messy_one_point"
    CUSTOM = "custom"  # For user-defined crossover functions


class MutationMethod(Enum):
    """Mutation methods available in DEAP."""

    GAUSSIAN = "gaussian"
    SHUFFLE_INDEXES = "shuffle_indexes"
    FLIP_BIT = "flip_bit"
    POLYNOMIAL_BOUNDED = "polynomial_bounded"
    UNIFORM_INT = "uniform_int"
    LOG_NORMAL_ES = "log_normal_es"
    CUSTOM = "custom"  # For user-defined mutation functions


class SelectionMethod(Enum):
    """Selection methods available in DEAP."""

    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    NSGA2 = "nsga2"
    NSGA3 = "nsga3"
    SPEA2 = "spea2"
    RANDOM = "random"
    BEST = "best"
    WORST = "worst"
    TOURNAMENT_DCD = "tournament_dcd"
    DOUBLE_TOURNAMENT = "double_tournament"
    STOCHASTIC_UNIVERSAL_SAMPLING = "stochastic_universal_sampling"
    LEXICASE = "lexicase"
    EPSILON_LEXICASE = "epsilon_lexicase"
    AUTOMATIC_EPSILON_LEXICASE = "automatic_epsilon_lexicase"
    CUSTOM = "custom"  # For user-defined selection functions


class NeurAlgMethod(Enum):
    """Evolutionary algorithms available in DEAP."""

    SIMPLE = "simple"
    MU_PLUS_LAMBDA = "mu_plus_lambda"
    MU_COMMA_LAMBDA = "mu_comma_lambda"
    GENERATE_UPDATE = "generate_update"
    CUSTOM = "custom"  # For user-defined evolutionary algorithms


class RLAlgMethod(Enum):
    """Classic RL algorithms supported in this project."""

    Q_LEARNING = "q_learning"
    MONTE_CARLO = "monte_carlo"
    SARSA = "sarsa"


# Mapping of enums to DEAP functions
CROSSOVER_FUNCTIONS = {
    CrossoverMethod.ONE_POINT: tools.cxOnePoint,
    CrossoverMethod.TWO_POINT: tools.cxTwoPoint,
    CrossoverMethod.UNIFORM: tools.cxUniform,
    CrossoverMethod.PARTIALLY_MATCHED: tools.cxPartialyMatched,
    CrossoverMethod.UNIFORM_PARTIALLY_MATCHED: tools.cxUniformPartialyMatched,
    CrossoverMethod.ORDERED: tools.cxOrdered,
    CrossoverMethod.BLEND: tools.cxBlend,
    CrossoverMethod.BLEND_ES: tools.cxESBlend,
    CrossoverMethod.TWO_POINT_ES: tools.cxESTwoPoint,
    CrossoverMethod.SIMULATED_BINARY: tools.cxSimulatedBinary,
    CrossoverMethod.SIMULATED_BINARY_BOUNDED: tools.cxSimulatedBinaryBounded,
    CrossoverMethod.MESSY_ONE_POINT: tools.cxMessyOnePoint,
    CrossoverMethod.CUSTOM: None,
}

MUTATION_FUNCTIONS = {
    MutationMethod.GAUSSIAN: tools.mutGaussian,
    MutationMethod.SHUFFLE_INDEXES: tools.mutShuffleIndexes,
    MutationMethod.FLIP_BIT: tools.mutFlipBit,
    MutationMethod.POLYNOMIAL_BOUNDED: tools.mutPolynomialBounded,
    MutationMethod.UNIFORM_INT: tools.mutUniformInt,
    MutationMethod.LOG_NORMAL_ES: tools.mutESLogNormal,
    MutationMethod.CUSTOM: None,
}

SELECTION_FUNCTIONS = {
    SelectionMethod.TOURNAMENT: tools.selTournament,
    SelectionMethod.ROULETTE: tools.selRoulette,
    SelectionMethod.NSGA2: tools.selNSGA2,
    SelectionMethod.NSGA3: tools.selNSGA3,
    SelectionMethod.SPEA2: tools.selSPEA2,
    SelectionMethod.RANDOM: tools.selRandom,
    SelectionMethod.BEST: tools.selBest,
    SelectionMethod.WORST: tools.selWorst,
    SelectionMethod.TOURNAMENT_DCD: tools.selTournamentDCD,
    SelectionMethod.DOUBLE_TOURNAMENT: tools.selDoubleTournament,
    SelectionMethod.STOCHASTIC_UNIVERSAL_SAMPLING: tools.selStochasticUniversalSampling,
    SelectionMethod.LEXICASE: tools.selLexicase,
    SelectionMethod.EPSILON_LEXICASE: tools.selEpsilonLexicase,
    SelectionMethod.AUTOMATIC_EPSILON_LEXICASE: tools.selAutomaticEpsilonLexicase,
    SelectionMethod.CUSTOM: None,
}

ALGORITHM_FUNCTIONS = {
    NeurAlgMethod.SIMPLE: algorithms.eaSimple,
    NeurAlgMethod.MU_PLUS_LAMBDA: algorithms.eaMuPlusLambda,
    NeurAlgMethod.MU_COMMA_LAMBDA: algorithms.eaMuCommaLambda,
    NeurAlgMethod.GENERATE_UPDATE: algorithms.eaGenerateUpdate,
    NeurAlgMethod.CUSTOM: None,
}


@dataclass
class CrossoverConfig:
    """Configuration for crossover operations"""

    method: CrossoverMethod = CrossoverMethod.BLEND
    probability: float = 0.75

    # Specific parameters for certain crossover methods
    alpha: Optional[float] = None  # For cxBlend or cxESBlend
    indpb: Optional[float] = None  # For cxUniform or cxUniformPartiallyMatched
    eta: Optional[float] = None  # For cxSimulatedBinary or cxSimulatedBinaryBounded
    low: Optional[float] = None  # For cxSimulatedBinaryBounded
    up: Optional[float] = None  # For cxSimulatedBinaryBounded
    others: Dict[str, Any] = (
        None  # For any additional parameters for custom crossover methods
    )

    def get_params(self) -> Dict[str, Any]:
        """Obtains the specific parameters for the selected crossover method."""
        params = {}
        if self.method == CrossoverMethod.BLEND:
            params["alpha"] = self.alpha or 0.5
        elif self.method == CrossoverMethod.UNIFORM:
            params["indpb"] = self.indpb or 0.5
        elif self.method == CrossoverMethod.UNIFORM_PARTIALLY_MATCHED:
            params["indpb"] = self.indpb or 0.5
        elif self.method == CrossoverMethod.BLEND_ES:
            params["alpha"] = self.alpha or 0.5
        elif self.method == CrossoverMethod.SIMULATED_BINARY:
            params["eta"] = self.eta or 20.0
        elif self.method == CrossoverMethod.SIMULATED_BINARY_BOUNDED:
            params["eta"] = self.eta or 20.0
            params["low"] = self.low or 0.0
            params["up"] = self.up or 1.0
        elif self.method == CrossoverMethod.CUSTOM and self.others is not None:
            params.update(self.others)
        return params


@dataclass
class MutationConfig:
    """Configuration for mutation operations"""

    method: MutationMethod = MutationMethod.GAUSSIAN
    probability: float = 0.25

    # Specific parameters for certain mutation methods
    mu: float = 0.0  # For mutGaussian
    sigma: float = 1.0  # For mutGaussian
    indpb: float = 0.1  # Independent probability per gene
    low: Optional[float] = None  # For mutPolynomialBounded, mutUniformInt
    up: Optional[float] = None  # For mutPolynomialBounded, mutUniformInt
    eta: Optional[float] = None  # For mutPolynomialBounded
    others: Dict[str, Any] = (
        None  # For any additional parameters for custom mutation methods
    )

    def get_params(self) -> Dict[str, Any]:
        """Returns the specific parameters for the selected mutation method."""
        params = {}
        if self.method == MutationMethod.GAUSSIAN:
            params["mu"] = self.mu
            params["sigma"] = self.sigma
            params["indpb"] = self.indpb
        elif self.method == MutationMethod.SHUFFLE_INDEXES:
            params["indpb"] = self.indpb
        elif self.method == MutationMethod.FLIP_BIT:
            params["indpb"] = self.indpb
        elif self.method == MutationMethod.POLYNOMIAL_BOUNDED:
            params["eta"] = self.eta or 20.0
            params["low"] = self.low or 0.0
            params["up"] = self.up or 1.0
            params["indpb"] = self.indpb
        elif self.method == MutationMethod.UNIFORM_INT:
            params["low"] = self.low or 0
            params["up"] = self.up or 100
            params["indpb"] = self.indpb
        elif self.method == MutationMethod.LOG_NORMAL_ES:
            params["eta"] = self.eta or 20.0
        elif self.method == MutationMethod.CUSTOM and self.others is not None:
            params.update(self.others)
        return params


@dataclass
class SelectionConfig:
    """Configuration for selection operations"""

    method: SelectionMethod = SelectionMethod.TOURNAMENT

    tournsize: int = 4  # For selTournament
    fitness_size: int = 7  # For selDoubleTournament,
    parsimony_size: int = 7  # For selDoubleTournament,
    fitness_first: bool = True  # For selDoubleTournament
    nsga_params: Dict[str, Any] = (
        None  # For NSGA2 and NSGA3 specific parameters (search in official documentation for details)
    )
    epsilon: Optional[float] = None  # For selEpsilonLexicase
    custom: Dict[str, Any] = (
        None  # For any additional parameters for custom selection methods
    )

    def get_params(self) -> Dict[str, Any]:
        """Returns the specific parameters for the selected selection method."""
        params = {}
        if self.method == SelectionMethod.TOURNAMENT:
            params["tournsize"] = self.tournsize
        elif self.method == SelectionMethod.DOUBLE_TOURNAMENT:
            params["fitness_size"] = self.fitness_size
            params["parsimony_size"] = self.parsimony_size
            params["fitness_first"] = self.fitness_first
        elif (
            self.method in {SelectionMethod.NSGA2, SelectionMethod.NSGA3}
            and self.nsga_params is not None
        ):
            params.update(self.nsga_params)
        elif self.method == SelectionMethod.EPSILON_LEXICASE:
            params["epsilon"] = self.epsilon or 0.01
        elif self.method == SelectionMethod.CUSTOM and self.custom is not None:
            params.update(self.custom)
        return params


@dataclass
class AlgorithmConfig:
    """Base configuration for algorithms classes"""

    def _to_serializable_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclass-specific serialization."""
        return config_dict

    @classmethod
    def _from_serializable_dict(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for subclass-specific deserialization."""
        return config_dict

    def to_dict(self) -> Dict[str, Any]:
        """Converts the algorithm configuration to a dictionary."""
        return self._to_serializable_dict(asdict(self))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AlgorithmConfig":
        """Creates an instance of the algorithm configuration from a dictionary."""
        normalized_config = cls._from_serializable_dict(deepcopy(config_dict))
        return cls(**normalized_config)

    def save_json(self, filepath: Path | str) -> None:
        """Saves the configuration to a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, filepath: Path | str) -> None:
        """Saves the configuration to a YAML file."""
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load_json(cls, filepath: Path | str) -> "AlgorithmConfig":
        """Loads the configuration from a JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def load_yaml(cls, filepath: Path | str) -> "AlgorithmConfig":
        """Loads the configuration from a YAML file."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


@dataclass
class NeurEvConfig(AlgorithmConfig):
    """Full configuration for running an evolutionary algorithm experiment."""

    population_size: int = 100
    num_generations: int = 300
    algorithm: NeurAlgMethod = NeurAlgMethod.MU_COMMA_LAMBDA
    crossover: CrossoverConfig = field(default_factory=CrossoverConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    mu: Optional[int] = None  # For eaMuPlusLambda, eaMuCommaLambda
    lambda_: Optional[int] = None  # For eaMuPlusLambda, eaMuCommaLambda

    # Elite
    elite_size: int = 10

    # Evaluation
    num_eval_episodes: int = 5

    # Others
    random_seed: Optional[int] = None
    n_workers: Optional[int] = None
    verbose: bool = True

    def __post_init__(self):
        """Sets default values for mu and lambda_ if not provided."""
        if self.mu is None:
            self.mu = self.population_size
        if self.lambda_ is None:
            self.lambda_ = self.population_size * 2

    def _to_serializable_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Converts enum fields to serializable values."""
        config_dict["algorithm"] = self.algorithm.value
        config_dict["crossover"]["method"] = self.crossover.method.value
        config_dict["mutation"]["method"] = self.mutation.method.value
        config_dict["selection"]["method"] = self.selection.method.value

        return config_dict

    @classmethod
    def _from_serializable_dict(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Converts serializable values to enum/dataclass fields."""
        if "algorithm" in config_dict:
            config_dict["algorithm"] = NeurAlgMethod(config_dict["algorithm"])

        if "crossover" in config_dict:
            crossover_dict = config_dict["crossover"]
            if "method" in crossover_dict:
                crossover_dict["method"] = CrossoverMethod(crossover_dict["method"])
            config_dict["crossover"] = CrossoverConfig(**crossover_dict)

        if "mutation" in config_dict:
            mutation_dict = config_dict["mutation"]
            if "method" in mutation_dict:
                mutation_dict["method"] = MutationMethod(mutation_dict["method"])
            config_dict["mutation"] = MutationConfig(**mutation_dict)

        if "selection" in config_dict:
            selection_dict = config_dict["selection"]
            if "method" in selection_dict:
                selection_dict["method"] = SelectionMethod(selection_dict["method"])
            config_dict["selection"] = SelectionConfig(**selection_dict)

        return config_dict

    def get_algorithm_params(self) -> Dict[str, Any]:
        """Returns the necessary parameters for the evolutionary algorithm."""
        params = {
            "cxpb": self.crossover.probability,
            "mutpb": self.mutation.probability,
            "ngen": self.num_generations,
            "verbose": self.verbose,
        }

        if self.algorithm in [
            NeurAlgMethod.MU_PLUS_LAMBDA,
            NeurAlgMethod.MU_COMMA_LAMBDA,
        ]:
            params["mu"] = self.mu
            params["lambda_"] = self.lambda_

        return params


@dataclass
class RLConfig(AlgorithmConfig):
    """Base configuration for classic tabular RL experiments."""

    algorithm: RLAlgMethod = RLAlgMethod.Q_LEARNING
    n_training_episodes: int = 10_000
    max_steps: int = 99
    learning_rate: float = 0.7
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.0005
    n_eval_episodes: int = 100
    n_bins: int = 10  # For discretizing continuous state spaces
    use_first_visit: bool = True  # Only used when algorithm == monte_carlo
    random_seed: Optional[int] = None

    def _to_serializable_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Converts enum fields to serializable values."""
        config_dict["algorithm"] = self.algorithm.value
        config_dict["random_seed"] = self.random_seed
        return config_dict

    @classmethod
    def _from_serializable_dict(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Converts serializable values to enum fields."""
        if "algorithm" in config_dict:
            config_dict["algorithm"] = RLAlgMethod(config_dict["algorithm"])
        return config_dict


@dataclass
class ExperimentConfig:
    """Full configuration for an evolutionary algorithm experiment, including algorithm parameters and environment settings."""

    environment_name: str
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    labels: Optional[List[List[str]]] = None
    architecture: List[int] = field(default_factory=lambda: [8, 6, 4])
    algorithm: AlgorithmConfig = field(default_factory=NeurEvConfig)
    success_threshold: Optional[float] = None
    experiments_dir: Path = Path("results")

    def to_dict(self) -> Dict[str, Any]:
        """Converts the full configuration to a dictionary."""
        return {
            "environment_name": self.environment_name,
            "labels": self.labels,
            "env_kwargs": self.env_kwargs,
            "architecture": self.architecture,
            "algorithm": self.algorithm.to_dict(),
            "success_threshold": self.success_threshold,
            "experiments_dir": str(self.experiments_dir),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """
        Creates an instance from a dictionary.

        :param config_dict: A dictionary containing the configuration parameters. It should have the same structure as the one produced by `to_dict()`.
        :return: An instance of `ExperimentConfig` with the parameters set according to the provided dictionary.
        """
        if "algorithm" in config_dict:
            algorithm_config = config_dict["algorithm"]
            if not isinstance(algorithm_config, dict):
                raise ValueError("'algorithm' configuration must be a dictionary")

            algorithm_name = algorithm_config.get("algorithm")
            if algorithm_name is None:
                raise ValueError(
                    "'algorithm.algorithm' is required to detect configuration type"
                )

            neur_ev_algorithms = {method.value for method in NeurAlgMethod}
            rl_algorithms = {method.value for method in RLAlgMethod}

            if algorithm_name in neur_ev_algorithms:
                config_dict["algorithm"] = NeurEvConfig.from_dict(algorithm_config)
            elif algorithm_name in rl_algorithms:
                config_dict["algorithm"] = RLConfig.from_dict(algorithm_config)
            else:
                valid_algorithms = sorted(neur_ev_algorithms | rl_algorithms)
                raise ValueError(
                    f"Unsupported algorithm '{algorithm_name}'. Supported values are: {valid_algorithms}"
                )
        if "experiments_dir" in config_dict:
            config_dict["experiments_dir"] = Path(config_dict["experiments_dir"])
        return cls(**config_dict)

    def save_json(self, filepath: Path | str) -> None:
        """Saves the configuration in JSON format."""
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, filepath: Path | str) -> None:
        """Saves the configuration in YAML format."""
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load_json(cls, filepath: Path | str) -> "ExperimentConfig":
        """Loads the configuration from JSON."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def load_yaml(cls, filepath: Path | str) -> "ExperimentConfig":
        """Loads the configuration from YAML."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
