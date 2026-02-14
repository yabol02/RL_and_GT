# multilayer perceptron of any architecture
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


class MLP:
    """
    Class to define Multilayer Perceptrons.
    Declare instance with MLP(layers).
    """

    def __init__(
        self, layers: list[int], labels: tuple[list[str], list[str]] | None = None
    ):
        """
        :param layers: a tuple with (ninputs, nhidden1, nhidden2, ... noutput)
        :param labels: a tuple with (input_labels, output_labels) to show in the graph
        """
        self.layers = layers
        self.trace = False
        self.threshold = 5.0
        self.labels = labels  # text for the labels [input-list, output-list]

        self.size = 0
        self.W = []  # list of numpy matrices
        self.b = []  # list of numpy vectors
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1]) - 0.5
            b = np.random.rand(layers[i + 1]) - 0.5
            self.W.append(w)
            self.b.append(b)
            self.size += layers[i] * layers[i + 1] + layers[i + 1]

        self.lRMS = []  # hold all traced RMSs to draw graph
        self.laccuracy = []  # hold all traced accuracies to draw graph

    def sigm(self, neta):
        return 1.0 / (1.0 + np.exp(-neta))

    def forward(self, x):  # fast forward (optimized in time, but not use to train!)
        for i in range(len(self.b)):
            net = np.dot(x, self.W[i]) + self.b[i]
            x = self.sigm(net)
        return x

    def to_chromosome(self):
        """
        Convert weights and biases to a flatten list to use in AG.
        """
        ch = []
        for w, b in zip(self.W, self.b):
            ch += w.flatten().tolist()
            ch += b.flatten().tolist()
        return ch

    def from_chromosome(self, ch):
        """
        Convert a flatten list (chromosome from a GA) to internal weights and biases.
        """
        if len(ch) != self.size:
            print(self.size)
            raise ValueError("Chromosome legnth doesn't match architecture")
        self.W = []
        self.b = []
        pos = 0
        for i in range(len(self.layers) - 1):  # for each layer
            to = self.layers[i] * self.layers[i + 1]  # number of weight
            w = np.array(ch[pos : pos + to]).reshape(self.layers[i], self.layers[i + 1])
            pos += to
            to = self.layers[i + 1]  # number of bias
            b = np.array(ch[pos : pos + to]).reshape(self.layers[i + 1])
            pos += to

            self.W.append(w)
            self.b.append(b)

    def plot_network(
        self, figsize=(12, 8), node_radius=0.03
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Generate a visualization of the MLP architecture with nodes and weighted connections.

        :param figsize: Tuple specifying the size of the figure (width, height).
        :param node_radius: Radius of the circles representing the nodes in the graph.
        :return: A tuple containing the Matplotlib Figure and Axes objects for further customization or saving.
        """
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

        # Node positions and labels
        layer_sizes = self.layers
        n_layers = len(layer_sizes)

        if self.labels is not None:
            input_labels = self.labels[0] if len(self.labels) > 0 else []
            output_labels = self.labels[1] if len(self.labels) > 1 else []

        v_spacing = 1.0 / (max(layer_sizes) + 1)
        h_spacing = 1.0 / (n_layers + 1)

        # Connections with color and thickness based on weight values
        for layer in range(n_layers - 1):
            layer_top = v_spacing * (max(layer_sizes) - layer_sizes[layer]) / 2.0
            next_layer_top = (
                v_spacing * (max(layer_sizes) - layer_sizes[layer + 1]) / 2.0
            )

            for i in range(layer_sizes[layer]):
                for j in range(layer_sizes[layer + 1]):
                    x1 = (layer + 1) * h_spacing
                    y1 = layer_top + (i + 1) * v_spacing
                    x2 = (layer + 2) * h_spacing
                    y2 = next_layer_top + (j + 1) * v_spacing

                    # Get weight
                    weight_value = self.W[layer][i, j]
                    weight_abs = abs(weight_value)

                    # Color based on weight sign
                    color = "green" if weight_value > 0 else "red"

                    # Line width based on absolute weight value
                    max_weight = abs(self.W[layer]).max()
                    linewidth = 0.3 + 3.0 * weight_abs / (max_weight + 0.001)

                    # Alpha based on absolute weight value
                    alpha = 0.2 + 0.7 * weight_abs / (max_weight + 0.001)

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=1,
                    )

        # Draw nodes with color based on layer type
        for layer in range(n_layers):
            layer_top = v_spacing * (max(layer_sizes) - layer_sizes[layer]) / 2.0

            for i in range(layer_sizes[layer]):
                x = (layer + 1) * h_spacing
                y = layer_top + (i + 1) * v_spacing

                if layer == 0:
                    color = "lightblue"
                elif layer == n_layers - 1:
                    color = "lightcoral"
                else:
                    color = "lightyellow"

                circle = plt.Circle(
                    (x, y),
                    node_radius,
                    color=color,
                    ec="black",
                    zorder=2,
                    linewidth=1.5,
                )
                ax.add_artist(circle)

                # Input labels
                if layer == 0 and i < len(input_labels):
                    ax.text(
                        x - node_radius - 0.02,
                        y,
                        input_labels[i],
                        fontsize=9,
                        ha="right",
                        va="center",
                        weight="bold",
                        c="black",
                    )

                # Output labels
                if layer == n_layers - 1 and i < len(output_labels):
                    ax.text(
                        x + node_radius + 0.02,
                        y,
                        output_labels[i],
                        fontsize=9,
                        ha="left",
                        va="center",
                        weight="bold",
                        c="black",
                    )

            if layer == 0:
                layer_label = "Input"
            elif layer == n_layers - 1:
                layer_label = "Output"
            else:
                layer_label = f"Hidden {layer}"

            ax.text(
                (layer + 1) * h_spacing,
                -0.05,
                layer_label,
                fontsize=11,
                ha="center",
                weight="bold",
                c="black",
            )

        architecture = " → ".join([str(size) for size in layer_sizes])
        ax.text(
            0.5,
            1.05,
            f"Architecture: {architecture}",
            fontsize=14,
            ha="center",
            weight="bold",
            c="black",
        )

        legend_elements = [
            Line2D([0], [0], color="green", lw=2, label="Positive weight"),
            Line2D([0], [0], color="red", lw=2, label="Negative weight"),
            Line2D([0], [0], color="gray", lw=0.5, label="Weak weight"),
            Line2D([0], [0], color="gray", lw=3, label="Strong weight"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        return fig, ax
