class Perceptron:
    """
    Cannot learn xor by itself, so we use a trick.
    Xor = | but not !&
    """
    def __init__(self, num_inputs, learning_rate=1.0, initialization_method="zero"):
        match initialization_method:
            case "random":
                import random
                self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
                self.bias = random.uniform(-0.5, 0.5)
            case "zero":
                self.weights = [0.0] * num_inputs
                self.bias = 0.0
        self.learning_rate = learning_rate
        print(f"Perceptron initialized with {num_inputs} inputs. Initial weights: {self.weights}, bias: {self.bias}")

    def predict(self, inputs):
        weighted_sum = 0

        for w, x in zip(self.weights, inputs):
            weighted_sum += (w * x)

        # W1X1 + W2X2 + ... + B
        total_activation = weighted_sum + self.bias

        if total_activation > 0: # Step function threshold
            return 1
        return 0

    def train(self, training_data, epochs=10):
        print(f"\n--- Starting Training ---")
        for epoch in range(epochs):
            error_count = 0

            for inputs, target in training_data:
                prediction = self.predict(inputs)

                if prediction != target:
                    error_count += 1
                    error = target - prediction

                    for index, weight in enumerate(self.weights):
                        # Weight_new = Weight_old + (LearningRate * Error * Input)
                        self.weights[index] += self.learning_rate * error * inputs[index]

                    # Bias_new = Bias_old + (LearningRate * Error)
                    self.bias += self.learning_rate * error

            print(f"Epoch {epoch + 1}: Errors = {error_count}, Weights = {self.weights}, Bias = {self.bias}")

            if error_count == 0:
                print(f"--- Training Converged Successfully in {epoch + 1} epochs! ---")
                return

        print("--- Training finished after maximum epochs. ---")


if __name__ == "__main__":
    PRETTY_LINE = "=" * 50

    print(PRETTY_LINE)
    print("======= Training Neuron Y1 (for AND logic) =======")
    print(PRETTY_LINE)

    data_and = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1)
    ]

    neuron_y1 = Perceptron(num_inputs=2, learning_rate=1.0)
    neuron_y1.train(data_and)

    print("\n")
    print(PRETTY_LINE)
    print("======= Training Neuron Y2 (for OR logic)  =======")
    print(PRETTY_LINE)

    data_or = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]

    neuron_y2 = Perceptron(num_inputs=2, learning_rate=1.0)
    neuron_y2.train(data_or)

    print("\n")
    print(PRETTY_LINE)
    print("======= Training Output Neuron Z           =======")
    print(PRETTY_LINE)

    data_z = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([0, 1], 1),
        ([1, 1], 0)
    ]

    neuron_z = Perceptron(num_inputs=2, learning_rate=1.0)
    neuron_z.train(data_z)

    print("\n")
    print(PRETTY_LINE)
    print("======= Testing the complete XOR Network   =======")
    print(PRETTY_LINE)

    def xor_network(inputs):
        output_y1 = neuron_y1.predict(inputs)
        output_y2 = neuron_y2.predict(inputs)
        final_output = neuron_z.predict([output_y1, output_y2])

        return final_output

    test_inputs = [
        [0, 0], [0, 1], [1, 0], [1, 1]
    ]
    for test_input in test_inputs:
        result = xor_network(test_input)
        print(f"Input: {test_input} -> XOR Output: {result}")