class Perceptron:
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
        return 1 if sum(w * x for w, x in zip(self.weights, inputs)) + self.bias > 0 else 0

    def train(self, training_data, epochs=10):
        print(f"\n--- Starting Training ---")
        for epoch in range(epochs):
            error_count = 0

            for inputs, target in training_data:
                prediction = self.predict(inputs)

                if prediction != target:
                    error_count += 1
                    error = target - prediction

                    for i, w in enumerate(self.weights):
                        self.weights[i] += self.learning_rate * error * inputs[i]

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