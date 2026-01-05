import numpy as np


class SimpleMamdaniFIS:
    def __init__(self, n_mf):
        self.n_mf = n_mf  # Number of membership functions
        self.rules = {}  # Rules dictionary
        self.centers = {}  # The Central membership function for each variable will be stored here

    def _create_centers(self, min_val, max_val):
        """
        Generates evenly spaced points.
        """
        return np.linspace(min_val, max_val, self.n_mf)

    @staticmethod
    def _membership(x, center, width):
        """
        Calculates the Triangular Membership Degree (mu) for a given input x.
        Formula: mu(x) = max(0, 1 - |x - c| / w)
        """
        return np.maximum(0, 1 - np.abs(x - center) / width)

    def fit(self, train_data):
        X = train_data[:, :2]  # Inputs: X1, X2
        Y = train_data[:, 2]  # Output

        # 1. Define the domain and centers for inputs and output
        self.centers['x1'] = self._create_centers(1, 10)
        self.centers['x2'] = self._create_centers(1, 10)
        self.centers['y'] = self._create_centers(np.min(Y), np.max(Y))

        # Width of membership functions (distance between two centers)
        width_x = self.centers['x1'][1] - self.centers['x1'][0]

        # 2. Generate rules from the data
        for i in range(len(Y)):
            # Find the nearest membership function for each input/output
            # argmin calculates the distance and returns the index of the nearest center
            idx_x1 = np.argmin(np.abs(X[i, 0] - self.centers['x1']))
            idx_x2 = np.argmin(np.abs(X[i, 1] - self.centers['x2']))
            idx_y = np.argmin(np.abs(Y[i] - self.centers['y']))

            # Calculate the rule strength degree (product of membership degrees)
            mu_x1 = self._membership(X[i, 0], self.centers['x1'][idx_x1], width_x)
            mu_x2 = self._membership(X[i, 1], self.centers['x2'][idx_x2], width_x)
            degree = mu_x1 * mu_x2

            rule_key = (idx_x1, idx_x2)

            # If a rule with this condition already exists, keep the one with the higher degree
            if rule_key not in self.rules:
                self.rules[rule_key] = (idx_y, degree)
            elif degree > self.rules[rule_key][1]:
                    self.rules[rule_key] = (idx_y, degree)

    def predict(self, test_data, use_fuzzy_output=True):
        predictions = []
        width_x = self.centers['x1'][1] - self.centers['x1'][0]

        for i in range(len(test_data)):
            x1_val, x2_val = test_data[i, 0], test_data[i, 1]

            # Fuzzy calculations
            numerator = 0.0
            denominator = 0.0

            # Not fuzzy calculations
            max_fire = 0.0
            best_output = 0.0

            # Check all existing rules
            for (r_idx1, r_idx2), (r_out_idx, _) in self.rules.items():
                c1 = self.centers['x1'][r_idx1]
                c2 = self.centers['x2'][r_idx2]
                out_center = self.centers['y'][r_out_idx]

                # Calculate the firing strength of the rule for new data
                mu1 = self._membership(x1_val, c1, width_x)
                mu2 = self._membership(x2_val, c2, width_x)
                firing_strength = mu1 * mu2  # (Product Inference)

                if firing_strength > 0:
                    # Method 1: Using fuzzy output (weighted average)
                    numerator += firing_strength * out_center
                    denominator += firing_strength

                    # Method 2: For non-fuzzy output cases (finding the absolute winner)
                    if firing_strength > max_fire:
                        max_fire = firing_strength
                        best_output = out_center

            if use_fuzzy_output:
                # Fuzzy output mode: Weighted average of centers
                if denominator == 0:
                    predictions.append(0)  # Or a global average value
                else:
                    predictions.append(numerator / denominator)
            else:
                # Non-fuzzy output mode (Look-up Table): Output of the strongest rule
                predictions.append(best_output)

        return np.array(predictions)
