# GaussianProcessRegressor is used for regression problems and provides a probabilistic prediction
from sklearn.gaussian_process import GaussianProcessRegressor
# Pipeline is used to assemble several steps (preparing data, training models, and making predictions)
# that can be cross-validated together while setting different parameters. This approach enhances reproducibility,
# modularity, efficiency, and scalability, making it easier to develop and maintain machine learning workflows.
# It is designed to apply transformations just to the input features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF


class GPR:
    def __init__(self, kernel=None, scaler=None, feats=None, idx=-1, pipe=None):
        """
        Initialize a GPR instance.
        :param kernel: The kernel used for Gaussian Process.
        :param scaler: The scaler used for data preprocessing.
        :param feats: Indices of feature columns used for the model.
        :param idx: Iteration counter as an index identifier.
        :param pipe: Optional pipeline object.
        """
        self._scaler = scaler     # Private attribute for scaler
        self.kernel = kernel
        self.feats = feats
        self.r2 = None            # The R-squared value of the model
        self.marg_lh = None       # The marginal likelihood of the model
        self.rmse = None          # The root mean squared error of the model
        self.mae = None           # The mean absolute error of the model
        self.idx = idx
        self._pipe = pipe         # Private attribute for pipe
        self.pipe_update = False  # Flag to indicate pipeline needs to be updated

    @property
    def scaler(self):
        """
        Getter for "scaler"
        :return: The scaler used for data preprocessing
        """
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):
        """
        Setter for scaler
        :param scaler: The scaler used for data preprocessing
        :return: None
        """
        self._scaler = scaler
        self.pipe_update = True
        if self.pipe_update:
            self.make_pipe()
            self.pipe_update = False  # Reset the flag

    @property
    def pipe(self):
        """
        Getter for "pipe"
        :return: The pipeline object combining scaler and Gaussian Process Regressor.
        """
        if self._pipe is None and self.scaler is not None:
            self.make_pipe()
        return self._pipe

    @pipe.setter
    def pipe(self, pipe):
        """
        Setter for "pipe"
        :param pipe: The pipeline object combining scaler and Gaussian Process Regressor.
        :return: None
        """
        self._pipe = pipe
        if pipe is not None:
            self.kernel = pipe[1].kernel
            self.scaler = pipe[0]

    def make_pipe(self):
        """
        Create a pipeline with the scaler and Gaussian Process Regressor.
        :return: None
        """
        if self.scaler is not None and self.kernel is not None:
            self._pipe = Pipeline([
                ("scaler", self.scaler),
                ("gp", GaussianProcessRegressor(
                    kernel=self.kernel,
                    # Number of optimizer restarts to find best params
                    n_restarts_optimizer=100,
                    # Sets a random seed for reproducibility
                    random_state=42,
                    # Normalizes the target values (y)
                    # during the fitting process
                    normalize_y=True,
                    # Value added to the kernel matrix diagonal during
                    # fitting, which helps in numerical stability
                    alpha=1e-10
                ))
            ])

    def __str__(self):
        """
        Overloading and magic method
        :return: STR representation of the GPRPars instance
        """
        attributes = [
            f"kernel: {self.kernel}",
            f"scaler: {self.scaler}",
            f"features: {self.feats}",
            f"pipe: {self.pipe}",
            f"R2: {self.r2}" if self.r2 is not None else "",
            f"log marginal likelihood: {self.marg_lh}" if self.marg_lh is not None else "",
            f"RMSE: {self.rmse}" if self.rmse is not None else "",
            f"MAE: {self.mae}" if self.mae is not None else ""
        ]
        return "\n".join(filter(lambda x: len(x) > 0, attributes))

# Example usage:
if __name__ == "__main__":
    # Create an instance of GPRPars
    gpr = GPR(kernel=RBF(length_scale=1.0))

    # Print the first state of the instance
    print("First state of GPRPars instance:")
    print(gpr)

    # Set the scaler using the setter method
    scaler_1 = StandardScaler()
    gpr.scaler = scaler_1

    # Print the second state of the instance by using the "scaler" setter
    print(f"\nSecond state of GPRPars instance:")
    print(gpr)

    # Print the third state of the instance by indicating the scaler within the class
    gpr = GPR(kernel=RBF(length_scale=1.0), scaler=StandardScaler())
    print(f"\nThird state of GPRPars instance:")
    print(gpr)

    # Create a pipeline and assign it to the instance
    pipe_1 = Pipeline([
        ("scaler", StandardScaler()),
        ("gp", GaussianProcessRegressor(kernel=RBF(length_scale=0.5)))
    ])
    gpr.pipe = pipe_1

    # Print the forth state of the instance by using the "pipe" setter
    print("\nForth state of GPRPars instance with pipeline:")
    print(gpr)

    # Print the fifth state of the instance by indicating the pipe within the class
    gpr = GPR(kernel=RBF(length_scale=1.0), scaler=StandardScaler(), pipe=Pipeline([
        ("scaler", StandardScaler()),
        ("gp", GaussianProcessRegressor(kernel=RBF(length_scale=0.5)))
    ]))
    print(f"\nFifth state of GPRPars instance:")
    print(gpr)
