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
        self._scaler = scaler  # Private attribute for scaler
        self.kernel = kernel
        self.feats = feats
        self.r2 = None         # The R-squared value of the model
        self.marg_lh = None    # The marginal likelihood of the model
        self.rmse = None       # The root mean squared error of the model
        self.mae = None        # The mean absolute error of the model
        self.idx = idx
        self.pipe = pipe       # Set pipeline using setter method

    # Getter for "scaler"
    @property
    def scaler(self):
        return self._scaler

    # Setter for scaler
    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler
        if self.pipe is not None:
            self.mk_pipe()

    # Getter for "pipe"
    @property
    def pipe(self):
        if self._pipe is None and self.scaler is not None:
            self.mk_pipe()
        return self._pipe

    # Setter for "pipe"
    @pipe.setter
    def pipe(self, pipe):
        self._pipe = pipe
        if pipe is not None:
            self.kernel = pipe[1].kernel
            self.scaler = pipe[0]

    def mk_pipe(self):
        """
        Create a pipeline with the scaler and Gaussian Process Regressor.
        """
        if self.scaler is not None and self.kernel is not None:
            self._pipe = Pipeline([
                ("scaler", self.scaler),
                ("gp", GaussianProcessRegressor(
                    kernel=self.kernel,
                    n_restarts_optimizer=100,
                    random_state=42,
                    normalize_y=True,
                    alpha=1e-10
                ))
            ])

    def __str__(self):
        """
        Return a string representation of the GPRPars instance.
        """
        attributes = [
            f"kernel: {self.kernel}",
            f"scaler: {self.scaler}",
            f"features: {self.feats}",
            f"R2: {self.r2}" if self.r2 is not None else "",
            f"log marginal likelihood: {self.marg_lh}" if self.marg_lh is not None else "",
            f"RMSE: {self.rmse}" if self.rmse is not None else "",
            f"MAE: {self.mae}" if self.mae is not None else ""
        ]
        return "\n".join(filter(lambda x: len(x) > 0, attributes))

# Example usage:
if __name__ == "__main__":
    # Create an instance of GPRPars
    gpr_pars = GPR(kernel=RBF(length_scale=1.0))

    print(f"\n{gpr_pars.scaler}\n")

    # Print the current state of the instance
    print("Current state of GPRPars instance:")
    print(gpr_pars)

    # Set the scaler using the setter method
    scaler_1 = StandardScaler()
    gpr_pars.scaler = scaler_1

    # Print the current state of the instance
    print(f"\nSecond state of GPRPars instance:")
    print(gpr_pars)

    # Create a pipeline and assign it to the instance
    pipe_1 = Pipeline([
        ("scaler", StandardScaler()),
        ("gp", GaussianProcessRegressor(kernel=RBF(length_scale=0.5)))
    ])
    gpr_pars.pipe = pipe_1

    # Print the updated state of the instance
    print("\nUpdated state of GPRPars instance with pipeline:")
    print(gpr_pars)
