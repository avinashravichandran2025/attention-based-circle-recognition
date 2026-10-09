Circle Recognition Using Attention-Based Models

Attention Models for Pattern Recognition | Regression-Only Approach

A Transformer-based approach to geometric pattern recognition using synthetic point clouds, displacement-vector regression, and normalized signed-distance prediction.

1. Overview

This project investigates circle recognition using attention-based models. Given an unordered set of points, the model learns to estimate the circle centre and determine whether each point lies inside or outside the circle.

The final approach uses a regression-only formulation. Instead of predicting a separate binary classification score, the model predicts a displacement vector and a normalized signed-distance factor for each point.

The project evaluates the effects of point-cloud size, coordinate noise, model complexity, training-set size, and input embedding on circle-recognition performance.

2. Research Objectives

* Investigate Transformer-based attention for geometric pattern recognition.
* Recognize circles from unordered point-cloud data.
* Predict displacement vectors pointing toward the circle centre.
* Use normalized signed-distance regression to infer inside/outside membership.
* Replace the combined regression-and-BCE objective with a regression-only formulation.
* Evaluate model robustness and localization performance under different experimental settings.

3. Problem Formulation

Input Representation

Each point contains three input features:

p = (x, y, intensity)

where:

* x: horizontal coordinate.
* y: vertical coordinate.
* intensity: grayscale value.

Coordinates are sampled from the square [-1, 1] × [-1, 1].

The synthetic dataset uses:

* 1.0 for points inside the circle.
* 0.0 for points outside the circle.

The circle centre and radius are unknown to the model.

Model Outputs

For every point, the model predicts:

1. Displacement vector: A two-dimensional vector pointing from the point toward the circle centre.
2. Normalized signed-distance factor: A scalar describing the point’s position relative to the circle boundary.

These predictions provide the geometric information needed to estimate the circle centre and infer inside/outside membership.

4. Why Regression-Only Instead of BCE?

The initial formulation combined displacement-vector regression with binary cross-entropy (BCE) classification.

Initial loss function

L_initial = MSE(v, v_hat) + BCE(y, p_hat)

Here, v is the ground-truth displacement vector, v_hat is its prediction, y is the binary inside/outside label, and p_hat is the predicted classification probability.

The two objectives shared the model’s learned features but operated on different scales. Balancing them required an additional loss-weighting factor. The earlier formulation also exhibited undesirable validation-loss variance.

Final Regression-Only Formulation

The final approach removes the separate classification head and predicts a normalized signed-distance factor.

Normalized signed distance

alpha = (d / r) - 1

where:

* d = ||p - c|| is the Euclidean distance between point p and the true circle centre c.
* r is the circle radius.

The sign of alpha provides the inside/outside decision:

Value	Interpretation
alpha = -1	Point at the circle centre
-1 <= alpha < 0	Point inside the circle
alpha = 0	Point on the circle boundary
alpha > 0	Point outside the circle

Because the sign of the signed-distance prediction already encodes inside/outside membership, a separate BCE classification objective is unnecessary for this formulation.

Final Training Loss

L_final = MSE(v, v_hat) + MSE(r * alpha, r * alpha_hat)

The signed-distance target and prediction are multiplied by the circle radius so that the corresponding regression term is expressed in distance units, matching the scale of the displacement-vector term.

This formulation eliminates the manually tuned loss-weighting factor.

Benefits of the Regression-Only Approach

* Simpler objective: No separate BCE classification loss is required.
* Geometric information: Signed-distance regression represents a point’s position relative to the circle boundary.
* Consistent loss scales: Both regression terms are expressed in distance units.
* Fewer hyperparameters: No additional loss-weighting factor is needed.

This design is motivated by the specific requirements of circle recognition. It does not establish that regression-only models are universally better than BCE-based models.

5. Model Architecture

The baseline model uses a Transformer-based architecture.

1. Input: Point-cloud tensor with shape (batch_size, num_points, 3).
2. Input embedding: Linear projection from 3 input features to 128 hidden dimensions.
3. Self-attention: One pre-normalization Transformer block.
4. Feed-forward network: Hidden dimensions 128 → 512 → 128, using ReLU activation.
5. Displacement head: Predicts two displacement-vector components.
6. Signed-distance head: Predicts the normalized signed-distance factor alpha.

The architecture uses LayerNorm, residual connections, and self-attention without positional encodings.

Permutation Equivariance

The point order should not determine the geometric prediction. The model is designed so that permuting the input points permutes their corresponding outputs in the same way.

No-Embedding Ablation

A separate configuration applies attention directly to raw three-dimensional point features using manually implemented query, key, and value projections. This allows comparison with the standard learned input embedding.

6. Dataset and Training Configuration

The project uses a synthetic circle dataset generated on demand.

Parameter	Baseline setting
Input features	3
Points per sample	100
Coordinate range	[-1, 1]
Hidden dimension	128
Attention heads	1
Transformer layers	1
Batch size	32
Optimizer	Adam
Learning rate	0.001
Maximum epochs	100
Validation seed	42
Early stopping	Based on validation loss

Training samples are generated afresh each epoch, while validation data is fixed using seed 42.

Gaussian noise is applied to coordinates only. The grayscale intensity values remain unchanged. The final report records early stopping at epoch 81.

7. Circle-Centre Estimation

The displacement vector is defined as:

v = c - p

where p is the point coordinate and c is the circle centre.

The predicted displacement provides a point-wise centre estimate:

c_hat = p + v_hat

The final centre estimate is obtained by averaging these point-wise estimates over points predicted to be inside the circle.

The signed-distance prediction supplies the inside/outside decision through the sign of alpha.

8. Experimental Design

Five principal experiments investigate the model’s performance.

Experiment	Purpose
Experiment 1: Point-cloud size	Evaluates the effect of using 10–200 points.
Experiment 2: Noise robustness	Evaluates coordinate noise with standard deviation from 0 to 1.0.
Experiment 3: Model complexity	Compares hidden dimensions, attention heads, and Transformer depth.
Experiment 4: Training-set size	Compares 1,000, 5,000, and 10,000 training samples.
Experiment 5: Embedding ablation	Compares standard input embedding with the no-embedding variant.

Evaluation Metrics

* Classification accuracy: Inside/outside membership inferred from the sign of the predicted signed-distance factor.
* Centre error: Euclidean distance between the estimated and true circle centres.
* Inference runtime: Average processing time per sample under the reported evaluation setup.

Centre error is especially important because classification accuracy can remain high even when the estimated circle centre is not sufficiently precise.

9. Experimental Results

The following results are reported in the final project report.

Baseline Performance

Evaluation on 500 seeded synthetic patches produced:

Metric	Reported result
Classification accuracy	99.60%
Centre error	0.079
Inference runtime	1.79 ms/sample

Experiment 1: Point-Cloud Size

As the number of points increases from 10 to 200, centre error decreases from 0.191 to 0.059. Classification accuracy remains approximately 99.4–99.9%.

Experiment 2: Noise Robustness

Centre error increases from 0.077 on clean coordinates to 0.243 at a noise standard deviation of 1.0. Classification accuracy dips to approximately 94.7% around a standard deviation of 0.5 and recovers to around 96% at 1.0.

This demonstrates that high classification accuracy does not necessarily guarantee precise localization.

Experiment 3: Model Complexity

Centre error decreases from 0.081 to 0.039 across the tested configurations. The best reported four-head, three-layer configuration approximately halves the baseline centre error.

Experiment 4: Training-Set Size

Centre error improves from 0.088 to 0.078 and then 0.075 as the training-set size increases from 1,000 to 5,000 and 10,000 samples.

The improvement becomes smaller at larger training-set sizes.

Experiment 5: Input-Embedding Ablation

The standard embedding achieves 99.8% classification accuracy and 0.083 centre error. The no-embedding variant achieves 90.8% accuracy and 0.195 centre error.

These results indicate that the learned input embedding improves performance in the tested configurations.

10. Visualizations

Evaluation Results

Training Loss

Failure Cases

The corresponding CSV files contain the numerical experimental results. The repository also includes a supplementary loss-weighting CSV and plot from the earlier formulation.

11. Failure Cases and Limitations

High Coordinate Noise

Increasing coordinate noise makes the geometric structure less reliable and increases centre-localization error.

Low Contrast

When intensity contrast is weak, the model may fail to predict any points as inside the circle. Without reliable inside-point estimates, centre recovery becomes unreliable.

Sparse Point Clouds

When only a small number of points are available, the model has less geometric evidence for precise localization.

General Limitations

* The dataset is synthetic and does not establish performance on real-world point clouds.
* Results depend on the noise level, random seed, and model configuration.
* Classification accuracy alone does not fully characterize localization quality.
* Inference runtime depends on the hardware and software environment.
* The results do not prove that regression-only training is universally superior to BCE-based training.

12. Repository Structure

attention-based-circle-recognition/
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── experiments.py
├── generate_failures.py
├── exp1_points.csv
├── exp2_noise.csv
├── exp3_complexity.csv
├── exp4_trainsize.csv
├── exp5_embedding_comparison.csv
├── exp6_loss_weighting.csv
├── evaluation_results.png
├── exp1_points.png
├── exp2_noise.png
├── exp3_complexity.png
├── exp4_trainsize.png
├── exp5_embedding_comparison.png
├── exp6_loss_weighting.png
├── failure_cases.png
├── loss_curve.png
├── requirements.txt
├── .gitignore
└── README.md

The loss-weighting CSV and plot are retained as supplementary artifacts from the earlier formulation. The final method described here uses regression-only training.

13. Installation

Step 1: Clone the repository

git clone https://github.com/avinashravichandran2025/attention-based-circle-recognition.git
cd attention-based-circle-recognition

Step 2: Create a virtual environment

python3 -m venv venv
source venv/bin/activate

Step 3: Install dependencies

pip install -r requirements.txt

Ensure that requirements.txt lists the Python packages required by the final implementation.

14. Running the Project

Train the model

python train.py

Evaluate the trained model

python evaluate.py

Generate failure-case visualizations

python generate_failures.py

Run the experiments

python experiments.py

Run training first if evaluation or failure-case generation requires a saved model checkpoint. Consult the Python scripts for the exact configuration and output paths.

15. Conclusion

This project investigates regression-only circle recognition using a Transformer-based model on synthetic point clouds.

The final formulation replaces the separate BCE classification objective with normalized signed-distance regression. Radius scaling expresses the signed-distance loss in the same distance units as the displacement-vector loss, removing the need for a manually tuned loss-weighting factor.

The reported experiments demonstrate the feasibility of this approach and show that model depth, input embedding, point-cloud size, and coordinate noise affect localization performance. Low-contrast inputs remain an important limitation.

Contributors

Project: Attention Models for Pattern Recognition — Circle Recognition Using Point Clouds.

Add the names of all team members and their agreed contributions, clearly identifying individual contributions to the circle-recognition component.
