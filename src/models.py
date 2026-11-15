from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# model overview:
# - Logistic Regression: Linear model based on straight-line equations.
# - Decision Tree: Single rule-based model using simple yes/no splits.
# - Random Forest: Ensemble model combining multiple decision trees via majority voting.
#
# why these 3 models?
# running dozens of models across 50 perturbation rounds would waste CPU and memory.
# these three models have completely different mathematical structures, giving a clear
# and balanced comparison on the same dataset.


def create_decision_tree_model(seed_value: int) -> DecisionTreeClassifier:
    """
    creates a single Decision Tree classifier.
    - max_depth=5: prevents memorizing the training data (overfitting).
    - strength: very fast and easy to interpret.
    - weakness: high variance; sensitive to small changes in data.
    """
    model = DecisionTreeClassifier(max_depth=5, random_state=seed_value)
    return model


def create_random_forest_model(seed_value: int) -> RandomForestClassifier:
    """
    creates a random forest model with 50 independent trees.
    - strength: reduces instability; multiple trees correct individual errors.
    - weakness: uses more cpu and ram compared to single models.
    """
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed_value)
    return model


def create_logistic_regression_model(seed_value: int) -> LogisticRegression:
    """
    creates a linear logistic regression model.
    - max_iter=1000: gives the solver enough steps to find the best boundary.
    - strength: very stable against data changes; uses minimal resources.
    - weakness: cnnot capture complex, non linear fraud patterns.
    """
    model = LogisticRegression(random_state=seed_value, max_iter=1000)
    return model
