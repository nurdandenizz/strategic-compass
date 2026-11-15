import argparse
import numpy as np

from src.models import create_decision_tree_model
from src.models import create_random_forest_model
from src.models import create_logistic_regression_model


def generate_synthetic_bank_data():
    """
    generates synthetic banking dataset for simulation.
    simulates an imbalanced distribution (95% normal, 5% fraud)
    and injects realistic behavioral signals into fraud transactions.
    """
    np.random.seed(42)
    total_transactions = 1000
    num_features = 10
    
    X = np.random.randn(total_transactions, num_features)
    y = np.random.choice([0, 1], size=total_transactions, p=[0.95, 0.05])
    
    # injecting behavioral anomalies into fraud (y == 1) transactions
    fraud_indices = np.where(y == 1)[0]
    
    # features 1 and 2 (e.g., transaction amount and velocity): higher mean in fraud cases
    X[fraud_indices, 0] += 2.2
    X[fraud_indices, 1] += 1.8
    
    # feature 3 (e.g., account age or location consistency): lower mean in fraud cases
    X[fraud_indices, 2] -= 1.5
    
    return X, y


def main():

    # -------------------------------------------------------------------------
    # PHASE 1: Infrastructure and Command-Line Argument Management
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Asymmetric Financial Risk and Model Stability Benchmark Platform",
        formatter_class=argparse.RawTextHelpFormatter
    )
    args = parser.parse_args()
     
    print("[INFO] Benchmark engine initialized successfully...")
    print("[INFO] Platform architecture: 3 distinct machine learning models will be evaluated concurrently.")
    print("[INFO] 1000 time-ordered transactions loaded.")
    
    X, y = generate_synthetic_bank_data()
    
    # temporal wall: chronological split to prevent data leakage
    X_train, X_test = X[:750], X[750:]
    y_train, y_test = y[:750], y[750:]
    print("[INFO] Temporal wall applied! Training set: 750, Test set: 250\n")
    
    # dynamic model registry: mapping factory functions to eliminate conditional branching
    model_pool = {
        "DT": {"name": "Decision Tree (Rule-Based Tree Model)", "func": create_decision_tree_model},
        "RF": {"name": "Random Forest (Ensemble Tree Model)", "func": create_random_forest_model},
        "LR": {"name": "Logistic Regression (Linear Parametric Model)", "func": create_logistic_regression_model}
    }
    
    final_model_losses = {}
    final_model_entropies = {}

    # -------------------------------------------------------------------------
    # PHASE 2: Monte Carlo Simulation and Bootstrap Data Perturbation
    # -------------------------------------------------------------------------
    for model_code, model_info in model_pool.items():
        model_name = model_info["name"]
        model_factory_func = model_info["func"]
        
        print(f"[!] Starting 50-round Monte Carlo stress test for {model_name}...")
        all_rounds_probabilities = []
        
        for perturbation_seed in range(1, 51):
            rng = np.random.default_rng(seed=perturbation_seed)
            
            # perturbing training data distribution via bootstrap resampling
            selected_indices = rng.choice(750, size=750, replace=True)
            X_train_perturbed = X_train[selected_indices]
            y_train_perturbed = y_train[selected_indices]
            
            # dynamic model instantiation: bulky if/elif blocks completely removed
            model = model_factory_func(perturbation_seed)

            model.fit(X_train_perturbed, y_train_perturbed)
            
            round_predictions = model.predict_proba(X_test)[:, 1]
            all_rounds_probabilities.append(round_predictions)

        # -------------------------------------------------------------------------
        # PHASE 3: Selection Entropy (Model Instability) Engine
        # -------------------------------------------------------------------------
        predictions_matrix = np.array(all_rounds_probabilities)
        mean_probabilities = np.mean(predictions_matrix, axis=0)
        mean_probabilities = np.clip(mean_probabilities, 1e-15, 1 - 1e-15)
        
        selection_entropy = - (mean_probabilities * np.log2(mean_probabilities) + (1 - mean_probabilities) * np.log2(1 - mean_probabilities))
        overall_model_instability = np.mean(selection_entropy)
        
        final_model_entropies[model_code] = overall_model_instability

        # -------------------------------------------------------------------------
        # PHASE 4: Cost-Sensitive Evaluation (Asymmetric Financial Filter)
        # -------------------------------------------------------------------------
        c_fn = 20  # false negative cost
        c_fp = 1   # false positive cost
        
        financial_threshold = c_fp / (c_fp + c_fn)
        financial_decisions = (mean_probabilities >= financial_threshold).astype(int)

        actual_frauds = y_test.astype(int)
        false_alarms = np.sum((financial_decisions == 1) & (actual_frauds == 0))
        missed_frauds = np.sum((financial_decisions == 0) & (actual_frauds == 1))
        total_financial_loss = (false_alarms * c_fp) + (missed_frauds * c_fn)
        
        final_model_losses[model_code] = total_financial_loss
        
        print(f"    [+] Mathematical Optimal Threshold: {financial_threshold:.4f}")
        print(f"    [+] Blocked Customers (FP): {false_alarms} | Missed Frauds (FN): {missed_frauds}")
        print(f"    [+] Net Financial Loss: {total_financial_loss} USD\n")

    # -------------------------------------------------------------------------
    # PHASE 5: Comparative Analysis and Dynamic Model Selection
    # -------------------------------------------------------------------------
    print("="*75)
    print("      OFFICIAL FINANCIAL BENCHMARK AND MODEL COMPARISON REPORT")
    print("="*75)
    
    for model_code, model_info in model_pool.items():
        print(f"MODEL: {model_code:<4} | Selection Entropy: {final_model_entropies[model_code]:.4f} | Expected Financial Loss: {final_model_losses[model_code]:>5} USD")
    
    champion_model_code = min(final_model_losses, key=final_model_losses.get)
    
    print("\n[ANALYSIS RESULT AND DYNAMIC MODEL SELECTION]")
    print("    Passed the asymmetric cost filter and maintained the most")
    print("    optimized risk balance across perturbation rounds:")
    print(f"     SELECTED CHAMPION MODEL: {model_pool[champion_model_code]['name']}")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()
