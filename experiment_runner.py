import argparse
import numpy as np

from src.models import create_decision_tree_model
from src.models import create_random_forest_model
from src.models import create_logistic_regression_model


def generate_synthetic_bank_data():
    """
    generates synthetic banking dataset with temporal concept drift for simulation.
    simulates an imbalanced distribution (95% normal, 5% fraud)
    and injects realistic behavioral signals into fraud transactions.
    """
    rng = np.random.default_rng(seed=42)
    total_transactions = 1000
    num_features = 10

    x = rng.standard_normal((total_transactions, num_features))
    y = rng.choice(list((0, 1)), size=total_transactions, p=list((0.95, 0.05)))

    # temporal drift (concept drift): transaction behavior gradually shifts over time
    time_trend = np.linspace(0.0, 0.6, total_transactions)
    x[:, 0] += time_trend
    x[:, 1] -= time_trend * 0.5

    # injecting behavioral anomalies into fraud (y == 1) transactions
    fraud_indices = np.where(y == 1)[0]

    # features 1 and 2 (transaction amount and velocity): higher mean in fraud cases
    x[fraud_indices, 0] += 2.2
    x[fraud_indices, 1] += 1.8

    # feature 3 (account age or location consistency): lower mean in fraud cases
    x[fraud_indices, 2] -= 1.5

    return x, y


def compute_binary_entropy(prob_array):
    """shannon binary entropy formula: h(p) = -p*log2(p) - (1-p)*log2(1-p)"""
    p = np.clip(prob_array, 1e-15, 1.0 - 1e-15)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def main():
    # -------------------------------------------------------------------------
    # phase 1: infrastructure and command-line argument management
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="asymmetric financial risk and model stability benchmark platform (final)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--rounds", type=int, default=50, help="number of bootstrap rounds (default: 50)")
    parser.add_argument("--fn-cost", type=float, default=20.0, help="false negative cost (default: 20.0)")
    parser.add_argument("--fp-cost", type=float, default=1.0, help="false positive cost (default: 1.0)")
    parser.add_argument("--instability-penalty", type=float, default=50.0, help="epistemic instability penalty weight (default: 50.0)")
    args = parser.parse_args()

    print("[INFO] Benchmark engine initialized successfully...")
    print(f"[INFO] Parameters -> Rounds: {args.rounds} | FN Cost: {args.fn_cost} | FP Cost: {args.fp_cost}")

    x, y = generate_synthetic_bank_data()

    # temporal wall: chronological split to prevent data leakage
    x_train, x_test = x[:750], x[750:]
    y_train, y_test = y[:750], y[750:]
    print("[INFO] Temporal wall applied! Training set: 750, Test set: 250\n")

    model_pool = {
        "dt": {"name": "Decision Tree (Rule-Based Tree Model)", "func": create_decision_tree_model},
        "rf": {"name": "Random Forest (Ensemble Tree Model)", "func": create_random_forest_model},
        "lr": {"name": "Logistic Regression (Linear Parametric Model)", "func": create_logistic_regression_model},
    }

    final_model_losses = {}
    final_model_instabilities = {}
    final_composite_scores = {}

    # -------------------------------------------------------------------------
    # phase 2: monte carlo simulation and bootstrap data perturbation
    # -------------------------------------------------------------------------
    for model_code, model_info in model_pool.items():
        model_name = model_info["name"]
        model_factory_func = model_info["func"]

        print(f"[!] Starting {args.rounds}-round Monte Carlo stress test for {model_name}...")
        all_rounds_probabilities = []

        for perturbation_seed in range(1, args.rounds + 1):
            rng = np.random.default_rng(seed=perturbation_seed)

            selected_indices = rng.choice(len(x_train), size=len(x_train), replace=True)
            x_train_perturbed = x_train[selected_indices]
            y_train_perturbed = y_train[selected_indices]

            model = model_factory_func(perturbation_seed)
            model.fit(x_train_perturbed, y_train_perturbed)

            round_predictions = model.predict_proba(x_test)[:, 1]
            all_rounds_probabilities.append(round_predictions)

        # -------------------------------------------------------------------------
        # phase 3: epistemic uncertainty (true inter-round model instability)
        # -------------------------------------------------------------------------
        predictions_matrix = np.array(all_rounds_probabilities)
        mean_probabilities = np.mean(predictions_matrix, axis=0)

        total_entropy = compute_binary_entropy(mean_probabilities)
        mean_round_entropy = np.mean(compute_binary_entropy(predictions_matrix), axis=0)
        epistemic_instability = np.maximum(0.0, total_entropy - mean_round_entropy)

        overall_model_instability = float(np.mean(epistemic_instability))
        final_model_instabilities[model_code] = overall_model_instability

        # -------------------------------------------------------------------------
        # phase 4: cost-sensitive evaluation (asymmetric financial filter)
        # -------------------------------------------------------------------------
        c_fn = args.fn_cost
        c_fp = args.fp_cost

        financial_threshold = c_fp / (c_fp + c_fn)
        financial_decisions = (mean_probabilities >= financial_threshold).astype(int)

        actual_frauds = y_test.astype(int)
        false_alarms = int(np.sum((financial_decisions == 1) & (actual_frauds == 0)))
        missed_frauds = int(np.sum((financial_decisions == 0) & (actual_frauds == 1)))
        total_financial_loss = (false_alarms * c_fp) + (missed_frauds * c_fn)

        composite_risk_score = total_financial_loss + (overall_model_instability * args.instability_penalty)

        final_model_losses[model_code] = total_financial_loss
        final_composite_scores[model_code] = composite_risk_score

        print(f"    [+] Optimal Threshold: {financial_threshold:.4f} | FP: {false_alarms} | FN: {missed_frauds}")
        print(f"    [+] Financial Loss: {total_financial_loss:.1f} USD | Epistemic Instability: {overall_model_instability:.4f}\n")

    # -------------------------------------------------------------------------
    # phase 5: comparative analysis and risk-weighted model selection
    # -------------------------------------------------------------------------
    print("=" * 88)
    print("           OFFICIAL FINANCIAL BENCHMARK AND MODEL COMPARISON REPORT")
    print("=" * 88)

    for model_code in model_pool:
        print(
            f"MODEL: {model_code.upper():<4} | "
            f"Epistemic Instability: {final_model_instabilities[model_code]:.4f} | "
            f"Financial Loss: {final_model_losses[model_code]:>6.1f} USD | "
            f"Composite Risk Score: {final_composite_scores[model_code]:>6.2f}"
        )

    champion_model_code = min(final_composite_scores, key=final_composite_scores.get)

    print("\n[ANALYSIS RESULT AND DYNAMIC MODEL SELECTION]")
    print(f"     SELECTED CHAMPION MODEL: {model_pool[champion_model_code]['name']}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
