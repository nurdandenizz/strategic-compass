import argparse

def main():
    parser = argparse.ArgumentParser(description="Strategic Compass - Model Selection Benchmark")
    parser.add_argument("--model", type=str, required=True, choices=["lr", "dt", "rf"], 
                        help="Seçenekler: lr (Logistic Regression), dt (Decision Tree), rf (Random Forest)")

    args = parser.parse_args()
    print(f"[INFO] Benchmark motoru başarıyla başlatıldı...")
    print(f"[INFO] Seçilen Model: {args.model.upper()}")

if __name__ == "__main__":
    main()
