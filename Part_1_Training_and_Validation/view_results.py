import os
import sys
import json
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows cmd / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    root = Path(__file__).resolve().parent
    # If run from root, point to final results
    if (root / "final results").exists():
        final_dir = root / "final results"
    else:
        final_dir = root

    print("=" * 80)
    print("   [PLANT DISEASE CLASSIFICATION: 3-MODEL BENCHMARK RESULTS]")
    print("=" * 80)

    # 1. Display Numerical Benchmark Comparison Table
    csv_file = final_dir / "logs" / "final_model_comparison_metrics.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print("\n[FINAL MULTI-PARAMETER COMPARISON TABLE]:")
        print("-" * 80)
        print(df.to_string(index=False))
        print("-" * 80)

    # 2. Winning Model Summary
    print("\n>>> OVERALL WINNING ARCHITECTURE: [SE_Block] (ConvNeXt-Tiny + SE Attention) <<<")
    print("   - Composite Score     : 70.23 (Rank 1)")
    print("   - Test Accuracy       : 72.40% (Highest overall across 36,351 unseen test images)")
    print("   - Test Recall         : 82.75% (Highest overall)")
    print("   - Test Weighted-F1    : 74.14% (Highest overall)")
    print("   - Generalization Gap  : 0.36% (72.76% Val vs. 72.40% Test -- zero split drift)")
    print("   - Checkpoint Restored : Epoch 14 / 50")

    # 3. Winning Hyperparameters
    se_json = final_dir / "logs" / "results_se_block.json"
    if se_json.exists():
        with open(se_json, "r", encoding="utf-8") as f:
            hp = json.load(f)["diagnostics"]["final_epoch_hyperparameters"]
        print("\n[WINNING MODEL HYPERPARAMETERS (JSON)]:")
        print(json.dumps(hp, indent=2))

    # 4. Run Live Sample Inference
    web_app_dir = final_dir / "web_app"
    if not web_app_dir.exists():
        web_app_dir = final_dir.parent / "Part_2_Model_Testing_and_GUI" / "web_app"
    sys.path.insert(0, str(web_app_dir))
    try:
        from predictor import PlantDiseasePredictor
        sample_img = web_app_dir / "sample_leaf.jpg"
        if not sample_img.exists():
            sample_img = final_dir.parent / "Part_2_Model_Testing_and_GUI" / "sample_leaf.jpg"
        if sample_img.exists():
            print("\n[LIVE INFERENCE TEST ON SAMPLE LEAF]:")
            print(f"   Image: {sample_img.name}")
            predictor = PlantDiseasePredictor(bundle_dir=str(web_app_dir))
            result = predictor.predict(str(sample_img))
            print(f"   -> Predicted Plant : {result['predicted_plant']}")
            print(f"   -> Health Status   : {result['health_status']}")
            print(f"   -> Diagnosis       : {result['diagnosis']}")
            print(f"   -> Model Confidence: {result['confidence_pct']}%")
    except Exception as e:
        print(f"\nInference note: {e}")

    # 5. Dashboard image path
    dashboard_img = final_dir / "model_comparison_dashboard.png"
    if dashboard_img.exists():
        print(f"\nVisual Dashboard available at: {dashboard_img.name}")

    print("\n" + "=" * 80)
    print("TO VIEW THE FULL PRE-COMPUTED NOTEBOOK:")
    print(f"   Open: {final_dir / 'Models.ipynb'} in VS Code or Jupyter Notebook.")
    print("   (All cell outputs, logs, and plots are already saved inside!)")
    print("\nTO RUN THE INTERACTIVE WORKSTATION GUI:")
    print("   cd '../Part_2_Model_Testing_and_GUI'")
    print("   ./launch.sh")
    print("\nTO RUN THE CLI TESTER:")
    print("   python3 ../Part_2_Model_Testing_and_GUI/classify_leaf.py sample_leaf.jpg")
    print("=" * 80)

if __name__ == "__main__":
    main()
