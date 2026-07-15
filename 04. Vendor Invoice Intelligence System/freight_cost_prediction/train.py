import joblib
from pathlib import Path

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (
    train_linear_regression,
    train_Decision_tree,
    train_random_forest,
    evaluate_model
)

def main():
    db_path = (r"C:\Users\Shivam Mourya\ML PRO\Vendor Invoice Intelligence System\data\inventory.db")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok = True)

    #load data
    df = load_vendor_invoice_data(db_path)

    #prepare data
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    #train models
    lr_model = train_linear_regression(X_train, y_train)
    df_model = train_Decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    #evaluate models
    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(df_model, X_test, y_test, "Decision tree Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest Regression"))

    #select best model (lowest MAE)
    best_model_info = min(results, key= lambda x:x["mae"])
    best_model_name = best_model_info["model_name"]

    best_model = {
        "Linear Regression": lr_model,
        "Decision tree Regression": df_model,
        "Random Forest Regression": rf_model
    }[best_model_name]

    #save best model
    model_path = model_dir/ "Predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model Path: {model_path}")

if __name__ == "__main__":
    main()
    