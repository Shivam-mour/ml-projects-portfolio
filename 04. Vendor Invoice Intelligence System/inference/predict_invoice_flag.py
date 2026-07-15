import joblib
import pandas as pd

MODEL_PATH = (r"C:\Users\Shivam Mourya\ML PRO\Vendor Invoice Intelligence System\invoice_flagging\models\predict_flag_invoice.pkl")

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame with predicted freight cost
    """

    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df["Predicted_Freight"] = model.predict(input_df).round()
    return input_df


if __name__ == "__main__":

    # Example inference run (local testing)
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200]
    }

    prediction = predict_invoice_flag(sample_data)
    print(prediction)