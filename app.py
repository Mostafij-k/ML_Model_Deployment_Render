from flask import Flask, request, render_template
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Create feature array
        features = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])

        # Make prediction
        prediction = model.predict(features)

        # Get flower name
        flower_name = prediction[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Flower: {flower_name}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)