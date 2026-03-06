from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

loaded_data = joblib.load("model_and_scaler.pkl")
model = loaded_data["model"]
scaler = loaded_data["scaler"]

print("✅ Model and Scaler loaded successfully!")

crop_data = {
    ("Kanchipuram, Chengalpet, Tiruvallur, Cuddalore, Villupuram, Kallakuruchi, Vellore, Tirupathur, Ranipet, Tiruvannamalai", "100-200"): {"rainfall": 1105, "crops": ["Rice", "Pearl Millet", "Sorghum", "Gingelly", "Finger Millet", "Groundnut", "Red Gram", "Sugarcane", "Cashew", "Mango", "Guards", "Green Chillies", "Brinjal", "Tapioca", "Yam", "Banana", "Jack", "Guava", "Watermelon", "Turmeric", "Tube rose", "Crossandra", "Lemongrass"]},
    ("Dharmapuri, Salem, Namakkal", "200-600"): {"rainfall": 875, "crops": ["Sorghum", "Rice", "Millet", "Groundnut", "Horse Gram", "Cotton", "Sugarcane", "Tapioca", "Cotton", "Gingelly", "Chillies", "Mango", "Banana", "Tobacco", "Pulses", "Jack", "Tomato", "Radish", "Brinjal", "Ladies Finger", "Pepper", "Arecanut", "Cocoa", "Coconut", "Palmarosa", "Chrysanthemum", "Jasmine", "Marigold", "Rose", "Tuberose", "Cutflowers", "Turmeric", "Red Chillies"]},
    ("Erode, Coimbatore, Karur (part), Namakkal (part), Dindigul (part), Theni (part)", "200-600"): {"rainfall": 715, "crops": ["Sorghum", "Pulses", "Groundnut", "Rice", "Millets", "Cumbu", "Cotton", "Sugarcane", "Ragi", "Black Gram", "Sunflower", "Green Gram", "Gingelly", "Red Gram", "Turmeric", "Maize", "Banana", "Onion", "Castor", "Tobacco", "Guava", "Onion", "Guards", "Tomato", "Tea", "Coffee", "Coconut", "Gloriosa", "Flowers", "Tapioca", "Jasmine", "Rose", "Vegetables"]},
    ("Tiruchi, Perambalur, Pudukottai (part), Thanjavur, Nagapattinam, Mayiladuthurai, Tiruvarur, Cuddalore (part)", "100-200"): {"rainfall": 984, "crops": ["Rice", "Cumbu", "Maize", "Cholam", "Ragi", "Black Gram", "Green Gram", "Coconut", "Gingelly", "Castor", "Groundnut", "Banana", "Onion", "Cashew", "Betel vine", "Citrus", "Jack", "Vegetables"]},
    ("Madurai, Sivagangai, Ramanathapuram, Virudhunagar, Tirunelveli, Tenkasi, Thoothukudi", "100-600"): {"rainfall": 857, "crops": ["Rice", "Maize", "Cumbu", "Cholam", "Ragi", "Black Gram", "Greengram", "Groundnut", "Fodder Crops", "Gingelly", "Castor", "Cotton", "Chillies", "Banana", "Jasmine", "Coriander", "Onion", "Lime", "Cashew", "Amla"]},
    ("Kanniyakumari", "100-2000"): {"rainfall": 1420, "crops": ["Rice", "Banana", "Jackfruit", "Mango", "Tapioca", "Cashew nut", "Coconut", "Clove", "Vegetables", "Tamarind"]},
    ("Nilgiris, Kodaikanal", ">2000"): {"rainfall": 2124, "crops": ["Wheat", "Garlic", "Lemon", "Lime", "Pomegranate", "Pineapple", "Beans", "Beetroot", "Cabbage", "Chowchow", "Cotton", "Pepper", "Coffee", "Potato", "Banana", "Mandarin", "Orange", "Pear", "Cardamom", "Cutflowers"]}
}

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/atmosphere")
def atmosphere():
    return render_template("atmosphere.html")
    
@app.route("/land")
def land():
    return render_template("land.html")

@app.route("/get_districts", methods=["GET"])
def get_districts():
    districts = sorted(set([key[0] for key in crop_data.keys()]))
    return jsonify({"districts": districts})

@app.route("/predict/atmosphere", methods=["POST"])
def predict_atmosphere():
    try:
        data = request.json
        features = np.array([[data["temperature"], data["humidity"], data["ph"],
                                data["nitrogen"], data["phosphorus"], data["potassium"],
                                data["rainfall"]]])

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        img = generate_atmosphere_chart(data, prediction)
        return jsonify({"crop": prediction, "chart": img})
    except KeyError as e:
        return jsonify({"error": f"Missing key: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

@app.route("/predict/land", methods=["POST"])
def predict_land():
    try:
        data = request.json
        key = (data["district"], data["altitude"])

        if key in crop_data:
            rainfall = crop_data[key]["rainfall"]
            crops = crop_data[key]["crops"]
            img = generate_land_chart(crops, rainfall)
            return jsonify({"rainfall": rainfall, "crops": ", ".join(crops), "chart": img})
        else:
            return jsonify({"error": "No data available for the selected District & Altitude"}), 400
    except Exception as e:
        app.logger.error(f"Error processing request: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

def generate_atmosphere_chart(data, predicted_crop):
    plt.figure(figsize=(7, 5))
    features = ["Temperature", "Humidity", "pH", "Nitrogen", "Phosphorus", "Potassium", "Rainfall"]
    values = [data["temperature"], data["humidity"], data["ph"],
              data["nitrogen"], data["phosphorus"], data["potassium"],
              data["rainfall"]]
    plt.bar(features, values, color="skyblue")
    plt.xlabel("Parameters")
    plt.ylabel("Values")
    plt.title(f"Atmospheric Conditions for {predicted_crop}")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    return save_chart()

def generate_land_chart(crops, rainfall):
    plt.figure(figsize=(6, 6))
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]
    plt.pie([1] * len(crops), labels=crops, autopct="%1.1f%%", colors=colors[:len(crops)])
    plt.title(f"Recommended Crops (Rainfall: {rainfall}mm)")
    return save_chart()

def save_chart():
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode("utf-8")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
