import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import gc

from utils.Xrayprocessing import preprocess_image
from utils.Pridicted import predict_model

# ================= CONFIG =================
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= INIT =================
app = Flask(__name__)
CORS(app)

print("🚀 Starting Medical AI API...")

# ================= HELPERS =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_safe(path):
    try:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return None

        model = tf.keras.models.load_model(path, compile=False )
        print(f"✅ Loaded: {path}")
        return model

    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None


# ================= LOAD MODELS =================
print("📦 Loading models...")

MODELS = {
    "brain": {
        "path": "Models/mri_brain_model_final.keras",
        "model": None,
        "classes": ["Alzehaimer", "Glioma-Tumor ", "Meningioma-Tumor", "Multiple Sclerosis ", "Normal", "Pituitary-Tumor"],
        "img_size": (300, 300)
    },
    "chest": {
        "path": "Models/chest_ct_cancer_model.keras",
        "model": None,
        "classes": ["adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib ", "normal", "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa "],
        "img_size": (256, 256)
    },
    "kidney": {
        "path": "Models/final_model_25.keras",
        "model": None,
        "classes": ["Non-Stone", "Stone"],
        "img_size": (300, 300)
    },
    "bone": {
        "path": "Models/bone_2_fracture_model.keras",
        "model": None,
        "classes": ["Fractured", "Not Fractured"],
        "img_size": (300, 300)
    },
    "brainStroke": {
        "path": "Models/Brain_Stroke.keras",
        "model": None,
        "classes": ["Bleeding", "Ischemia", "Normal "],
        "img_size": (300, 300)
    }
}
CURRENT_MODEL = {"name": None, "model": None}

def get_model(model_name):
    global CURRENT_MODEL

    # If same model → reuse
    if CURRENT_MODEL["name"] == model_name:
        return CURRENT_MODEL["model"]

    # Free old model
    if CURRENT_MODEL["model"] is not None:
        del CURRENT_MODEL["model"]
        gc.collect()

    model_data = MODELS[model_name]
    model = load_model_safe(model_data["path"])

    CURRENT_MODEL["name"] = model_name
    CURRENT_MODEL["model"] = model

    return model



# ================= MEDICAL INFO =================
MEDICAL_INFO = {
    
  "bone": {
    "Fractured": {
      "symptoms": [
        "Severe pain",
        "Swelling",
        "Difficulty moving limb",
        "Bruising",
        "Deformity of limb",
        "Tenderness",
        "Inability to bear weight",
        "Crack sound at injury",
        "Muscle spasms",
        "Limited range of motion"
      ],
      "hospitals": [
        "AIIMS Delhi",
        "Apollo Hospitals",
        "Fortis Healthcare",
        "Max Healthcare",
        "Manipal Hospitals",
        "Medanta - The Medicity",
        "Kokilaben Dhirubhai Ambani Hospital",
        "Narayana Health",
        "Artemis Hospital",
        "Sir Ganga Ram Hospital"
      ]
    },
    "Not Fractured": {
      "symptoms": [
        "Mild pain",
        "Minor swelling",
        "Soft tissue injury",
        "Bruise without bone damage"
      ],
      "hospitals": []
    }
  },

  "kidney": {
    "Stone": {
      "symptoms": [
        "Severe abdominal pain",
        "Blood in urine",
        "Nausea",
        "Vomiting",
        "Frequent urination",
        "Burning sensation while urinating",
        "Cloudy urine",
        "Back pain",
        "Pain radiating to groin",
        "Fever with chills"
      ],
      "hospitals": [
        "AIIMS Delhi",
        "Manipal Hospitals",
        "Apollo Hospitals",
        "Fortis Healthcare",
        "Max Healthcare",
        "Medanta - The Medicity",
        "Narayana Health",
        "Kokilaben Hospital",
        "Artemis Hospital",
        "Sir Ganga Ram Hospital"
      ]
    },
    "Non-Stone": {
      "symptoms": [
        "Normal urination",
        "No abdominal pain",
        "Healthy kidney function"
      ],
      "hospitals": []
    }
  },

  "brainStroke": {
    "Bleeding": {
      "symptoms": [
        "Severe headache",
        "Vomiting",
        "Weakness",
        "Loss of consciousness",
        "Blurred vision",
        "Seizures",
        "Confusion",
        "Sudden numbness",
        "Difficulty speaking",
        "Loss of balance"
      ],
      "hospitals": [
        "AIIMS Delhi",
        "Fortis Healthcare",
        "Apollo Hospitals",
        "Max Healthcare",
        "Medanta - The Medicity",
        "NIMHANS",
        "Kokilaben Hospital",
        "Narayana Health",
        "Artemis Hospital",
        "Sir Ganga Ram Hospital"
      ]
    },
    "Ischemia": {
      "symptoms": [
        "Numbness",
        "Speech difficulty",
        "Weakness on one side",
        "Vision loss",
        "Dizziness",
        "Loss of coordination",
        "Confusion",
        "Difficulty understanding speech",
        "Facial drooping",
        "Sudden severe headache"
      ],
      "hospitals": [
        "Apollo Hospitals",
        "Max Healthcare",
        "AIIMS Delhi",
        "Fortis Healthcare",
        "Medanta - The Medicity",
        "NIMHANS",
        "Narayana Health",
        "Kokilaben Hospital",
        "Artemis Hospital",
        "Sir Ganga Ram Hospital"
      ]
    },
    "Normal": {
      "symptoms": [
        "No neurological deficit",
        "Normal speech",
        "No weakness",
        "Clear vision"
      ],
      "hospitals": []
    }
  },

  "chest": {
    "Normal": {
      "symptoms": [
        "Normal breathing",
        "No chest pain",
        "No cough",
        "Clear X-ray"
      ],
      "hospitals": []
    },
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": {
      "symptoms": [
        "Persistent cough",
        "Chest pain",
        "Weight loss",
        "Shortness of breath",
        "Fatigue",
        "Loss of appetite",
        "Coughing blood",
        "Hoarseness",
        "Frequent infections",
        "Shoulder pain"
      ],
      "hospitals": [
        "AIIMS Delhi",
        "Tata Memorial Hospital",
        "Apollo Hospitals",
        "Fortis Healthcare",
        "Max Healthcare",
        "Medanta - The Medicity",
        "Narayana Health",
        "Kokilaben Hospital",
        "Artemis Hospital",
        "HCG Cancer Centre"
      ]
    },
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": {
      "symptoms": [
        "Coughing blood",
        "Shortness of breath",
        "Chest pain",
        "Fatigue",
        "Weight loss",
        "Persistent cough",
        "Wheezing",
        "Loss of appetite",
        "Hoarseness",
        "Frequent lung infections"
      ],
      "hospitals": [
        "Apollo Hospitals",
        "Fortis Healthcare",
        "AIIMS Delhi",
        "Tata Memorial Hospital",
        "Max Healthcare",
        "Medanta - The Medicity",
        "Narayana Health",
        "Kokilaben Hospital",
        "Artemis Hospital",
        "HCG Cancer Centre"
      ]
    }
  },

  "brain": {
    "Normal": {
      "symptoms": [
        "Normal cognitive function",
        "No memory loss",
        "Stable mood",
        "Clear thinking"
      ],
      "hospitals": []
    },
    "Alzehaimer": {
      "symptoms": [
        "Memory loss",
        "Confusion",
        "Difficulty planning",
        "Mood changes",
        "Difficulty speaking",
        "Disorientation",
        "Poor judgment",
        "Misplacing items",
        "Difficulty recognizing people",
        "Behavioral changes"
      ],
      "hospitals": [
        "AIIMS Delhi",
        "NIMHANS",
        "Apollo Hospitals",
        "Fortis Healthcare",
        "Max Healthcare",
        "Medanta - The Medicity",
        "Narayana Health",
        "Kokilaben Hospital",
        "Artemis Hospital",
        "Sir Ganga Ram Hospital"
      ]
    },
    "Glioma-Tumor": {
      "symptoms": [
        "Headache",
        "Seizures",
        "Nausea",
        "Vomiting",
        "Vision problems",
        "Memory issues",
        "Personality changes",
        "Difficulty speaking",
        "Weakness",
        "Balance problems"
      ],
      "hospitals": [
        "AIIMS Delhi",
        "Apollo Hospitals",
        "Fortis Healthcare",
        "Max Healthcare",
        "Medanta - The Medicity",
        "NIMHANS",
        "Narayana Health",
        "Kokilaben Hospital",
        "Artemis Hospital",
        "Tata Memorial Hospital"
      ]
    },
    "Meningioma-Tumor": {
      "symptoms": [
        "Vision problems",
        "Headache",
        "Hearing loss",
        "Memory issues",
        "Seizures",
        "Weakness",
        "Difficulty concentrating",
        "Personality changes",
        "Numbness",
        "Speech difficulty"
      ],
      "hospitals": [
        "Fortis Healthcare",
        "AIIMS Delhi",
        "Apollo Hospitals",
        "Max Healthcare",
        "Medanta - The Medicity",
        "NIMHANS",
        "Narayana Health",
        "Kokilaben Hospital",
        "Artemis Hospital",
        "Sir Ganga Ram Hospital"
      ]
    },
    "Multiple Sclerosis": {
      "symptoms": [
        "Numbness",
        "Weakness",
        "Vision problems",
        "Fatigue",
        "Difficulty walking",
        "Muscle spasms",
        "Balance issues",
        "Tingling sensation",
        "Dizziness",
        "Bladder problems"
      ],
      "hospitals": [
        "NIMHANS",
        "AIIMS Delhi",
        "Apollo Hospitals",
        "Fortis Healthcare",
        "Max Healthcare",
        "Medanta - The Medicity",
        "Narayana Health",
        "Kokilaben Hospital",
        "Artemis Hospital",
        "Sir Ganga Ram Hospital"
      ]
    },
    "Pituitary-Tumor": {
      "symptoms": [
        "Hormonal imbalance",
        "Vision issues",
        "Headache",
        "Fatigue",
        "Unexplained weight gain",
        "Irregular periods",
        "Growth abnormalities",
        "Mood changes",
        "Nausea",
        "Loss of libido"
      ],
      "hospitals": [
        "AIIMS Delhi",
        "Apollo Hospitals",
        "Fortis Healthcare",
        "Max Healthcare",
        "Medanta - The Medicity",
        "NIMHANS",
        "Narayana Health",
        "Kokilaben Hospital",
        "Artemis Hospital",
        "Tata Memorial Hospital"
      ]
    }
  }
}


# ================= ROUTES =================
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Medical AI API 🚀"
    })


# ================= PREDICT =================
@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):

    if model_name not in MODELS:
        return jsonify({"error": "Invalid model name"}), 400

    model_data = MODELS[model_name]
    model = get_model(model_name)

    if model_data["model"] is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        file.save(file_path)

        img = preprocess_image(file_path, model_data["img_size"])

        if img is None:
            return jsonify({"error": "Image processing failed"}), 400

        result = predict_model(
            model=model_data["model"],
            image=img,
            class_names=model_data["classes"]
        )

        if "error" in result:
            return jsonify(result), 500

        predicted_class = result["top_prediction"]
        confidence = float(result["confidence"])

        info = MEDICAL_INFO.get(model_name, {}).get(predicted_class, {})

        return jsonify({
            "model": model_name,
            "top_prediction": predicted_class,
            "confidence": confidence,
            "top_k_predictions": result.get("top_k_predictions", []),
            "all_predictions": result.get("all_predictions", {}),
            "symptoms": info.get("symptoms", []),
            "hospitals": info.get("hospitals", [])
        })

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ================= HEALTH =================
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": {
            name: model["model"] is not None
            for name, model in MODELS.items()
        }
    })


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)