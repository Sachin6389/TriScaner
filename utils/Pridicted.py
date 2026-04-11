import numpy as np

def predict_model(model, image, class_names, top_k=3, threshold=0.4):
    try:
        preds = model.predict(image, verbose=0)
        

        preds = preds[0]

        # ✅ Detect Binary properly
        is_binary = np.ndim(preds) == 0 or (isinstance(preds, np.ndarray) and preds.shape == (1,))
        

        # =======================
        # ✅ BINARY
        # =======================
        if is_binary:
            prob_class1 = float(preds)  # sigmoid output
            prob_class0 = 1 - prob_class1

            probs = [prob_class0, prob_class1]

            # 🔥 APPLY THRESHOLD FIX (IMPORTANT)
            class_index = 1 if prob_class1 >= threshold else 0
            confidence = probs[class_index]

            all_predictions = {
                class_names[i]: float(round(probs[i] * 100, 2))
                for i in range(len(class_names))
            }

            top_predictions = [
                {
                    "class": class_names[class_index],
                    "confidence": float(round(confidence * 100, 2))
                }
            ]

        # =======================
        # ✅ MULTI-CLASS
        # =======================
        else:
            preds = preds.astype(float)

            class_index = int(np.argmax(preds))
            confidence = float(np.max(preds))

            all_predictions = {
                class_names[i]: float(round(preds[i] * 100, 2))
                for i in range(len(preds))
            }

            top_k = min(top_k, len(preds))
            top_indices = preds.argsort()[-top_k:][::-1]

            top_predictions = [
                {
                    "class": class_names[i],
                    "confidence": float(round(preds[i] * 100, 2))
                }
                for i in top_indices
            ]

        result = {
            "top_prediction": class_names[class_index],
            "confidence": float(round(confidence * 100, 2)),
            "top_k_predictions": top_predictions,
            "all_predictions": all_predictions
        }

        
        return result

    except Exception as e:
        print("❌ Error:", str(e))
        return {"error": str(e)}