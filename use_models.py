import os
import cv2
import numpy as np
import pandas as pd
import onnx
from onnx import hub
import onnxruntime

# ONNX models
TAGS = ["vision","classification"]
MODEL_REPO = "./onnx_models"
IMAGE_PATH = "/home/edge/Desktop/data/new_images/br5.jpg"

# Setting the cache location 
hub.set_dir("./onnx_models/cache")
print(hub.get_dir())

# Get available models from ONNX hub 
# https://onnx.ai/onnx/repo-docs/Hub.html
def list_and_download_model():
      
    try:  
        # Fetch all available models matching a given "tag"
        models_list = hub.list_models(tags=TAGS)
        print(f"Models with TAG: {TAGS}: {len(models_list)}")
        
        # Export to see file- Can be removed
        with open("./extra/models_list.txt", "w") as output:
            output.write(str(models_list))

        while True:
            # Display models with metadata
            print("\nAvailable ONNX Models:")
            for idx, model_info in enumerate(models_list, start=1):
                tags = model_info.metadata.get('tags', [])
                inputs = model_info.metadata.get('io_ports', {}).get('inputs', [{}])
                input_shape = inputs[0].get('shape', '?') if inputs else '?'
                print(f"[{idx}] {model_info.model} (opset {model_info.opset})")
                print(f"    Tags: {', '.join(tags)}")
                print(f"    Input: {input_shape}")
                print("-" * 80)
            print("[0] Exit")

            # Let user select
            try:
                choice = int(input("\nEnter the number of the model to download (0 to exit): "))
                if choice == 0:
                    print("Exiting.")
                    break
                selected = models_list[choice - 1]
            except (ValueError, IndexError):
                print("Invalid selection! Please enter a valid number.")
                continue

            # Download the selected model
            print(f"\nDownloading {selected.model} (opset {selected.opset})...")
            model = hub.load(
                model=selected.model,
                opset=selected.opset,
                force_reload=True  # Ensure fresh download
            )
            
            # Save to current directory
            output_name = f"./onnx_models/{selected.model}-opset{selected.opset}.onnx"
            onnx.save(model, output_name)
            print(f"Model saved to {output_name}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def load_image(onnx_session, normalize=True, default_size=(224, 224)):
    try:
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            raise ValueError(f"Failed to load image: {IMAGE_PATH}")

        input_info = onnx_session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape  # e.g., [1, 3, 224, 224] or [3, 224, 224]

        # Handle 3D input shape (C, H, W)
        if len(input_shape) == 3 and input_shape[0] == 3:
            height = 224
            width = 224
            resized_image = cv2.resize(image, (width, height))
            input_image = resized_image.astype(np.float32)
            if normalize:
                input_image /= 255.0
            input_image = np.transpose(input_image, (2, 0, 1))  # HWC → CHW
            # Note: do NOT add batch dimension
            return input_image, input_name

        # Handle 4D input shape (N, C, H, W) or (N, H, W, C)
        elif len(input_shape) == 4:
            def safe_dim(dim, fallback):
                return dim if isinstance(dim, int) and dim > 0 else fallback

            if input_shape[1] == 3:  # NCHW
                height = safe_dim(input_shape[2], default_size[0])
                width = safe_dim(input_shape[3], default_size[1])
                resized_image = cv2.resize(image, (width, height))
                input_image = resized_image.astype(np.float32)
                if normalize:
                    input_image /= 255.0
                input_image = np.transpose(input_image, (2, 0, 1))  # HWC → CHW
                input_image = np.expand_dims(input_image, axis=0)  # Add batch dim
                return input_image, input_name

            elif input_shape[-1] == 3:  # NHWC
                height = safe_dim(input_shape[1], default_size[0])
                width = safe_dim(input_shape[2], default_size[1])
                resized_image = cv2.resize(image, (width, height))
                input_image = resized_image.astype(np.float32)
                if normalize:
                    input_image /= 255.0
                input_image = np.expand_dims(input_image, axis=0)
                return input_image, input_name

        raise ValueError(f"Unsupported input shape rank: {input_shape}")

    except Exception as e:
        print(f"Image loading failed: {e}")
        return None, None

def run_models():
    try:
        results = []

        model_files = [os.path.join(MODEL_REPO, f) for f in os.listdir(MODEL_REPO) if f.endswith('.onnx')]
        print(f"Found {len(model_files)} ONNX models in {MODEL_REPO}.")

        for model_path in model_files:
            try:
                print(f"\nRunning model: {model_path}")
                session = onnxruntime.InferenceSession(model_path)
                output_name = session.get_outputs()[0].name

                input_image, input_name = load_image(session)
                if input_image is None:
                    raise RuntimeError("Input image processing failed.")

                outputs = session.run([output_name], {input_name: input_image})
                output_array = outputs[0]

                if output_array.ndim == 2:
                    # Classification: Apply softmax
                    e_x = np.exp(output_array - np.max(output_array, axis=1, keepdims=True))
                    scores = e_x / np.sum(e_x, axis=1, keepdims=True)
                    pred_class = int(np.argmax(scores, axis=1)[0])
                else:
                    # Other tasks (e.g., detection): just use raw scores
                    scores = output_array
                    pred_class = int(np.argmax(scores))

                results.append({
                    "model": os.path.basename(model_path),
                    "input_shape": session.get_inputs()[0].shape,
                    "output_shape": session.get_outputs()[0].shape,
                    "predicted_class": pred_class,
                    "scores": scores.tolist(),
                    "error": None
                })

            except Exception as e:
                print(f"Error with model {model_path}: {e}")
                results.append({
                    "model": os.path.basename(model_path),
                    "input_shape": None,
                    "output_shape": None,
                    "predicted_class": None,
                    "scores": None,
                    "error": str(e)
                })

        # Output results
        df = pd.DataFrame(results)
        print("\nResults:")
        print(df)
        df.to_csv('./extra/results.csv', index=False)
        
    except Exception as e:
        print(f"An error occurred in run_models: {e}")
    try:
        results = []
        y_pred = []

        model_files = [os.path.join(MODEL_REPO, f) for f in os.listdir(MODEL_REPO) if f.endswith('.onnx')]
        print(f"Found {len(model_files)} ONNX models in {MODEL_REPO}.")

        for model_path in model_files:
            try:
                print(f"\nRunning model: {model_path}")
                session = onnxruntime.InferenceSession(model_path)
                output_name = session.get_outputs()[0].name

                input_image, input_name = load_image(session)
                if input_image is None:
                    raise RuntimeError("Input image processing failed.")

                outputs = session.run([output_name], {input_name: input_image})

                output_array = outputs[0]
                if output_array.ndim == 2:
                    e_x = np.exp(output_array - np.max(output_array, axis=1, keepdims=True))
                    scores = e_x / np.sum(e_x, axis=1, keepdims=True)
                    pred_class = int(np.argmax(scores, axis=1)[0])
                else:
                    scores = output_array
                    pred_class = int(np.argmax(scores))

                y_pred.append(pred_class)

                results.append({
                    "model": os.path.basename(model_path),
                    "predicted_class": pred_class,
                    "scores": scores.tolist(),
                    "error": None
                })

            except Exception as e:
                print(f"Error with model {model_path}: {e}")
                results.append({
                    "model": os.path.basename(model_path),
                    "predicted_class": None,
                    "scores": None,
                    "error": str(e)
                })

        df = pd.DataFrame(results)
        print("\nResults:")
        print(df)

        df.to_csv('./extra/results.csv', index=False)

    except Exception as e:
        print(f"An error occurred in run_models: {e}")

if __name__ == "__main__":
    
    # List and download models
    list_and_download_model()            
    
    # Run loaded models
    run_models()
