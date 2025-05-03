import os                      
import cv2                     
from matplotlib import pyplot as plt
import numpy as np             
import pandas as pd            
import onnx                    
from onnx import hub           
import onnxruntime   
from PIL import Image          

# Define tags to filter models from ONNX Hub
TAGS = ["vision", "classification"]
MODEL_REPO = "./onnx_models"
IMAGE_PATH = "/home/edge/Desktop/data/new_images/br5.jpg"

# Set the cache directory for ONNX Hub downloads to avoid re-downloading models repeatedly
hub.set_dir("./onnx_models/cache")
#print(hub.get_dir())

# Function to list available ONNX models with specified tags and allow user to download them
def list_and_download_model():
    try:
        # Fetch all models from ONNX Hub matching the specified tags
        models_list = hub.list_models(tags=TAGS)
        print(f"Models with TAG: {TAGS}: {len(models_list)}")

        # Save the list of models to a text file for reference (optional)
        with open("./extra/models_list.txt", "w") as output:
            output.write(str(models_list))

        # Interactive loop to display models and let user select which to download
        while True:
            print("\nAvailable ONNX Models:")
            # Loop through all models and print their metadata for user info
            for idx, model_info in enumerate(models_list, start=1):
                # Extract tags and input shape info from model metadata
                tags = model_info.metadata.get('tags', [])
                inputs = model_info.metadata.get('io_ports', {}).get('inputs', [{}])
                input_shape = inputs[0].get('shape', '?') if inputs else '?'

                # Display model name, opset version, tags, and input shape
                print(f"[{idx}] {model_info.model} (opset {model_info.opset})")
                print(f"    Tags: {', '.join(tags)}")
                print(f"    Input: {input_shape}")
                print("-" * 80)
            print("[0] Exit")  # Option to exit the selection loop

            # Prompt user to enter a number corresponding to the model to download
            try:
                choice = int(input("\nEnter the number of the model to download (0 to exit): "))
                if choice == 0:
                    print("Exiting.")
                    break  
                selected = models_list[choice - 1] 
            
            except (ValueError, IndexError):
                print("Invalid selection! Please enter a valid number.")
                continue 

            # Download the selected model from ONNX Hub with force_reload to ensure fresh copy
            print(f"\nDownloading {selected.model} (opset {selected.opset})...")
            model = hub.load(
                model=selected.model,
                opset=selected.opset,
                force_reload=True  # Force re-download even if cached
            )

            # Save the downloaded model to the local MODEL_REPO directory with a descriptive filename
            output_name = f"./onnx_models/{selected.model}-opset{selected.opset}.onnx"
            onnx.save(model, output_name)
            print(f"Model saved to {output_name}\n")

    except Exception as e:
        print(f"An error occurred: {e}")


# Function to load and preprocess the input image according to the ONNX model's input requirements
def load_image(onnx_session, normalize=True, default_size=(224, 224)):
    try:        
        image = cv2.imread(IMAGE_PATH)
        if image is None:
            raise ValueError(f"Failed to load image: {IMAGE_PATH}")

        # Get the first input tensor information from the ONNX model session
        input_info = onnx_session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape  # Expected input shape, e.g., [1, 3, 224, 224]

      
        # N: Number of images in batch (batch size)
        # C: Number of channels (e.g., 3 for RGB)
        # H: Height of image in pixels
        # W: Width of image in pixels
        
        # Handle 3D input shape (C, H, W) - no batch dimension
        if len(input_shape) == 3 and input_shape[0] == 3:
            height = 224 
            width = 224 
            resized_image = cv2.resize(image, (width, height)) 
            input_image = resized_image.astype(np.float32)  # Convert to float32 for model compatibility
            
            if normalize:
                input_image /= 255.0  # Normalize pixel values to [0,1]
            input_image = np.transpose(input_image, (2, 0, 1))  # Convert from HWC to CHW format (channels first)
            # Note: No batch dimension added here as model expects 3D input
            return input_image, input_name

        # Handle 4D input shape (N, C, H, W) or (N, H, W, C)
        elif len(input_shape) == 4:
            # Helper function to safely get dimension size or fallback to default
            def safe_dim(dim, fallback):
                return dim if isinstance(dim, int) and dim > 0 else fallback

            # Check if input format is NCHW (channels first)
            if input_shape[1] == 3:
                height = safe_dim(input_shape[2], default_size[0])  # Get height or default
                width = safe_dim(input_shape[3], default_size[1])   # Get width or default
                resized_image = cv2.resize(image, (width, height))  # Resize image
                input_image = resized_image.astype(np.float32)
                if normalize:
                    input_image /= 255.0  # Normalize pixel values
                input_image = np.transpose(input_image, (2, 0, 1))  # Convert HWC to CHW
                input_image = np.expand_dims(input_image, axis=0)   # Add batch dimension at axis 0
                return input_image, input_name

            # Check if input format is NHWC (channels last)
            elif input_shape[-1] == 3:
                height = safe_dim(input_shape[1], default_size[0])
                width = safe_dim(input_shape[2], default_size[1])
                resized_image = cv2.resize(image, (width, height))
                input_image = resized_image.astype(np.float32)
                if normalize:
                    input_image /= 255.0
                input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension at axis 0
                return input_image, input_name
      
        raise ValueError(f"Unsupported input shape rank: {input_shape}")

    except Exception as e:
        print(f"Image loading failed: {e}")
        return None, None


# Function to display image with model's classification result
def display_image_with_classification(model_name, pred_class, image_path, save_to_file=False):
    try:
        img = Image.open(image_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"Model: {model_name}\nPredicted Class: {pred_class}")
        plt.axis('off')
        if save_to_file:
            plt.savefig(f"./extra/{model_name}_result.png", bbox_inches='tight', dpi=150)
            #plt.show()
        else:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"Could not display image for {model_name}: {e}")


# Function to run inference on all ONNX models found in the MODEL_REPO directory
def run_models():
    try:
        results = []

        # List all ONNX model files in the model repository directory
        model_files = [os.path.join(MODEL_REPO, f) for f in os.listdir(MODEL_REPO) if f.endswith('.onnx')]
        print(f"Found {len(model_files)} ONNX models in {MODEL_REPO}.")

        # Iterate over each model file to run inference
        for model_path in model_files:
            try:
                
                model_name = os.path.basename(model_path)
                print(f"\nRunning model: {model_name}")
                
                                
                # Create an ONNX Runtime inference session for the model
                session = onnxruntime.InferenceSession(model_path)

                # Get the name of the first output tensor
                output_name = session.get_outputs()[0].name

                # Load and preprocess the input image according to model input requirements
                input_image, input_name = load_image(session)
                if input_image is None:                   
                    raise RuntimeError("Input image processing failed.") # Raise error if image preprocessing failed

                # Run inference on the input image
                outputs = session.run([output_name], {input_name: input_image})
                output_array = outputs[0]  # Get the output array from the model

                # If output is 2D (e.g., batch_size x num_classes), apply softmax for classification
                if output_array.ndim == 2:
                    # Compute softmax probabilities in a numerically stable way
                    e_x = np.exp(output_array - np.max(output_array, axis=1, keepdims=True))
                    scores = e_x / np.sum(e_x, axis=1, keepdims=True)
                    pred_class = int(np.argmax(scores, axis=1)[0])  # Predicted class index
                else:
                    # For other output shapes (e.g., detection), just use raw scores
                    scores = output_array
                    pred_class = int(np.argmax(scores))
                    
                    
                    
                # To show image
                display_image_with_classification(model_name, pred_class, IMAGE_PATH, save_to_file=True)

                # Append the results for this model
                results.append({
                    "model": os.path.basename(model_path),          
                    "input_shape": session.get_inputs()[0].shape,   
                    "output_shape": session.get_outputs()[0].shape, 
                    "predicted_class": pred_class,                    
                    "scores": scores.tolist(),                         
                    "error": None                                     
                })

            except Exception as e:
                # Catch and log errors for individual models without stopping the loop
                print(f"Error with model {model_path}: {e}")
                results.append({
                    "model": os.path.basename(model_path),
                    "input_shape": None,
                    "output_shape": None,
                    "predicted_class": None,
                    "scores": None,
                    "error": str(e)  # Store error message for troubleshooting
                })

        # Convert the results list to a Pandas DataFrame for easy viewing and saving
        df = pd.DataFrame(results)
        print("\nResults:")
        print(df)  # Print the DataFrame to console

        # Save the results to a CSV file for later analysis
        df.to_csv('./extra/results.csv', index=False)

    except Exception as e:
        print(f"An error occurred in run_models: {e}")


if __name__ == "__main__":
    
    # Step 1: List available models and allow user to download desired ones
    list_and_download_model()

    # Step 2: Run inference on all downloaded models and save results
    run_models()



