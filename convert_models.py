# Exporting a YOLO model to ONNX format https://docs.ultralytics.com/integrations/onnx/
from matplotlib import pyplot as plt
import numpy as np

import torch
from ultralytics import YOLO
import onnx
import onnxruntime as ort

from PIL import Image
from torchvision.transforms import functional as F
import torch.nn.functional as F_torch  # For softmax

#IMAGE_PATH = "/home/edge/Desktop/data/train/healthy/healthy_train.0.jpg"
#IMAGE_PATH = "/home/edge/Desktop/data/train/angular_leaf_spot/angular_leaf_spot_train.2.jpg"
#IMAGE_PATH = "/home/edge/Desktop/data/test/bean_rust/bean_rust_val.1.jpg"
IMAGE_PATH = "/home/edge/Desktop/data/new_images/h5.jpg"


# Instantiate the model
# TODO: Estou assumindo que o modelo original eh YOLOv8n, mas pode ser outro modelo. Verificar isso.
model = YOLO('./ref_model/best_model.pt')

# Export the model to ONNX format see arguments https://docs.ultralytics.com/integrations/onnx/#export-arguments
model.export(format="onnx", imgsz=240, task='classify') 

#print(model.eval())


# Load the exported ONNX model for validation
try:     
    #onnx_model = YOLO("./original_models/best_model.onnx")
    model = onnx.load("./ref_model/best_model.onnx")
    input_shape = model.graph.input[0].type.tensor_type.shape
    print(f"Input shape: {[dim.dim_value for dim in input_shape.dim]}")  # e.g., [1, 3, 240, 240]
    print(f"Model expects input spatial size: {[dim.dim_value for dim in input_shape.dim[2:]]}")  # [240, 240]
    
    # Loads the ONNX model file into a Python object for inspection, validation, or editing.
    onnx.checker.check_model(model)  # Basic structural validation
    
except Exception as e:
    print(f"Error loading ONNX model: {e}")


# Run inference
try: 
    
    # Preprocess image
    image = Image.open(IMAGE_PATH).convert("RGB")
    image = F.resize(image, (240, 240))
    input_array = np.array(image).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    input_array /= 255.0  # Normalize to [0, 1]

    # Inference session
    # Loads the ONNX model into ONNX Runtime for inference
    session = ort.InferenceSession("./ref_model/best_model.onnx")
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # Run inference
    results = session.run(output_names, {input_name: input_array})
    print(f"Output shapes: {[r.shape for r in results]}")

    logits = results[0]  # Should be (1, 3)
    probabilities = F_torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predicted_class = np.argmax(probabilities, axis=1)[0]
    confidence = probabilities[0, predicted_class]

    # Class names
    class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
    print(f"Predicted class: {class_names[predicted_class]} (confidence: {confidence:.2f})")

    # Display image with prediction
    plt.imshow(image)
    plt.title(f"Predicted: {class_names[predicted_class]} (conf: {confidence:.2f})")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"Error running ONNX model: {e}")

