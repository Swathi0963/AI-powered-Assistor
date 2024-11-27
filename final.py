import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import cv2
import numpy as np

# Feature One: Text Extraction and TTS
def feature_one():
    # Function to extract text from the uploaded image
    def extract_text_from_image(image):
        try:
            extracted_text = pytesseract.image_to_string(image, lang='eng')  # Using Tesseract OCR
            return extracted_text.strip()
        except Exception as e:
            return f"Error in OCR process: {e}"

    # Function to convert text to speech
    def convert_text_to_speech(text, language='en', rate=150, voice_index=0):
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if voice_index < len(voices):
                engine.setProperty('voice', voices[voice_index].id)
            engine.setProperty('rate', rate)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error in TTS process: {e}")

    st.title("Text-to-Speech Conversion for Visual Content")
    st.markdown("Upload an image to extract text and listen to it being read aloud.")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract text from image
        st.write("Extracting text from the image...")
        extracted_text = extract_text_from_image(image)
        
        if extracted_text:
            # Display extracted text
            st.write("### Extracted Text:")
            st.write(extracted_text)

            # Option to read text aloud
            if st.button("Read Text Aloud"):
                convert_text_to_speech(extracted_text)
        else:
            st.write("No text found in the image.")

# Feature Two: Object Detection using YOLO
def feature_two():
    # Custom CSS for the app styling
    st.markdown("""
        <style>
            .stApp {
                background-color: white;
            }
            .stTitle {
                font-size: 10px;
                color: #4CAF50;
            }
            .stMarkdown {
                font-size: 18px;
                color: #555555;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                border-radius: 8px;
                padding: 10px 20px;
            }
            .stFileUploader>div {
                background-color: #ffffff;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }
            .stImage>img {
                border-radius: 8px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }
                
  
    

            </style>
    """, unsafe_allow_html=True)

    # Load YOLO model
    def load_yolo():
        net = cv2.dnn.readNet("C://GenAI_app//yolov3.weights", "C://GenAI_app//yolov3.cfg")  # Load YOLOv3 weights and config file
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, output_layers

    # Load class names (COCO dataset)
    def load_classes():
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    # Process image and detect objects
    def detect_objects(image, net, output_layers, classes):
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Filter weak detections
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    # Draw bounding boxes and labels
    def draw_boxes(image, boxes, confidences, class_ids, classes):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_SIMPLEX

        detected_objects = set()  # To store unique objects

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Green for obstacle
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), font, 0.6, color, 2)

                # Add the detected object to the set
                detected_objects.add(label)

        return image, detected_objects

    st.title("Object and Obstacle Detection for Safe Navigation")
    st.write("Upload an image to detect objects.")
    
    # Load YOLO model and classes
    net, output_layers = load_yolo()
    classes = load_classes()

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        
        # Convert RGBA to RGB if the image has an alpha channel
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Convert PIL image to OpenCV image (BGR format)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detect objects
        boxes, confidences, class_ids = detect_objects(image, net, output_layers, classes)

        # Draw bounding boxes on the detected objects
        image_with_boxes, detected_objects = draw_boxes(image, boxes, confidences, class_ids, classes)

        # Convert image to RGB for display in Streamlit
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

        # Display the result
        st.image(image_with_boxes, caption='Processed Image', use_column_width=True)

        # Show detected objects
        st.markdown(f"### Detected Objects: {', '.join(detected_objects)}", unsafe_allow_html=True)

# Main function to manage navigation
def main():
    st.title("AI Powered Solution for Assisting Visually Impaired Individuals")
    
    # Create a radio button for navigation
    feature = st.radio("Choose a Feature", ("Text-to-Speech Conversion for Visual Content", "Object and Obstacle Detection for Safe Navigation"))
    
    if feature == "Text-to-Speech Conversion for Visual Content":
        feature_one()
    elif feature == "Object and Obstacle Detection for Safe Navigation":
        feature_two()

# Run the Streamlit app
if __name__ == "__main__":
    main()
