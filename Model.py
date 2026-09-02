import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter

@st.cache_resource
def get_model():
    return YOLO("yolov8n.pt")

model = get_model()

st.title("YOLO detection model")
file = st.file_uploader("Upload an image",type = ["jpg","jpeg","png","webp"])

confidence_threshold = st.slider(
    "Select Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    step=0.05,
    value = 0.25
)

if file:
    image = Image.open(file)
    st.image(image = image, caption= "Uploaded_Image", use_container_width= True)

    with st.spinner("Detecting . . . . . . ."):
        results = model.predict(source= image, conf = confidence_threshold)
        result = results[0]

        new_image = result.plot()
        st.image(image = new_image, channels= "BGR", caption="Image with Detected Objects", use_container_width=True )

        boxes = result.boxes
        class_names = model.names
        detected_classes = []

        c1 , c2 = st.columns(2)

        with c1:
            st.subheader("Detection and Scores")
            if not boxes:
                st.write("No objects detected")
            else:
                for box in boxes:
                    cls_ID = int(box.cls[0].item())
                    box_conf = float(box.conf[0].item())
                    label = class_names[cls_ID].capitalize()
                    detected_classes.append(label)

                    st.write(f"**{label}**: {box_conf: .2f}")

        with c2:
            st.subheader("Object Summary")
            if detected_classes:
                counts = Counter(detected_classes)
                for obj , count in counts.items():
                    st.write(f"{obj}: {count}")
