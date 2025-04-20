import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import openai
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Hairstyle Recommender", layout="centered")
st.title("💇‍♀️ AI Hairstyle Recommendation")

st.write("Upload a clear photo of your face to get hairstyle suggestions based on your face shape and gender.")

# Step 1: Select Gender
gender = st.selectbox("Select Your Gender", ["Male", "Female"])

# Step 2: Upload Image
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

# Step 3: Face Shape Detection Function

def detect_face_shape(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return "Unknown"

        landmarks = results.multi_face_landmarks[0].landmark

        def get_landmark_coords(idx):
            h, w = image.shape[:2]
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            return np.array([x, y])

        # Key measurements
        jaw = get_landmark_coords(152)[1] - get_landmark_coords(10)[1]       # Chin to forehead (length)
        cheekbone = get_landmark_coords(454)[0] - get_landmark_coords(234)[0]  # Cheekbone width
        jawline = get_landmark_coords(447)[0] - get_landmark_coords(127)[0]    # Jaw width
        forehead = get_landmark_coords(71)[0] - get_landmark_coords(301)[0]    # Forehead width

        face_length = abs(jaw)
        widths = [cheekbone, jawline, forehead]
        avg_width = sum(map(abs, widths)) / len(widths)

        length_to_width = face_length / avg_width

        # Simple heuristic rules
        if length_to_width > 1.6:
            if forehead > jawline:
                return "Heart"
            else:
                return "Oblong"
        elif 1.4 < length_to_width <= 1.6:
            return "Oval"
        elif 1.1 < length_to_width <= 1.4:
            if abs(cheekbone - jawline) < 15:
                return "Round"
            elif cheekbone > jawline:
                return "Diamond"
        elif length_to_width <= 1.1:
            if jawline > cheekbone:
                return "Triangle"
            else:
                return "Square"
        return "Unknown"
# Step 4: Recommendation Engine
def recommend_hairstyles(face_shape, gender, hair_type=None):
    hairstyles = {
        "Male": {
            "Oval": ["Pompadour", "Crew Cut", "Faux Hawk", "Side Part"],
            "Round": ["Undercut", "Quiff", "Textured Crop", "High Fade"],
            "Square": ["Buzz Cut", "Side Part", "Classic Taper", "Spiky Hair"],
            "Heart": ["Comb Over", "Taper Fade", "Textured Top"],
            "Diamond": ["Fringe", "Shag", "Messy Crop"],
            "Unknown": ["Try uploading a clearer photo to detect your face shape."]
        },
        "Female": {
            "Oval": ["Layered Bob", "Wavy Lob", "Curtain Bangs", "Long Layers"],
            "Round": ["Side-swept Pixie", "Angled Bob", "High Ponytail", "Long Waves"],
            "Square": ["Textured Layers", "Side Bangs", "Soft Waves", "Feathered Cut"],
            "Heart": ["Choppy Layers", "Side Swept Bangs", "Voluminous Waves"],
            "Diamond": ["Side-Parted Styles", "Long Layers", "Fringes"],
            "Unknown": ["Try uploading a clearer photo to detect your face shape."]
        }
    }

    gender_styles = hairstyles.get(gender)
    if not gender_styles:
        return ["Invalid gender selected. Please choose Male or Female."]

    face_styles = gender_styles.get(face_shape)
    if not face_styles:
        return ["We don't have hairstyles for that face shape yet."]

    # Optionally refine by hair type in future
    # if hair_type:
    #     # Add logic to filter based on hair type

    return face_styles

# Step 5: Handle Upload
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Your Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing your face..."):
        shape = detect_face_shape(image)

    if shape:
        st.success(f"Detected Face Shape: **{shape}**")
        suggestions = recommend_hairstyles(shape, gender)
        st.write("💡 Recommended Hairstyles:")
        for style in suggestions:
            st.markdown(f"- {style}")
    else:
        st.error("No face detected. Please upload a clearer image.")




# Define a function to simulate the chatbot's responses
def get_chatbot_response(user_message):
    # Simple logic for haircuts-related questions
    responses = {
        "what are the best haircuts for men?": "Some popular men's haircuts are the buzz cut, fade, pompadour, quiff, and crew cut.",
        "what are the best haircuts for women?": "Popular women's haircuts include the bob, pixie cut, lob, shaggy cut, and layers.",
        "how to choose a haircut?": "Consider your face shape, hair texture, and lifestyle when choosing a haircut.",
        "what haircut is good for round faces?": "For round faces, go for haircuts with volume at the top, like a layered bob or long layered cuts.",
        "what haircut is good for oval faces?": "Oval faces are versatile, so you can try almost any haircut, like a pixie or a sleek bob.",
        "what is a fade haircut?": "A fade haircut gradually shortens the hair from the top of the head down to the sides, creating a blended look.",
        "what is a pompadour?": "A pompadour involves styling the hair high at the front, creating a voluminous look.",
        "how to take care of curly hair?": "For curly hair, use a sulfate-free shampoo, deep condition regularly, and avoid heat styling.",
        "how to take care of straight hair?": "For straight hair, use a good conditioner to keep it shiny, and avoid over-washing.",
	"is adel gay?":"Yes i think."
    }
    
    # Return a response based on user input
    return responses.get(user_message.lower(), "I'm sorry, I don't understand that question. Please try asking about haircuts.")

# Streamlit app layout
st.title("Haircut Chatbot")
st.write("Ask me about different haircuts! You can ask questions like:")

# Display the questions user can ask
questions = [
    "What are the best haircuts for men?",
    "What are the best haircuts for women?",
    "How to choose a haircut?",
    "What haircut is good for round faces?",
    "What haircut is good for oval faces?",
    "What is a fade haircut?",
    "What is a pompadour?",
    "How to take care of curly hair?",
    "How to take care of straight hair?"
]
st.write("\n".join(questions))

# User input
user_message = st.text_input("Ask a question about haircuts:")

# Get and display chatbot response
if user_message:
    bot_response = get_chatbot_response(user_message)
    st.write(f"Chatbot: {bot_response}")
