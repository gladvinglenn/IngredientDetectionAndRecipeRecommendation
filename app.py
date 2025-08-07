import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from recommender import IndianRecipeRecommender

# --- PAGE SETUP ---
st.set_page_config(page_title="DISHCOVERY", layout="wide")

# --- CUSTOM BACKGROUND & STYLING ---
page_bg = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@400;700&display=swap');

/* Background image with gradient overlay */
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(135deg, 
                      rgba(248, 217, 229, 0.5), 
                      rgba(246, 242, 216, 0.5), 
                      rgba(217, 232, 248, 0.5)),
                      url('https://img.freepik.com/free-photo/tortillas-mexican-food-tablecloth_23-2147740821.jpg?ga=GA1.1.722201503.1745704029&semt=ais_hybrid&w=740');
    background-size: cover;
    background-position: center;
    font-family: 'Lato', sans-serif;
}

/* Title and tagline styling */
.main-title {
    text-align: center;
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    font-weight: 700;
    color: #4A3F55;
    margin-bottom: 0.3rem;
    text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.5);
}

.subtitle {
    text-align: center;
    font-family: 'Lato', sans-serif;
    font-size: 3rem;
    font-weight: 400;
    color: #6B5B7A;
    margin-bottom: 2.5rem;
    text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.5);
}

/* Recipe content styling */
.recipe-content {
    margin-bottom: 1.5rem;
    padding: 1rem 0;
    background-color: rgba(255, 255, 255, 0.7);
    border-radius: 15px;
    padding: 1.5rem;
}

/* Recipe divider styling */
.recipe-divider {
    border: 0;
    height: 2px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(106, 90, 122, 0.75), rgba(0, 0, 0, 0));
    margin: 2rem 0;
}

/* Improve input field visibility */
[data-testid="stTextInput"] input {
    background-color: rgba(255, 255, 255, 0.8) !important;
    border-radius: 10px;
}

/* Remove slider card block */
[data-testid="stSlider"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Center button styling */
div[data-testid="stHorizontalBlock"] {
    display: flex;
    justify-content: center;
}

/* Improve warning visibility */
.st-emotion-cache-16idsys p {
    color: #663c00 !important;
    font-weight: bold !important;
    background-color: rgba(255, 237, 160, 0.9) !important;
    padding: 0.5rem !important;
    border-radius: 5px !important;
}

/* Improve info visibility */
.st-emotion-cache-r421ms p {
    color: #2c5282 !important;
    font-weight: bold !important;
    background-color: rgba(235, 248, 255, 0.9) !important;
    padding: 0.5rem !important;
    border-radius: 5px !important;
}

/* Make instructions more readable */
.recipe-content ol, .recipe-content ul {
    background-color: rgba(255, 255, 255, 0.8);
    padding: 1rem 1rem 1rem 2rem;
    border-radius: 5px;
}

/* Recipe selection button styling */
.recipe-button {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 10px;
    margin: 5px 0;
    transition: all 0.3s ease;
}

.recipe-button:hover {
    background-color: rgba(235, 248, 255, 0.9);
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'selected_recipe' not in st.session_state:
    st.session_state.selected_recipe = None

# --- LOAD MODELS ---
@st.cache_resource
def load_detection_model(model_path='trained_model.h5', labels_path='Labels.txt'):
    model = tf.keras.models.load_model(model_path)
    with open(labels_path) as f:
        labels = [l.strip() for l in f]
    return model, labels

@st.cache_resource
def load_recommender(recipes_path='cleaned_indian_recipes.csv'):
    return IndianRecipeRecommender(recipes_path)

# Initialize models (stop on error)
try:
    detection_model, labels = load_detection_model()
    recommender = load_recommender()
except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()

# --- HEADER ---
st.markdown("<h1 class='main-title'>DISHCOVERY!!</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Your Culinary Adventure Starts Here!</p>", unsafe_allow_html=True)
st.markdown("---")

# --- STEP 1: MULTIPLE IMAGE UPLOAD ---
st.subheader("Step 1: Upload Ingredient Images")
image_files = st.file_uploader(
    "Upload images of your ingredients (you can select multiple)",
    type=["jpg","jpeg","png"], accept_multiple_files=True
)
detected_original = []
detected_normalized = []
if image_files:
    cols = st.columns(len(image_files))
    for i, img_file in enumerate(image_files):
        img = Image.open(img_file).convert('RGB')
        cols[i].image(img, caption=f'Image {i+1}', width=150)
        arr = np.expand_dims(np.array(img.resize((64,64))), axis=0)
        pred = detection_model.predict(arr)
        label = labels[np.argmax(pred)]
        detected_original.append(label)
        detected_normalized.append(label.strip().lower())
    # dedupe
    seen = set()
    detected_normalized = [x for x in detected_normalized if not (x in seen or seen.add(x))]
    seen_orig = set()
    detected_original = [x for x in detected_original if not (x in seen_orig or seen_orig.add(x))]
    st.success(f"Detected Ingredients: {', '.join(detected_original)}")

# --- STEP 2: INGREDIENT INPUT (FALLBACK) ---
if detected_normalized:
    ingredients_list = detected_normalized
else:
    user_input = st.text_input(
        "Or enter ingredients manually (comma-separated)",
        placeholder="e.g., rice, tomato, onion"
    )
    ingredients_list = [x.strip().lower() for x in user_input.split(",") if x.strip()]

# --- SELECT NUMBER OF RECIPES ---
top_n = st.slider("How many recipes to show?", 1, 10, 5)

# --- ACTION BUTTON ---
_, middle, _ = st.columns(3)
with middle:
    go = st.button("Start My Culinary Trip ✈️", use_container_width=True)

# --- RECOMMENDATION LOGIC ---
if go:
    if not ingredients_list:
        st.warning("Please upload images or enter at least one ingredient.")
    else:
        st.session_state.selected_recipe = None
        with st.spinner('✈️ Packing your bags...'):
            st.session_state.results = recommender.recommend_recipes(
                ingredients_list, top_n=top_n
            )

# --- DISPLAY RESULTS ---
if st.session_state.results:
    st.success(f"🍴 Found {len(st.session_state.results)} recipes!")
    titles = [f"🍽️ {r['title']}" for r in st.session_state.results]
    idx = st.radio(
        "Choose a recipe:", options=range(len(titles)),
        format_func=lambda i: titles[i], key="sel_radio"
    )
    st.session_state.selected_recipe = idx
    if idx is not None:
        st.markdown("<hr class='recipe-divider'>", unsafe_allow_html=True)
        r = st.session_state.results[idx]
        st.markdown(f"<div class='recipe-content'>", unsafe_allow_html=True)
        st.subheader(r['title'])
        st.write(f"**Confidence:** {r['confidence']}%")
        st.write(f"**Ingredients:** {r['ingredients']}")
        if r['missing_ingredients']:
            miss = ', '.join(r['missing_ingredients'])
            st.markdown(
                f"<div style='background-color: rgba(255,220,120,0.95); color:#663c00; padding:10px; border-radius:5px; margin:10px 0; font-weight:bold;'>⚠️ Missing: {miss}</div>",
                unsafe_allow_html=True
            )
        else:
            st.info("You have all required ingredients! 🎉")
        st.markdown("**Instructions:**")
        for i, step in enumerate(r['instructions'], 1):
            st.write(f"{i}. {step}")
        st.markdown("</div>", unsafe_allow_html=True)
