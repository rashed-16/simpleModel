import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image



st.set_page_config(
    page_title="Cat vs Dog AI",
    page_icon=":material/pets:",
    layout="centered",
)



st.markdown(
    """
    <style>
    .deco {
        position: fixed;
        font-size: 80px;
        opacity: 0.12;
        z-index: 0;
        pointer-events: none;
        user-select: none;
        animation: floaty 6s ease-in-out infinite;
    }

    @keyframes floaty {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-14px); }
    }

    .deco-1 { top: 6%;    left: 5%;    animation-delay: 0s;    }
    .deco-2 { top: 14%;   right: 6%;   animation-delay: 1s;    }
    .deco-3 { top: 40%;   left: 3%;    animation-delay: 2s;    }
    .deco-4 { top: 46%;   right: 4%;   animation-delay: 0.5s;  }
    .deco-5 { bottom: 8%; left: 7%;    animation-delay: 1.5s;  }
    .deco-6 { bottom: 12%; right: 8%;  animation-delay: 2.5s;  }
    .deco-7 { top: 30%;   left: 13%;   font-size: 40px; animation-delay: 0.3s; }
    .deco-8 { top: 34%;   right: 14%;  font-size: 40px; animation-delay: 0.8s; }

    /* ===== Responsive tweaks ===== */

    [data-testid="stImage"] img {
        max-width: 400px;
        width: 100% !important;
        height: auto !important;
        margin: 0 auto;
        display: block;
        border-radius: 12px;
    }

    @media (max-width: 768px) {
        .deco { font-size: 44px !important; opacity: 0.10; }
        .deco-7, .deco-8 { font-size: 26px !important; }
    }

    @media (max-width: 480px) {
        .deco { font-size: 30px !important; opacity: 0.08; }
        .deco-7, .deco-8 { font-size: 20px !important; }
        h1 { font-size: 26px !important; }
    }
    </style>

    <div class="deco deco-1">🐱</div>
    <div class="deco deco-2">🐶</div>
    <div class="deco deco-3">🐾</div>
    <div class="deco deco-4">🐱</div>
    <div class="deco deco-5">🐶</div>
    <div class="deco deco-6">🐾</div>
    <div class="deco deco-7">🐱</div>
    <div class="deco deco-8">🐶</div>
    """,
    unsafe_allow_html=True,
)



MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cats_vs_dogs")


@st.cache_resource
def load_model():
    """
    Load TensorFlow SavedModel directly.

    The model was trained with an older Keras 2 / TensorFlow
    version, so its SavedModel embeds a Keras 2 optimizer
    object. TensorFlow 2.16+ ships Keras 3, which cannot
    restore those optimizer slot variables and raises:

        AttributeError: '_UserObject' object has no attribute 'add_slot'

    We only need the serving signature for inference, so we
    monkey-patch Trackable.add_slot to create dummy slot
    variables. The optimizer state is unused at serving time.
    """
    from tensorflow.python.trackable.base import Trackable

    if not hasattr(Trackable, "_add_slot_patched"):

        def add_slot(self, var, slot_name, initializer="zeros", shape=None):
            shape = shape if shape is not None else var.shape
            return tf.Variable(
                tf.zeros(shape, dtype=var.dtype),
                name=slot_name
            )

        Trackable.add_slot = add_slot
        Trackable._add_slot_patched = True

    return tf.saved_model.load(MODEL_PATH)


model = load_model()


infer = model.signatures["serving_default"]

input_name = list(infer.structured_input_signature[1].keys())[0]
output_name = list(infer.structured_outputs.keys())[0]




st.title("Cat vs Dog AI", text_alignment="center")

st.caption(
    "Upload an image and let artificial intelligence decide "
    "whether it's a cat or a dog.",
    text_alignment="center",
)

st.space("medium")


uploaded_file = st.file_uploader(
    "Upload your image",
    help="Upload a clear picture of a cat or dog (PNG, JPG, WEBP, GIF, BMP, ...).",
    label_visibility="collapsed",
)




if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error(
            "This file doesn't look like a supported image. "
            "Please upload a picture (PNG, JPG, WEBP, GIF, BMP, ...)."
        )
        st.stop()

    st.space("small")

    with st.container(horizontal_alignment="center"):
        st.image(image, caption="Your uploaded image", use_container_width=True)

    st.space("small")

    if st.button(
        "Let AI guess",
        type="primary",
        icon=":material/auto_awesome:",
        width="stretch",
    ):

        with st.spinner("AI is thinking..."):

          
            img = image.resize((150, 150))

            img_array = np.array(img).astype(np.float32)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

          

            result = infer(**{input_name: tf.constant(img_array)})
            prediction = result[output_name].numpy()[0][0]

         
            if prediction >= 0.5:
                label = "Cat"
                confidence = prediction
                emoji = "🐱"
                message = (
                    "The AI thinks this little friend is a cat!"
                )
            else:
                label = "Dog"
                confidence = 1 - prediction
                emoji = "🐶"
                message = (
                    "The AI thinks this little friend is a dog!"
                )



        st.space("small")

        with st.container(border=True):

            st.subheader(
                f"{emoji} AI Prediction",
                text_alignment="center",
            )

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    label="Result",
                    value=label,
                )

            with metric_col2:
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.2%}",
                )

            st.success(message)




st.space("large")

st.subheader("Features", text_alignment="center")

st.space("small")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    with st.container(border=True):
        st.markdown(":material/psychology:")

        st.markdown("**Smart AI**")

        st.caption(
            "An AI model trained to recognize cats and dogs."
        )

with feature_col2:
    with st.container(border=True):
        st.markdown(":material/bolt:")

        st.markdown("**Fast prediction**")

        st.caption(
            "Upload an image and get the prediction instantly."
        )

with feature_col3:
    with st.container(border=True):
        st.markdown(":material/auto_awesome:")

        st.markdown("**AI magic**")

        st.caption(
            "See how AI can understand images."
        )




st.space("large")

st.caption(
    "Built with TensorFlow & Streamlit — Cat vs Dog AI Classifier",
    text_alignment="center",
)
