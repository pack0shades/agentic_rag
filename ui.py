import streamlit as st
from PIL import Image
import os
import tempfile
import pandas as pd


def main():
    # Page configuration
    st.set_page_config(
        page_title="Pathway Demo",
        layout="wide"
    )

    # Initialize session state variables if they don't exist
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'file_path' not in st.session_state:
        st.session_state.file_path = None
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None
    if 'show_gt' not in st.session_state:
        st.session_state.show_gt = False

    # Custom CSS for styling
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #0066cc;
            color: white;
        }
        .delete-button > button {
            background-color: #ff4b4b;
            color: white;
        }
        .submit-button > button {
            width: 100%;
        }
        .title-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: -70px;
        }
        .logo-container {
            position: absolute;
            left: 20px;
            top: 20px;
        }
        .title-text {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #0066cc;
        }
        .gray-background {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Logo container (left corner)
    st.markdown("""
        <div class="logo-container">
            <img src="data:image/png;base64,{}" width="100">
        </div>
    """.format(get_base64_logo()), unsafe_allow_html=True)

    # Add some space after the title
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Centered title with cloud emoji
    st.markdown("""
        <div class="title-container">
            <h1 class="title-text">☁️ Chat with data</h1>
        </div>
    """, unsafe_allow_html=True)

    # Add some space after the title
    st.markdown("<br><br>", unsafe_allow_html=True)

    # File upload section
    st.subheader("Select File Source")

    # File source buttons
    col3, col4 = st.columns([1, 4])
    with col3:
        st.button("Google Drive", key="gdrive", disabled=True)
    with col4:
        st.button("SP", key="sp", disabled=True)

    # File upload and display section
    if not st.session_state.file_uploaded:
        uploaded_file = st.file_uploader(
            "Choose a file", type=['csv', 'txt', 'pdf'])
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.file_path = tmp_file.name
                st.session_state.file_name = uploaded_file.name
            st.session_state.file_uploaded = True
            st.rerun()

    # Display current file and delete option
    if st.session_state.file_uploaded:
        col_file, col_delete = st.columns([4, 1])
        with col_file:
            st.info(f"Current file: {st.session_state.file_name}")
        with col_delete:
            st.markdown('<div class="delete-button">', unsafe_allow_html=True)
            if st.button("Delete File", key="delete_file"):
                if st.session_state.file_path and os.path.exists(st.session_state.file_path):
                    os.remove(st.session_state.file_path)
                st.session_state.file_uploaded = False
                st.session_state.file_path = None
                st.session_state.file_name = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.info("Hey, I am ready to answer query. Give query below")

        # Query input section
        user_query = st.text_input("Enter your query:", key="query_input")

        st.markdown('<div class="submit-button">', unsafe_allow_html=True)
        if st.button("Submit", key="submit_query"):
            if user_query:
                st.write("Processing query:", user_query)
                # Add your query processing logic here

                # Show "View GT" and "RAGAS" buttons after submitting the query
                st.session_state.show_gt = True
        st.markdown('</div>', unsafe_allow_html=True)

        # Additional feature buttons
        if st.session_state.show_gt:
            col7, col8 = st.columns([1, 8])
            with col7:
                if st.button("View GT", key="view_gt"):
                    display_csv_data()
            with col8:
                st.button("RAGAS", key="ragas")

# new_data = pd.read_csv('top_tracks_features.csv')


def display_csv_data():
    """Function to display CSV data with gray background for each row"""
    try:
        data = pd.read_csv('top_tracks_features.csv')

        # CSS for grid tile styling
        st.markdown("""
            <style>
            .grid-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px; /* Space between tiles */
                justify-content: center; /* Center the tiles */
            }
            .tile-container {
                background-color: #ffffff; /* Tile background color */
                padding: 15px;
                border-radius: 8px;
                width: 300px; /* Fixed width for uniform tile size */
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow */
                border: 1px solid #e0e0e0; /* Light border for better definition */
                color: #333333;
                font-size: 0.9rem;
                line-height: 1.6;
            }
            .tile-header {
                font-weight: bold;
                color: #0066cc; /* Accent color for headers */
            }
            </style>
        """, unsafe_allow_html=True)

        # Display header
        st.markdown("<div class='tile-header'>CSV File Content:</div>",
                    unsafe_allow_html=True)

        # Start the grid container
        st.markdown("<div class='grid-container'>", unsafe_allow_html=True)

        # Display each row as a tile within the grid container
        for _, row in data.iterrows():
            row_content = "<br>".join(
                [f"<strong>{col}:</strong> {val}" for col, val in row.items()])
            st.markdown(
                f"<div class='tile-container'>{row_content}</div>", unsafe_allow_html=True)

        # Close the grid container
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying file data: {e}")


def get_base64_logo():
    """Function to load and encode the logo image"""
    try:
        with open("image.png", "rb") as image_file:
            import base64
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return ""


if __name__ == "__main__":
    main()
