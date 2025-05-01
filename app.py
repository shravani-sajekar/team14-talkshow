import streamlit as st
import os
import subprocess
import logging
import traceback
import tempfile

def process_audio(audio_file_path):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Received audio file: {audio_file_path}")
        logger.info(f"Audio file exists: {os.path.exists(audio_file_path)}")

        if not audio_file_path or not os.path.exists(audio_file_path):
            raise ValueError(f"Invalid or non-existent audio file: {audio_file_path}")

        output_dir = os.path.join("visualise", "video", "body-pixel2")
        os.makedirs(output_dir, exist_ok=True)

        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Audio file path: {audio_file_path}")
        logger.debug(f"Audio file size: {os.path.getsize(audio_file_path)} bytes")

        # Build paths relative to this script's location
        base_dir = os.path.dirname(__file__)
        demo_script_path = os.path.join(base_dir, 'scripts', 'demo.py')
        config_file_path = os.path.join(base_dir, 'config', 'body_pixel.json')

        # Generate output filename based on input audio name
        audio_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        output_path = os.path.join(output_dir, f"{audio_filename}.mp4")
        logger.info(f"Expected output path: {output_path}")

        cmd = [
            "python3",
            demo_script_path,
            "--config_file", config_file_path,
            "--infer",
            "--audio_file", audio_file_path,
            "--id", "0",
            "--whole_body"
        ]

        logger.info(f"Executing command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd(),
            timeout=180
        )

        logger.info(f"Command STDOUT: {result.stdout}")
        logger.error(f"Command STDERR: {result.stderr}")

        if os.path.exists(output_path):
            logger.info(f"Output video found: {output_path}")
            return output_path, None
        else:
            return None, f"Error: Output video not generated. STDERR: {result.stderr}"

    except subprocess.TimeoutExpired:
        return None, "Error: Inference process took too long"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return None, f"Unexpected error: {str(e)}"

# Streamlit UI
st.title("TalkSHOW: Speech-to-Motion Translation System")
st.markdown("Convert speech audio to realistic 3D human motion using the SMPL-X model.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(uploaded_file.read())
        tmp_audio_path = tmp_audio.name

    st.info("Processing audio...")
    video_path, error = process_audio(tmp_audio_path)

    if error:
        st.error(error)
    elif video_path:
        st.success("Motion video generated successfully!")
        st.video(video_path)
