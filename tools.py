from flask import Flask, render_template, request, session
from transformers import BarkModel, AutoProcessor
from IPython.display import Audio
from pytube import YouTube

import whisper

model_base = whisper.load_model("small")

app = Flask(__name__)
app.secret_key = "hani"


@app.route("/")
def home():
    session["text_in_audio"] = ""
    session["text_in_video"] = ""

    return render_template("tts.html")


@app.route("/process_text", methods=["POST"])
def process_text():
    model = BarkModel.from_pretrained("suno/bark")
    sampling_rate = model.generation_config.sample_rate
    processor = AutoProcessor.from_pretrained("suno/bark")

    user_lang = request.form.get("dropdown_lang")
    speaker = request.form.get("dropdown_voice")
    user_text = request.form.get("A13")
    if user_lang in {
        "English",
        "Chinese",
        "French",
        "German",
        "Hindi",
        "Italian",
        "Japanese",
        "Korean",
        "Polish",
        "Portuguese",
        "Russian",
        "Spanish",
        "Turkish",
    }:
        print("User selected a valid language.")
        voice_preset = "v2/" + speaker
        # prepare the inputs
        text_prompt = user_text
        inputs = processor(text_prompt, voice_preset=voice_preset)

        # generate speech
        speech_output = model.generate(**inputs)

        # let's hear it
        audio_data = Audio(speech_output[0].cpu().numpy(), rate=sampling_rate)

    print(f"You entered: {voice_preset}")
    return render_template(
        "tts.html",
        user_lang=user_lang,
        speaker=speaker,
        user_text=user_text,
        audio_data=audio_data,
    )


@app.route("/process_audio", methods=["POST"])
def process_audio():
    if "gen" in request.form:
        selected_language = request.form.get("dropdown_lang")

        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]

        if file.filename == "":
            return "No selected file"
        audio_name = "uploads/" + file.filename

        # Save the file to a specific directory (e.g., "uploads" folder)
        file.save("static/uploads/" + file.filename)

        # Get the path of the uploaded file
        audio_path = "static/uploads/" + file.filename

        options = {"fp16": False, "language": selected_language, "task": "transcribe"}
        result = model_base.transcribe(audio_path, **options)
        text_in_audio = "Your audio says:-" + result["text"]

        session["text_in_audio"] = text_in_audio
        return render_template(
            "stt.html",
            selected_language=selected_language,
            audio_path=audio_path,
            audio_name=audio_name,
            text_in_audio=text_in_audio,
        )
    elif "down" in request.form:
        text_in_audio = session.get("text_in_audio", None)
        if text_in_audio == "":
            return "Generate audio transcription first"
        else:
            with open("transcription.txt", "w", encoding="utf-8") as file:
                file.write(text_in_audio)
                file.close()
            return "Audio transcription downloaded successfully!!"


@app.route("/process_video", methods=["POST"])
def process_video():
    if "gen2" in request.form:
        video_url = request.form.get("user_text")
        video = YouTube(video_url)
        model = whisper.load_model("base")
        streams = video.streams.filter(only_audio=True)
        stream = streams.first()
        stream.download(filename="motivation.mp4")
        output = model.transcribe("motivation.mp4")
        text_in_video = output["text"]

        session["text_in_video"] = text_in_video

        return render_template(
            "stt.html", video_url=video_url, text_in_video=text_in_video
        )
    elif "down2" in request.form:
        text_in_video = session.get("text_in_video", None)
        if text_in_video == "":
            return "Generate video transcription first"
        else:
            with open("transcript.txt", "w", encoding="utf-8") as file:
                file.write(text_in_video)
                file.close()
            return "Video transcription downloaded successfully!!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
