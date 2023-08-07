from flask import Flask, render_template, request, session
from transformers import BarkModel, AutoProcessor
from IPython.display import Audio



model = BarkModel.from_pretrained("suno/bark")
sampling_rate = model.generation_config.sample_rate
processor = AutoProcessor.from_pretrained("suno/bark")




app = Flask(__name__)
app.secret_key = "hani"


@app.route("/")
def home():
    session["text_in_audio"] = ""
    session["text_in_video"] = ""

    return render_template("tts.html")


@app.route("/process_text", methods=["POST"])
def process_text():
 

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
