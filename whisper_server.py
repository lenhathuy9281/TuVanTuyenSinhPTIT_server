from flask import Flask, render_template, request, jsonify
import whisper

app = Flask(__name__)
app.static_folder = 'static'


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    return jsonify({"school": "oh ok"})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    # Handle the uploaded file (save it, process it, etc.)
    file.save('/home/lenhathuy/android_project/TuVanTuyenSinhPTIT_server/file.mp3')

    model = whisper.load_model("small")

    audio = whisper.load_audio("/home/lenhathuy/android_project/TuVanTuyenSinhPTIT_server/file.mp3")

    options = {
        "language": "vi", # input language, if omitted is auto detected
        "task": "transcribe" # or "transcribe" if you just want transcription
    }
    result = whisper.transcribe(model, audio, **options)
    # result = model.transcribe("/home/lenhathuy/android_project/TuVanTuyenSinhPTIT_server/file.mp3")

    print(result["text"])
    return result["text"]

@app.route("/school", methods = ["GET"])
def getSchoolInfo():
    # return jsonify({"school": db.school()})
    return jsonify({"school": "db.school()"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)