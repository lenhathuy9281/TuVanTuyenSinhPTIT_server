import chatbot


from flask import Flask, render_template, request, jsonify
import database.utils as db
import whisper

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/get_msg", methods = ["GET"])
def get_bot_response():
    userText = request.args.get('msg')
    return jsonify({
        "message": chatbot.getPredict(userText)
    })

@app.route("/school", methods = ["GET"])
def getSchoolInfo():
    data = db.school()
    return jsonify(
        {
            "id": data[0],
            "name": data[1],
            "description": data[2],
            "thanhPho": data[3],
            "quan": data[4],
            "duong": data[5],
            "image": data[6]
        }
    )

@app.route("/job", methods=["GET"])
def getJobInfo():
    data = db.get_job_data()
    
    # Check if there is any data
    if not data:
        return jsonify({"message": "No job data found"}), 404

    # List to store information for all jobs
    job_list = []

    # Iterate through each job in data
    for job in data:
        job_info = {
            "id": job[0],
            "name": job[1],
            "salary": job[2],
            "status": job[3],
            "description": job[4]
        }
        job_list.append(job_info)

    return jsonify({"jobs":job_list})

@app.route("/scholarship", methods=["GET"])
def get_all_scholarship_info():
    data = db.get_scholarship_data()
    
    # Check if there is any data
    if not data:
        return jsonify({"message": "No scholarship data found"}), 404

    # List to store information for all scholarships
    scholarship_list = []

    # Iterate through each scholarship in data
    for scholarship in data:
        scholarship_info = {
            "id": scholarship[0],
            "loaiHb": scholarship[1],
            "diemYc": scholarship[2],
            "hanhKiemYc": scholarship[3]
        }
        scholarship_list.append(scholarship_info)

    return jsonify({"scholarships":scholarship_list})

@app.route("/tuition", methods=["GET"])
def get_tuition_info():
    data = db.get_tuition_data()
    
    # Check if there is any data
    if not data:
        return jsonify({"message": "No tuition data found"}), 404

    # List to store information for all tuitions
    tuition_list = []

    # Iterate through each tuition in data
    for tuition in data:
        tuition_info = {
            "id": tuition[0],
            "soTien": tuition[1],
            "namHoc": tuition[2]
        }
        tuition_list.append(tuition_info)

    return jsonify({"tuitions":tuition_list})

@app.route("/target", methods=["GET"])
def get_target_info():
    data = db.get_target_data()

    # Kiểm tra xem có dữ liệu nào không
    if not data:
        return jsonify({"message": "No target data found"}), 404

    # List để lưu thông tin cho tất cả các chỉ tiêu
    target_list = []

    # Duyệt qua mỗi chỉ tiêu trong dữ liệu
    for target in data:
        target_info = {
            "id": target[0],
            "nam": target[1],
            "soLuong": target[2],
            "phuongThuc": target[3]
        }
        target_list.append(target_info)

    return jsonify({"targets":target_list})

@app.route("/major", methods=["GET"])
def get_major_info():
    data = db.get_major_data()

    # Kiểm tra xem có dữ liệu nào không
    if not data:
        return jsonify({"message": "No major data found"}), 404

    # List để lưu thông tin cho tất cả các ngành học
    major_list = []

    # Duyệt qua mỗi ngành học trong dữ liệu
    for major in data:
        major_info = {
            "id": major[0],
            "tenNganh": major[1],
            "maNganh": major[2],
            "moTa": major[3],
            "soTin": major[4],
            "coSo": major[5]
        }
        major_list.append(major_info)

    return jsonify({"majors":major_list})

@app.route("/news", methods=["GET"])
def get_news_info():
    data = db.get_news_data()

    # Kiểm tra xem có dữ liệu nào không
    if not data:
        return jsonify({"message": "No news data found"}), 404

    # List để lưu thông tin cho tất cả các tin tức
    news_list = []

    # Duyệt qua mỗi tin tức trong dữ liệu
    for news in data:
        news_info = {
            "id": news[0],
            "title": news[1],
            "image": news[2],
            "fullDescription": news[3],
            "time": news[4],
            "shortDescription": news[5]
        }
        news_list.append(news_info)

    return jsonify( {"news":news_list})

@app.route("/news/detail", methods=["GET"])
def get_news_detail_info():
    news_id = request.args.get("id")
    data = db.get_news_detail(news_id)

    # Kiểm tra xem có dữ liệu nào không
    if not data:
        return jsonify({"message": f"No news found with ID {news_id}"}), 404

    # Chuẩn bị thông tin chi tiết
    news_detail = {
        "id": data[0],
        "title": data[1],
        "image": data[2],
        "fullDescription": data[3],
        "time": data[4],
        "shortDescription": data[5]
    }

    return jsonify(news_detail)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':
        return jsonify( {
        "message": 'No selected file'
    })

    file.save('/home/lenhathuy/android_project/TuVanTuyenSinhPTIT_server/file.mp3')

    model = whisper.load_model("small")

    audio = whisper.load_audio("/home/lenhathuy/android_project/TuVanTuyenSinhPTIT_server/file.mp3")

    options = {
        "language": "vi", 
        "task": "transcribe" 
    }
    result = whisper.transcribe(model, audio, **options)
    # result = model.transcribe("/home/lenhathuy/android_project/TuVanTuyenSinhPTIT_server/file.mp3")

    print(result["text"])
    return  jsonify( {
        "message": result["text"]
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)