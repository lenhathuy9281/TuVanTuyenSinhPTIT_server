# import nltk
# nltk.download('popular')
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# import pickle
# import numpy as np

# from keras.models import load_model
# model = load_model('model.h5')
# import json
# import random
# intents = json.loads(open('data.json').read())
# words = pickle.load(open('texts.pkl','rb'))
# classes = pickle.load(open('labels.pkl','rb'))

# def clean_up_sentence(sentence):
#     # tokenize the pattern - split words into array
#     sentence_words = nltk.word_tokenize(sentence)
#     # stem each word - create short form for word
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

# def bow(sentence, words, show_details=True):
#     # tokenize the pattern
#     sentence_words = clean_up_sentence(sentence)
#     # bag of words - matrix of N words, vocabulary matrix
#     bag = [0]*len(words)  
#     for s in sentence_words:
#         for i,w in enumerate(words):
#             if w == s: 
#                 # assign 1 if current word is in the vocabulary position
#                 bag[i] = 1
#                 if show_details:
#                     print ("found in bag: %s" % w)
#     return(np.array(bag))

# def predict_class(sentence, model):
#     # filter out predictions below a threshold
#     p = bow(sentence, words,show_details=False)
#     res = model.predict(np.array([p]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
#     # sort by strength of probability
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
#     return return_list

# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
#     for i in list_of_intents:
#         if(i['tag']== tag):
#             result = random.choice(i['responses'])
#             break
#     return result

# def chatbot_response(msg):
#     ints = predict_class(msg, model)
#     res = getResponse(ints, intents)
#     return res


from flask import Flask, render_template, request, jsonify
import database.utils as db

app = Flask(__name__)
app.static_folder = 'static'

# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     return chatbot_response(userText)

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)