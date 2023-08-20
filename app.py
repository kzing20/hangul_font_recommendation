#app.py
from flask import Flask, jsonify, request, render_template, url_for
from flask_cors import CORS
from flask_restx import Resource, Api
import sys
# sys.path.append('total_font_recommend_model')
import total_font_recommend_model as font_model

app = Flask(__name__)
CORS(app)

#api
# api = Api(app, version='1.0', title='Flask API', description='Flask API', doc="/api-docs")
# api.add_namespace(total_model_recommend,'/api/recommend_model')

@app.route("/", methods=['GET', 'POST'])
def helloWorld():
  return "Hello world!"

@app.route("/font_recommend_test", methods=['GET', 'POST'])
def font_recommend():
  #가중치 -> 나중에 사용자 입력 값으로 수정
  total_weights = [2,1,3]

  # POST 요청으로부터 font_names 받아오기
  font_model.font_names = request.json.get('font_names', [])
  font_model.weights = request.json.get('weights', [])

  if font_model.weights==1: #디폴트 값 설정
    font_model.weights = [5] * len(font_model.font_names)

  print(font_model.font_names,font_model.weights) #확인용

  #폰트 추천 시스템 모듈 호출
  search_rank_list = font_model.total_model_recommend(total_weights)
  print(search_rank_list) #확인용
  return jsonify(search_rank_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)