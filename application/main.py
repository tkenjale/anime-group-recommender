from flask import Flask, render_template, request, jsonify
import sqlite3
from models.group_recommender_mf import GroupRecommenderMF
#from models.explicit_mf_with_bias import SGDExplicitBiasMF
import pandas as pd 
import json
from fuzzywuzzy import fuzz
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import gdown
import os

#from flask_cors import CORS #comment this on deployment

app = Flask(__name__, static_folder="static")
#CORS(app) #comment this on deployment

# Download necessary files
## Check if model file exist first
if not os.path.exists("./data/model_sgd_mf_v4_50__1666837325.pkl"):
    print("Downloading the pre-trained embedding model")
    url = 'https://drive.google.com/u/0/uc?id=1Gws01MJsveOuRVTSD4wRefoc0csa-RYm'
    output = 'data/model_sgd_mf_v4_50__1666837325.pkl'
    gdown.download(url, output, quiet=False)

group_mf = GroupRecommenderMF(
    full_model_file_path = "data/model_sgd_mf_v4_50__1666837325.pkl",
    item_encoder_file_path="data/anime_encoder.csv")

#anime_df = pd.read_csv("data/anime.csv")

lookup = pd.read_csv("data/anime_lookup.csv")

def connect_to_db():
    db_url = "data/test.db"
    conn = sqlite3.connect(db_url)
    conn.text_factory = str

    return conn


@app.route('/')
def index():

    # conn = connect_to_db()
    # get_data = "SELECT id, name, genres FROM anime_lookup WHERE name MATCH 'shingeki*'"
    # cursor = conn.execute(get_data)
    # data = cursor.fetchall()
    # print(data)


    # conn.close() 

    return render_template("index.html")

@app.route('/collect', methods=['POST'])
def collect_form_data():
    data = request.form['jdata']
    data = json.loads(data)

    # usernames = []
    # user_inputs = []
    # ratings = []
    # db_inputs = []
    # ids = []


    # for user in data:
    #     for i in range(len(user['ratings'])):

    #         max_j = 0
    #         max_r = 0
    #         for j in range(lookup.shape[0]):
    #             r = fuzz.partial_ratio(user['animes'][i], lookup['name'][j])
    
    #             if r > max_r:
    #                 max_r = r
    #                 max_j = j
    
    #         anime_name = lookup['name'][max_j]

    #         usernames.append(user['name'])
    #         user_inputs.append(user['animes'][i])
    #         ratings.append(user['ratings'][i])
    #         db_inputs.append(anime_name)
    #         ids.append(lookup['id'][max_j])

    # df = pd.DataFrame(list(zip(usernames, user_inputs, ratings, db_inputs, ids)), columns=["name", "anime_name_user_input", "rating", "anime_name_database", "anime_id"])
    # print(df)
    conn = connect_to_db()
    conn.execute("DELETE FROM test")
    conn.commit()

    for user in data:
        for i in range(len(user['ratings'])):

            max_j = 0
            max_r = 0
            for j in range(lookup.shape[0]):
                r = fuzz.partial_ratio(user['animes'][i], lookup['name'][j])
    
                if r > max_r:
                    max_r = r
                    max_j = j
    
            anime_name = lookup['name'][max_j]

            conn.execute("INSERT INTO test(name, anime_name_user_input, rating, anime_name_database, anime_id) VALUES ('{}', '{}', {}, '{}', {});"
                        .format(user['name'], user['animes'][i], user['ratings'][i], anime_name, lookup['id'][max_j]))
            conn.commit()

    conn.close()    

    return ""

def get_image_url(id):
    html_page = requests.get("https://myanimelist.net/anime/" + str(id))
    soup = BeautifulSoup(html_page.content, 'html.parser')
    image = soup.find('div', class_="leftside").find('img')
    url = image.attrs['data-src']
    
    return {str(id): url}

@app.route('/predict', methods=['POST'])
def generate_predictions():
    #print("generating")
    conn = connect_to_db()

    #cursor = conn.execute("SELECT name, anime_id, rating FROM group_rating_sample")
    cursor = conn.execute("SELECT name, anime_id, rating FROM test")

    data = cursor.fetchall()

    #print(data)
    conn.close()  


    group_rating_df = pd.DataFrame(data, columns=["user_name", "item_id", "rating"])
    #print(group_rating_df)

    recommendations = group_mf.recommend_group(group_rating_df, reg=float(request.form.get("reg")), 
                                                rec_type=request.form.get("rec_type"), 
                                                agg_method=request.form.get("agg_method"))

    #print(recommendations)
    users = recommendations.columns[1:-1].to_list() 
    #print(users)

    results = recommendations.merge(lookup, left_on="item_id", right_on="id", how="inner")

    results_dict = {"results": []}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        a, b, c, d, e = executor.map(get_image_url, results['item_id'][:5].tolist())

    a.update(b)
    a.update(c)
    a.update(d)
    a.update(e)

    for i in range(5):
                
        temp = {
            "anime_name": results['name'][i],
            "anime_id": int(results['item_id'][i]),
            "overall_score": round(results['recommendation_score'][i], 2),
            "genres": results['genres'][i].split(", "),
            "individual_predictions": {},
            "rank": i + 1,
            "image_url": a[str(results['item_id'][i])]
        }
        for user in users:
            temp["individual_predictions"][user] = round(results[user][i], 2)

        results_dict["results"].append(temp.copy())


    with open("static/data/predictions.json", "w") as outfile:
        json.dump(results_dict, outfile) 
 
    return jsonify(results_dict)
 
@app.route("/visualize", methods=["GET"])
def show_visualization():

    return render_template("tech.html") 

if __name__ == '__main__':
    app.run(debug=True)