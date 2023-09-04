from flask import Flask, request, render_template
import pandas as pd
import ADOPT
import datetime
app = Flask(__name__)

def create_custom_dataframe():
    data = {
        'Column1': [1, 2],
        'Column2': ['A', 'B'],
        'Column3': [True, False]
    }

    df = pd.DataFrame(data)
    return df

@app.route('/')
def home():
    return render_template("index.html/")

@app.route('/index.html', methods = ["GET", "POST"])
def inputs():
    if request.method == "POST":
        title = request.form.get("query")
        number_of_videos = int(request.form.get("nov"))
        MinLikesForComments = int(request.form.get("mlfc"))
        start_time = datetime.datetime.now()
        df = ADOPT.driver(ADOPT.api,title,number_of_videos,MinLikesForComments = MinLikesForComments)
        df=df.drop(['Sentiemnt', 'Conf Val'], axis = 1)
        end_time = datetime.datetime.now()
        time_taken = end_time - start_time
        # Convert DataFrame to HTML table
        result_table = df.to_html(index=False)
        return render_template("result.html", query = title, videos = number_of_videos, result=result_table, time = time_taken)
    return render_template("index.html")
if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)