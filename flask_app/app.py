import os
from flask import Flask, render_template, redirect, url_for
from flask_wtf import CSRFProtect, FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
from model.model import model_predict
from dotenv import load_dotenv

load_dotenv()


UPLOAD_FOLDER = "./static/uploads"


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
csrf = CSRFProtect(app)


class ImageForm(FlaskForm):
    image = FileField(validators=[FileRequired()])
    submit = SubmitField("Submit")


@app.route("/")
def home():
    form = ImageForm()
    return render_template("index.html", form=form)


@app.route("/predict", methods=["POST"])
def predict():
    form = ImageForm()

    if form.validate_on_submit():
        f = form.image.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        result = model_predict(filename)

        return render_template("predict.html", results=result, image_path=filename)

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run()
