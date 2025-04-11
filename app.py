from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from wtforms import StringField, SubmitField, PasswordField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
import os
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch
import json

# Configuration for quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit_fp32_cpu_offload=True
)

# Load model and tokenizer
path = "/mnt/c/Users/Dell/Downloads/codellama"
model = AutoModelForCausalLM.from_pretrained(path, token="Use_Your_Own", torch_dtype=torch.float16, quantization_config=bnb_config, device_map="cuda:0")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(path)

# Initialize Flask app and SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = os.urandom(24)
db = SQLAlchemy(app)

# Default model configuration
model_temperature = 1.0
model_max_new_tokens = 256

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Name %r>' % self.name

class RegistrationForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired()])
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField("Register")

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField("Login")

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = Users(name=form.name.data, email=form.email.data, username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('You have successfully registered!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = Users.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password.data, form.password.data):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', form=form)

@app.route('/')
def home():
    return render_template('index.html')

conversation_history = []

@app.route('/get-model-config', methods=['GET'])
def get_model_config():
    global model_temperature, model_max_new_tokens
    return jsonify({
        'temperature': model_temperature,
        'max_new_tokens': model_max_new_tokens
    })

@app.route('/update-model-config', methods=['POST'])
def update_model_config():
    data = request.json
    temperature = float(data.get('temperature', 1.0))
    max_new_tokens = int(data.get('max_new_tokens', 256))

    global model_temperature, model_max_new_tokens
    model_temperature = temperature
    model_max_new_tokens = max_new_tokens
    
    return jsonify({'status': 'success', 'temperature': model_temperature, 'max_new_tokens': model_max_new_tokens})

def generate_response_directly(prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=model_max_new_tokens,
        temperature=model_temperature,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    return response

@app.route('/submit-prompt', methods=['POST'])
def submit_prompt():
    global conversation_history
    if request.json.get('prompt') == 'NEW_CHAT':
        conversation_history = []
        return jsonify({'status': 'success'})
    data = request.json
    user_prompt = data.get('prompt', '')

    conversation_history.append({"role": "user", "content": user_prompt})

    full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])

    generated_response = generate_response_directly(full_prompt)

    conversation_history.append({"role": "assistant", "content": generated_response})

    response = {
        'status': 'success',
        'response': generated_response
    }
    
    response_json = json.dumps(response, indent=4)
    
    return response_json, 200, {'Content-Type': 'application/json'}

@app.route('/initdb')
def init_db():
    db.create_all()
    return 'Database initialized!'

@app.route('/users')
def list_users():
    users = Users.query.all()
    return render_template('list_users.html', users=users)

if __name__ == '__main__':
    app.run(debug=True)
