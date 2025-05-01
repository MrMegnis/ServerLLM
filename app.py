from flask import Flask, request, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load tokenizer and model
MODEL_NAME = "Qwen/Qwen3-1.7B"
print(f"Loading model {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map='auto'
)

app = Flask(__name__)

# Simple HTML template
template = '''
<!doctype html>
<title>Qwen Chat</title>
<h1>Qwen3-1.7B Chat</h1>
<form action="/chat" method="post">
  <input name="message" style="width: 80%;" autocomplete="off" autofocus>
  <input type="submit" value="Send">
</form>
{% if response %}
  <h2>Response:</h2>
  <p>{{ response }}</p>
{% endif %}
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(template)

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.form['message']
    inputs = tokenizer(user_msg, return_tensors='pt')
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Generate response
gen = model.generate(
        **inputs,
        max_length=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )
    reply = tokenizer.decode(gen[0], skip_special_tokens=True)
    return render_template_string(template, response=reply)

if __name__ == '__main__':
    # For CPU-only or simple testing
    app.run(host='0.0.0.0', port=5000)