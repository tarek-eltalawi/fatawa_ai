from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from service import ask_bot, memory

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Get answer from the Q&A system
        answer = ask_bot(question)
        
        # Return both answer and conversation history
        return jsonify({
            'answer': answer,
            'history': memory.to_dict()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        memory.clear()
        return jsonify({'message': 'Conversation history cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 