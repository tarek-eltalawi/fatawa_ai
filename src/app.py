from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_core.messages import HumanMessage
import os

# Use proper imports that work both when imported and when run directly
from src.retrieval_graph.service import graph
from src.utilities.utils import format_sources

# Get the project root directory to find templates and static files
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask with correct paths to templates and static files
app = Flask(__name__, 
            template_folder=os.path.join(project_root, 'templates'),
            static_folder=os.path.join(project_root, 'static'))
CORS(app)

# Define available sources
SOURCES = {
    'en': [
        {'id': 'dar-al-iftaa-en', 'name': "Egypt's Dar Al Iftaa"}
    ],
    'ar': [
        {'id': 'dar-al-iftaa-ar', 'name': "دار الإفتاء المصرية"}
    ]
}

# Define translations
TRANSLATIONS = {
    'en': {
        'placeholder': "Type your question here...",
        'thinking': "Thinking",
        'title': "Fatwa",
        'sources_title': "Sources",
        'error_no_question': "No question provided",
        'error_internal': "Internal server error",
        'history_cleared': "Conversation history cleared"
    },
    'ar': {
        'placeholder': "اكتب سؤالك هنا...",
        'thinking': "جارٍ التفكير",
        'title': "فتوى",
        'sources_title': "المصادر",
        'error_no_question': "لم يتم إدخال سؤال",
        'error_internal': "خطأ في النظام",
        'history_cleared': "تم مسح سجل المحادثة"
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translations/<lang>', methods=['GET'])
def get_translations(lang):
    """Return translations for specified language."""
    if lang not in TRANSLATIONS:
        lang = 'en'  # Default to English if language not found
    return jsonify(TRANSLATIONS[lang])

@app.route('/sources', methods=['GET'])
def get_sources():
    """Return available sources for both languages."""
    return jsonify(SOURCES)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '')
        provider = data.get('provider', '')
        lang = data.get('language', 'en')
        
        if not question:
            return jsonify({
                'error': TRANSLATIONS[lang]['error_no_question']
            }), 400

        # config = {"configurable": {"thread_id": "1"}}
        # Run the graph.
        final_state = graph.invoke({
            "language": lang,
            "provider": provider,
            "messages": [HumanMessage(content=question)]
        })

        answer = final_state.get('messages')[-1].content
        sources = final_state.get('sources', [])
        
        # Return structured response
        response = {
            "answer": answer,
            "sources": format_sources(sources, lang == 'ar'),
            "language": lang
        }
        
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error processing question: {str(e)}")
        return jsonify({
            'error': TRANSLATIONS[lang]['error_internal']
        }), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        lang = request.json.get('language', 'en')
        return jsonify({
            'message': TRANSLATIONS[lang]['history_cleared']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 