from flask import Flask, render_template
import os

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def serve_static(path):
    return app.send_static_file(path)

if __name__ == '__main__':  
    print("="*70)
    print("MARKET MOVEMENT CLASSIFIER - FRONTEND SERVER")
    print("="*70)
    print("Frontend running on: http://localhost:8080")
    print("Make sure backend API is running on: http://localhost:5000")
    print("="*70 + "\n")
    
    import webbrowser
    try:
        webbrowser.open('http://localhost:8080')
    except:
        pass
    
    app.run(debug=True, host='0.0.0.0', port=8080)