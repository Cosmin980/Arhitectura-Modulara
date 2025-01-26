import json
import sqlite3
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# -------------------------
# Variabile globale
# -------------------------
model = None
accuracy = None
loaded_data = None  # Datele ultimului fișier CSV încărcat (în memorie)

# -------------------------
# Funcții Bază de Date
# -------------------------
def init_db():
    """
    Creăm (dacă nu există deja) o bază de date locală (SQLite),
    cu un tabel 'uploaded_data' care va stoca rândurile în format JSON.
    """
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_data TEXT
        )
    ''')
    conn.commit()
    conn.close()

def store_in_db(df: pd.DataFrame):
    """
    Ia fiecare rând din DataFrame și îl inserează ca JSON în coloana 'row_data'.
    """
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    for _, row in df.iterrows():
        row_json = json.dumps(row.to_dict())  # transformăm rândul în dicționar, apoi în JSON
        c.execute('INSERT INTO uploaded_data (row_data) VALUES (?)', (row_json,))
    conn.commit()
    conn.close()

def load_from_db():
    """
    Citește toate rândurile din 'uploaded_data' și le reconstituie ca listă de dicționare.
    """
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT row_data FROM uploaded_data')
    rows = c.fetchall()
    conn.close()

    # rows este o listă de tuple [(row_data_json,), ...]
    data_list = []
    for (row_json,) in rows:
        data_dict = json.loads(row_json)
        data_list.append(data_dict)
    return data_list

# -------------------------
# Funcție de antrenare model
# -------------------------
def train_model(data_path):
    """
    1) Încarcă datele din CSV în Pandas
    2) Actualizează variabila globală loaded_data
    3) Stochează datele în baza de date
    4) Antrenează un model Logistic Regression
    5) Calculează metrica de acuratețe
    """
    global model, accuracy, loaded_data

    # 1) Citim CSV-ul
    data = pd.read_csv(data_path)
    loaded_data = data  # pentru afișarea rapidă din memorie

    # 2) Stocăm datele în DB
    store_in_db(data)

    # 3) Antrenăm model Logistic Regression
    X = data.iloc[:, :-1]  # toate coloanele mai puțin ultima
    y = data.iloc[:, -1]   # ultima coloană este considerată etichetă

    # împărțim datele în train și test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4) Calculăm accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

# -------------------------
# Inițializare DB (o singură dată, la pornirea aplicației)
# -------------------------
init_db()

# -------------------------
# Rute Flask
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Endpoint pentru încărcarea fișierului CSV:
      - Salvează fișierul local
      - Apelează train_model
      - Returnează JSON cu un mesaj și cu accuracy
    """
    file = request.files['file']
    file_path = f'./{file.filename}'
    file.save(file_path)

    train_model(file_path)

    return jsonify({"message": "Model trained successfully", "accuracy": accuracy})

@app.route('/results', methods=['GET'])
def results():
    """
    Returnează metrica de acuratețe pentru ultimul model antrenat
    """
    return jsonify({"accuracy": accuracy})

@app.route('/data', methods=['GET'])
def data_listing():
    """
    Returnează datele ultimului fișier încărcat (din memorie)
    """
    global loaded_data
    if loaded_data is None:
        return jsonify({"error": "No data loaded yet."}), 400
    data_dict = loaded_data.to_dict(orient='records')
    return jsonify(data_dict)

@app.route('/stats', methods=['GET'])
def stats():
    """
    Returnează statistici descriptive (describe) pentru ultimul fișier încărcat
    """
    global loaded_data
    if loaded_data is None:
        return jsonify({"error": "No data loaded yet."}), 400

    desc_df = loaded_data.describe(include='all')
    stats_dict = desc_df.to_dict()
    return jsonify(stats_dict)

@app.route('/db_data', methods=['GET'])
def db_data():
    """
    Returnează TOATE datele stocate în DB (din mai multe fișiere CSV, dacă s-au încărcat).
    """
    all_data = load_from_db()  # listă de dicționare
    return jsonify(all_data)

if __name__ == '__main__':
    # Rulăm aplicația Flask
    app.run(debug=True)