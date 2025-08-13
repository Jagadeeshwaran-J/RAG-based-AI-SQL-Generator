# 🤖 RAG-based AI SQL Generator

This project is a powerful AI-driven tool that takes **natural language questions** from the user, understands the schema of a **SQL Server database**, and generates valid **SQL queries** using the **Gemini Pro 2.0 Flash API** from Google. It uses **FAISS + SentenceTransformers** to retrieve relevant schema chunks and presents results with visualizations and export options.

---

## 🚀 Features

✅ Natural language to SQL conversion  
✅ Supports Microsoft SQL Server  
✅ Semantic schema retrieval using FAISS  
✅ Gemini 2.0 Flash API integration  
✅ Interactive table correction for SQL regeneration  
✅ Dynamic data visualization (Bar & Pie charts)  
✅ Download results as Excel file

---

## 🧩 Components & Why They Are Used

| Component              | Purpose |
|------------------------|---------|
| **Streamlit**          | Web interface |
| **PyODBC**             | Connects to SQL Server |
| **Pandas**             | DataFrame handling & export |
| **SentenceTransformers** | Convert text to embeddings |
| **FAISS**              | Fast semantic search on schema |
| **Requests**           | API calls to Gemini |
| **Matplotlib**         | Chart rendering |
| **NumPy**              | Numerical vector processing |
| **Scikit-learn**       | Embedding normalization |
| **Regex (`re`)**       | Extract & correct table names |
| **json & BytesIO**     | Caching and file handling |

---

## 🧠 How It Works

1. **Schema Extraction**
   - Connects to SQL Server using `pyodbc`.
   - Pulls schema details from `INFORMATION_SCHEMA.COLUMNS`.
   - Converts each table schema into a text "chunk".

2. **Embedding & Indexing**
   - Embeds schema chunks using `SentenceTransformer`.
   - Stores vectors in FAISS index (`faiss_schema.index`).

3. **Natural Query Processing**
   - User submits a question in plain English.
   - The query is embedded and used to search top matching schema chunks.

4. **SQL Generation**
   - Sends relevant schema + question as prompt to Gemini Flash 2.0 API.
   - Receives generated SQL.

5. **Execution & Results**
   - SQL is executed on SQL Server.
   - Results shown in table, charts (bar/pie), and downloadable as Excel.

6. **Error Handling & Edit**
   - If the query fails, user can specify the correct table.
   - Regenerates SQL with hinting.

---

## 🧰 Configuration (Streamlit Sidebar)

| Setting          | Example |
|------------------|---------|
| **ODBC Driver**  | `ODBC Driver 18 for SQL Server` |
| **Server IP**    | `your_ip_address` |
| **Database Name**| `your_db_name` |
| **Username**     | `your_sql_user` |
| **Password**     | `your_password` |

---

## 📁 Project Structure

```bash
├── app.py # Main Streamlit app
├── faiss_schema.index # FAISS vector index for schema
├── schema_chunks.json # Cached schema chunks
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)
```

---

## 🔧 Setup Instructions

### 1. ✅ Clone the Repository

```bash
https://github.com/Jagadeeshwaran-J/RAG-based-AI-SQL-Generator.git
cd RAG-based-AI-SQL-Generator
```

---

### 2. ✅ Install Requirements

```bash
pip install -r requirements.txt
```

---

### 3. ✅ Configure Google Gemini API Key

✅ Go to: https://aistudio.google.com/apikey

✅ Click on Get API Key

✅ Replace in app.py:

```bash
GEMINI_API_KEY = 'YOUR_API_KEY'
```

---

### 4. ✅ Update SQL Server Connection
In the Streamlit sidebar or default in app.py, provide:
```bash
driver = "ODBC Driver 18 for SQL Server"
server = "your_ip_address"
database = "your_db_name"
username = "your_sql_user"
password = "your_password"
```
💡 Ensure that the target database allows TCP connections and ODBC access.

---

### ▶️ Run the App
```bash
streamlit run app.py
```
Open in browser at: http://localhost:8501

---

### 🧪 Sample Workflow
1. Enter your DB connection details.

2. Ask a question like:
“What are the top 5 branches with the highest loan amount?”

3. App finds the right table → generates SQL → runs it → shows results.

4. If the table is wrong, correct it → regenerate SQL.

5. Download results as Excel.

---

### 📸 Screenshots

<img width="1365" height="629" alt="image" src="https://github.com/user-attachments/assets/2ea09988-562e-485a-a65e-ab6a48f75695" />

✨ SQL Generation from natural query

✏️ Editable Table Selection

📊 Query Results with Charts

📥 Excel Download Button

---

## 💡 Key Concepts

| Concept              | Explanation                                            |
|----------------------|--------------------------------------------------------|
| **RAG (Retrieval-Augmented Gen)** | Retrieves schema chunks before query generation       |
| **FAISS**            | Fast vector search over schema embeddings              |
| **Gemini 2.0 Flash** | Google LLM for generating SQL from prompt              |
| **Embeddings**       | Numeric vectors used for semantic similarity           |
| **Edit Mode**        | Manual correction of selected table(s)                 |

---
