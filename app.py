import streamlit as st
import pyodbc
import pandas as pd
import requests
import faiss
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from io import BytesIO
import numpy as np
import json
import re

# --- CONFIG ---
GEMINI_API_KEY = 'YOUR_API_KEY'
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VECTOR_STORE_PATH = 'faiss_schema.index'
SCHEMA_JSON = 'schema_chunks.json'

# --- Sidebar ---
st.sidebar.header("🛠️ Database Configuration")
driver = st.sidebar.text_input("ODBC Driver", "ODBC Driver 18 for SQL Server")
server = st.sidebar.text_input("Server", "your_ip_address")
database = st.sidebar.text_input("Database", "your_db_name")
username = st.sidebar.text_input("Username", "your_sql_user")
password = st.sidebar.text_input("Password", "your_password", type="password")

st.set_page_config(page_title="RAG-based AI SQL Generator", layout="wide")
st.title("🤖 RAG-based AI SQL Generator")

# --- Session State Initialization ---
if "schema_loaded" not in st.session_state:
    st.session_state.schema_loaded = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "generated_sql" not in st.session_state:
    st.session_state.generated_sql = ""
if "query_result" not in st.session_state:
    st.session_state.query_result = pd.DataFrame()
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "corrected_table" not in st.session_state:
    st.session_state.corrected_table = ""
if "initial_user_query" not in st.session_state:  # Store the initial user query for regeneration
    st.session_state.initial_user_query = ""


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_data(show_spinner=False)
def get_schema_chunks(db_config):
    query = """
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    conn_str = (
        f"DRIVER={{{db_config['driver']}}};"
        f"SERVER={db_config['server']};"
        f"DATABASE={db_config['database']};"
        f"UID={db_config['username']};"
        f"PWD={db_config['password']};"
        "TrustServerCertificate=yes;"
    )
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql(query, conn)
    conn.close()
    grouped = df.groupby("TABLE_NAME")
    chunks = []
    for table, group in grouped:
        columns = ", ".join(f"{row.COLUMN_NAME} {row.DATA_TYPE}" for _, row in group.iterrows())
        chunks.append(f"{table}({columns})")
    return chunks


def index_schema_chunks(chunks):
    model = load_embedding_model()
    embeddings = model.encode(chunks, convert_to_tensor=False)
    norm_embeddings = normalize(embeddings)
    dim = norm_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(norm_embeddings.astype(np.float32))
    faiss.write_index(index, VECTOR_STORE_PATH)
    with open(SCHEMA_JSON, 'w') as f:
        json.dump(chunks, f)


def retrieve_top_k_chunks(user_query, k=5, hint_table=None):
    model = load_embedding_model()
    # If a hint_table is provided, prioritize it
    if hint_table:
        user_query_with_hint = f"{user_query} (focus on table: {hint_table})"
        query_embedding = model.encode([user_query_with_hint], convert_to_tensor=False)
    else:
        query_embedding = model.encode([user_query], convert_to_tensor=False)

    query_embedding = normalize(query_embedding)
    index = faiss.read_index(VECTOR_STORE_PATH)
    D, I = index.search(query_embedding.astype(np.float32), k)
    with open(SCHEMA_JSON, 'r') as f:
        chunks = json.load(f)

    # Ensure the hint_table (if provided) is among the retrieved chunks,
    # and try to place it at the beginning if relevant
    retrieved_chunks = [chunks[i] for i in I[0]]
    if hint_table:
        hint_chunk_found = False
        for i, chunk in enumerate(retrieved_chunks):
            if chunk.lower().startswith(hint_table.lower() + "("):
                # Move the hint chunk to the front
                retrieved_chunks.insert(0, retrieved_chunks.pop(i))
                hint_chunk_found = True
                break
        # If hint_table was not in top K, find it and add it (if it exists in full schema)
        if not hint_chunk_found:
            all_chunks = get_schema_chunks(db_config)  # get all schema chunks to find the hint_table
            for chunk in all_chunks:
                if chunk.lower().startswith(hint_table.lower() + "("):
                    retrieved_chunks.insert(0, chunk)
                    retrieved_chunks = retrieved_chunks[:k]  # Trim if it exceeds k
                    break
    return retrieved_chunks


def generate_sql_from_gemini(schema_chunks, user_query):
    prompt = f"""
You are an expert SQL assistant.
Generate a valid Microsoft SQL Server query using the schema below.

Use only the table and column names exactly from the schema.
Do not guess.

Schema:
{chr(10).join(schema_chunks)}

Question: "{user_query}"

Respond only with the SQL query.
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048}
    }

    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        json=payload
    )
    response.raise_for_status()

    # Raw Gemini output
    raw_text = response.json()['candidates'][0]['content']['parts'][0]['text']

    # Clean SQL output from Gemini
    cleaned_lines = []
    for line in raw_text.splitlines():
        line = line.strip()
        if line.lower().startswith("sql") or line.startswith("```") or line.startswith("--"):
            continue
        cleaned_lines.append(line)

    result_sql = " ".join(cleaned_lines).strip(" ;`")
    return result_sql


def run_sql(sql, db_config):
    try:
        conn_str = (
            f"DRIVER={{{db_config['driver']}}};"
            f"SERVER={db_config['server']};"
            f"DATABASE={db_config['database']};"
            f"UID={db_config['username']};"
            f"PWD={db_config['password']};"
            "TrustServerCertificate=yes;"
        )
        conn = pyodbc.connect(conn_str)
        df = pd.read_sql(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)


def extract_table_name(sql):
    # Improved extraction to handle various SQL structures and casing
    # This regex is specifically designed to capture the table name after FROM
    # It tries to be broad to capture different naming conventions including schema.table
    match = re.search(r"FROM\s+([a-zA-Z0-9_.\"\[\]]+)", sql, re.IGNORECASE)
    if match:
        table_name = match.group(1).strip()
        # Remove square brackets or double quotes if they exist around table names
        table_name = table_name.strip('[]"').split('.')[-1]  # take last part for schema.table
        return table_name
    return ""


# The replace_table_in_sql is less relevant now as we're regenerating, but keep it robust
def replace_table_in_sql(original_sql, old_table, new_table):
    # Use re.sub for case-insensitive replacement
    # We use \b to ensure whole word matching (e.g., 'table' doesn't replace 'another_table')
    # We escape old_table to handle special characters if they exist in table names
    # This function is primarily for display or quick fixes, the regeneration is preferred for logic
    return re.sub(r'\b' + re.escape(old_table) + r'\b', new_table, original_sql, flags=re.IGNORECASE)


# DB Config
db_config = {
    "driver": driver,
    "server": server,
    "database": database,
    "username": username,
    "password": password,
}

# Reindex if needed
if st.sidebar.button("🔁 Force Reindex Schema"):
    if os.path.exists(VECTOR_STORE_PATH):
        os.remove(VECTOR_STORE_PATH)
    if os.path.exists(SCHEMA_JSON):
        os.remove(SCHEMA_JSON)
    st.session_state.schema_loaded = False
    st.rerun()

if not st.session_state.schema_loaded:
    if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(SCHEMA_JSON):
        with st.spinner("📚 Indexing full schema..."):
            chunks = get_schema_chunks(db_config)
            index_schema_chunks(chunks)
    st.session_state.schema_loaded = True

# Input form
with st.form("query_form"):
    user_query = st.text_input("🔍 Enter your natural language question:")
    submitted = st.form_submit_button("Generate & Run")

if submitted and user_query:
    st.session_state.last_query = user_query
    st.session_state.initial_user_query = user_query  # Store initial query
    with st.spinner("🔍 Retrieving schema & generating SQL..."):
        top_chunks = retrieve_top_k_chunks(user_query)
        generated_sql = generate_sql_from_gemini(top_chunks, user_query)
        df, err = run_sql(generated_sql, db_config)

        if err:
            st.error(f"❌ Database Error: {err}")
            st.session_state.generated_sql = generated_sql  # Still show the generated SQL even if error
            st.session_state.query_result = pd.DataFrame()  # Clear previous results
        else:
            st.session_state.generated_sql = generated_sql
            st.session_state.query_result = df
            st.session_state.edit_mode = False

if st.session_state.generated_sql:
    st.subheader("🧬 SQL Generated")

    if st.session_state.edit_mode:
        # Pass the original generated_sql to extract_table_name for display purposes
        current_table_display = extract_table_name(st.session_state.generated_sql)
        all_tables = [chunk.split('(')[0] for chunk in get_schema_chunks(db_config)]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.code(st.session_state.generated_sql, language="sql")
        with col2:
            st.session_state.corrected_table = st.selectbox(
                "Select correct table:",
                options=all_tables,
                # Try to find the exact table name for initial selection, otherwise default to first
                index=all_tables.index(current_table_display) if current_table_display in all_tables else
                (all_tables.index(current_table_display.upper()) if current_table_display.upper() in all_tables else
                 (all_tables.index(
                     current_table_display.lower()) if current_table_display.lower() in all_tables else 0))
            )

            if st.button("Apply Correction"):
                # *** Crucial Change: Re-generate SQL with the new table as a hint ***
                with st.spinner(f"Re-generating SQL for '{st.session_state.corrected_table}'..."):
                    # Use the initial user query and hint for the new table
                    top_chunks_re = retrieve_top_k_chunks(st.session_state.initial_user_query,
                                                          hint_table=st.session_state.corrected_table)
                    corrected_sql_re = generate_sql_from_gemini(top_chunks_re, st.session_state.initial_user_query)

                    df, err = run_sql(corrected_sql_re, db_config)
                    if err:
                        st.error(f"❌ Database Error: {err}")
                        st.session_state.generated_sql = corrected_sql_re  # Show the new SQL with error
                        st.session_state.query_result = pd.DataFrame()  # Clear results
                    else:
                        st.session_state.generated_sql = corrected_sql_re
                        st.session_state.query_result = df
                        st.session_state.edit_mode = False
                        st.toast("✅ SQL corrected and executed!")
                    st.rerun()  # Rerun to update the display immediately after applying correction

            if st.button("Cancel"):
                st.session_state.edit_mode = False
                st.rerun()
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.code(st.session_state.generated_sql, language="sql")
        with col2:
            if st.button("✏️ Edit Table"):
                st.session_state.edit_mode = True
                st.rerun()

if not st.session_state.query_result.empty:
    st.subheader("📋 Query Result")
    st.dataframe(st.session_state.query_result, use_container_width=True)

    st.subheader("📊 Visualize Data")
    chart_type = st.selectbox("Chart Type", ["Bar Chart", "Pie Chart"], key="chart")
    df = st.session_state.query_result

    if df.shape[1] >= 2:
        x_col = st.selectbox("X-axis", df.columns, key="xcol")
        y_col = st.selectbox("Y-axis (numeric)", df.select_dtypes(include='number').columns, key="ycol")

        # --- IMPORTANT CHANGE HERE ---
        if x_col and y_col:  # Ensure both are selected
            if x_col == y_col:
                st.warning("X-axis and Y-axis cannot be the same. Please select different columns.")
            elif chart_type == "Bar Chart":
                if not df.empty:
                    st.bar_chart(df.set_index(x_col)[y_col])
                else:
                    st.warning("No data to plot for Bar Chart.")
            elif chart_type == "Pie Chart":
                if not df.empty:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    data = df.groupby(x_col)[y_col].sum()
                    ax.pie(data, labels=data.index, autopct='%1.1f%%')
                    st.pyplot(fig)
                else:
                    st.warning("No data to plot for Pie Chart.")
        # --- END OF IMPORTANT CHANGE ---
    else:
        st.info("Select at least two columns to visualize data.")

    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)

    st.download_button(
        label="📅 Download Excel",
        data=excel_buffer,
        file_name="query_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )