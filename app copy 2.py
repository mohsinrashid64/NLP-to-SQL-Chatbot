import gradio as gr
import sqlite3
import pandas as pd
import re
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.llms import Ollama

db_path = None
db = None
query_chain = None

# ---- Load Database ----
def load_database(file):
    global db_path, db, query_chain
    try:
        if isinstance(file, dict):  # Gradio v4
            db_path = file["name"]
        else:
            db_path = file.name

        # Setup SQLAlchemy + LangChain DB
        engine = create_engine(f"sqlite:///{db_path}")
        db = SQLDatabase(engine)

        # Setup LLM + SQL chain
        llm = Ollama(model="qwen2.5:7b", base_url="http://127.0.0.1:11434")
        query_chain = create_sql_query_chain(llm, db)

        # Get tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        return gr.update(choices=tables, value=tables[0])
    except Exception as e:
        return gr.update(choices=[], value=None)

# ---- Show Schema + Data ----
def show_table(table_name):
    if not db_path or not table_name:
        return pd.DataFrame(), pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        schema = pd.read_sql_query(f"PRAGMA table_info({table_name});", conn)
        data = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 50;", conn)
        conn.close()
        return schema, data
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]}), pd.DataFrame()


def extract_sql(sql_response: str) -> str:
    """Extracts a clean SQL query from the LLM response."""
    # Remove leading/trailing whitespace
    text = sql_response.strip()

    # If inside triple backticks, grab content
    fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Otherwise look for SQLQuery: prefix
    query_match = re.search(r"SQLQuery:\s*([\s\S]*)", text, re.IGNORECASE)
    if query_match:
        return query_match.group(1).strip()

    # If nothing matches, just return full text (last fallback)
    return text



# ---- AI → SQL → Result ----
def ask_database(question):
    global query_chain, db
    if not query_chain or not db:
        return "❌ Load a database first!", pd.DataFrame()

    try:
        sql_response = query_chain.invoke({"question": question})
        clean_sql = extract_sql(sql_response)

        # Execute SQL
        sql_result = db.run(clean_sql)

        # Convert to DataFrame
        if isinstance(sql_result, list):
            df = pd.DataFrame(sql_result)
        elif isinstance(sql_result, str):
            df = pd.DataFrame([{"result": sql_result}])
        else:
            df = pd.DataFrame(sql_result)

        return clean_sql, df

    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame()


# ---- Gradio UI ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📂 AI-Powered SQLite Database Explorer")

    with gr.Row():
        # --- Column 1: DB Upload + Schema/Data ---
        with gr.Column(scale=1):
            db_file = gr.File(label="Upload SQLite Database", file_types=[".db", ".sqlite"])
            load_btn = gr.Button("Load Database", variant="primary")
            tables = gr.Dropdown(label="Tables", choices=[], interactive=True)

            gr.Markdown("### 🏗 Schema")
            schema_out = gr.Dataframe(label="Table Schema")
            gr.Markdown("### 📊 Data (first 50 rows)")
            data_out = gr.Dataframe(label="Table Data")

        # --- Column 2: AI SQL Assistant ---
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Ask a question in natural language",
                placeholder="e.g. Do people with diabetes have higher blood pressure?"
            )
            ask_btn = gr.Button("Generate & Run SQL", variant="primary")
            sql_out = gr.Code(label="Generated SQL Query", language="sql")
            result_out = gr.Dataframe(label="Query Result", wrap=True)

    # --- Events ---
    load_btn.click(load_database, inputs=db_file, outputs=[tables])
    tables.change(show_table, inputs=tables, outputs=[schema_out, data_out])
    ask_btn.click(ask_database, inputs=question, outputs=[sql_out, result_out])

demo.launch()
