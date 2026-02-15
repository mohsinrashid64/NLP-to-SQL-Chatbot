import gradio as gr
import sqlite3
import pandas as pd
import re
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import AzureChatOpenAI

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
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2024-08-01-preview",
            azure_endpoint="https://elunic-stulzgpt-openai.openai.azure.com/",
            api_key="CoFdfEEVHlFmjmf0jaEPTL2c2kp5R7FI9p1BhjYpq9rJlfNfc7vMJQQJ99BJACfhMk5XJ3w3AAABACOGYmBN",
            timeout=60,
            max_retries=3,
            temperature=0,
        )
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


def extract_sql(sql_response) -> str:
    """Extracts a clean SQL query from the LLM response."""
    # Handle AIMessage or other objects
    if hasattr(sql_response, 'content'):
        text = sql_response.content
    else:
        text = str(sql_response)

    text = text.strip()
    print(f"[DEBUG] Raw LLM response:\n{repr(text)}\n{'='*50}")

    # If inside triple backticks, grab content
    fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        result = fence_match.group(1).strip()
        print(f"[DEBUG] Extracted from code block: {result}")
        return result

    # Simple approach: find where the SQL actually starts
    upper_text = text.upper()
    for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']:
        if keyword in upper_text:
            idx = upper_text.index(keyword)
            sql = text[idx:]  # Take everything from the SQL keyword onwards
            # Remove anything after the SQL
            for stopper in ['\n\n', '\nSQLResult', '\nAnswer', '\nNote:', '\nExplanation']:
                if stopper in sql:
                    sql = sql.split(stopper)[0]
            sql = sql.strip()
            print(f"[DEBUG] Extracted SQL starting from {keyword}: {sql}")
            return sql

    print(f"[DEBUG] No SQL keyword found, returning text as-is")
    return text


def run_sql_with_headers(db_path, query):
    """Run SQL and return Pandas DataFrame with headers."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description] if cursor.description else []
    conn.close()
    return pd.DataFrame(rows, columns=col_names)


# ---- AI → SQL → Result ----
def ask_database(question):
    global query_chain, db, db_path
    if not query_chain or not db:
        return "❌ Load a database first!", pd.DataFrame()

    try:
        sql_response = query_chain.invoke({"question": question})
        print(f"[DEBUG] query_chain returned type: {type(sql_response)}")
        print(f"[DEBUG] query_chain returned: {repr(sql_response)}")

        clean_sql = extract_sql(sql_response)

        # FAILSAFE: If "SQLQuery" is still in the SQL, something went wrong
        if "SQLQuery" in clean_sql:
            print(f"[DEBUG] FAILSAFE TRIGGERED - SQLQuery still in SQL!")
            # Force extract from SELECT
            if "SELECT" in clean_sql.upper():
                idx = clean_sql.upper().index("SELECT")
                clean_sql = clean_sql[idx:]
                print(f"[DEBUG] After failsafe extraction: {clean_sql}")

        print(f"[DEBUG] Final SQL to execute: {repr(clean_sql)}")

        # Use helper to get results with headers
        df = run_sql_with_headers(db_path, clean_sql)

        return clean_sql, df

    except Exception as e:
        import traceback
        traceback.print_exc()
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

# demo.launch()
demo.launch(server_name="0.0.0.0", server_port=7860)
# demo.launch(share=True) # For Internet Live Link for one week.


