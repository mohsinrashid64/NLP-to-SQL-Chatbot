"""
Smart NLP-to-SQL App with Two-Stage Table Selection
====================================================
This app uses a two-stage LLM approach to handle databases with many tables:

Stage 1: Ask LLM which tables are relevant to the question (fast, cheap)
Stage 2: Load only those tables and generate SQL (accurate, efficient)

This makes queries fast even with 100+ tables!
"""

import gradio as gr
import sqlite3
import pandas as pd
import re
import json
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import AzureChatOpenAI

# Global state
db_path = None
engine = None
all_tables = []
table_schemas = {}  # Cache: {table_name: schema_info}
llm = None


def get_llm():
    """Get or create the LLM instance (singleton)"""
    global llm
    if llm is None:
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2024-08-01-preview",
            azure_endpoint="https://elunic-stulzgpt-openai.openai.azure.com/",
            api_key="CoFdfEEVHlFmjmf0jaEPTL2c2kp5R7FI9p1BhjYpq9rJlfNfc7vMJQQJ99BJACfhMk5XJ3w3AAABACOGYmBN",
            timeout=60,
            max_retries=3,
            temperature=0,
        )
    return llm


def load_database(file):
    """Load SQLite database and cache table information"""
    global db_path, engine, all_tables, table_schemas

    try:
        if isinstance(file, dict):
            db_path = file["name"]
        else:
            db_path = file.name

        # Create SQLAlchemy engine
        engine = create_engine(f"sqlite:///{db_path}")

        # Get all table names
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_tables = [row[0] for row in cursor.fetchall()]

        # Cache schema for each table (lightweight - just column info)
        table_schemas = {}
        for table in all_tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            table_schemas[table] = {
                "columns": [{"name": col[1], "type": col[2]} for col in columns],
                "column_names": [col[1] for col in columns]
            }

        conn.close()

        # Build table summary for display
        summary = f"✅ Loaded {len(all_tables)} tables:\n"
        for t in all_tables:
            cols = ", ".join(table_schemas[t]["column_names"][:5])
            if len(table_schemas[t]["column_names"]) > 5:
                cols += "..."
            summary += f"  • {t} ({len(table_schemas[t]['columns'])} cols): {cols}\n"

        return (
            gr.update(choices=all_tables, value=all_tables[0] if all_tables else None),
            summary
        )
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ Error: {str(e)}"


def show_table(table_name):
    """Show schema and data for selected table"""
    global db_path
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


def stage1_find_relevant_tables(question: str) -> list:
    """
    STAGE 1: Ask LLM which tables are relevant to the question.
    This is fast because we only send table names + column names, not full schemas.
    """
    global all_tables, table_schemas

    # Build a compact table summary
    table_summary = "Available tables and their columns:\n"
    for table in all_tables:
        cols = ", ".join(table_schemas[table]["column_names"])
        table_summary += f"- {table}: {cols}\n"

    prompt = f"""Given these database tables:

{table_summary}

Question: {question}

Which tables are needed to answer this question?
Return ONLY a JSON array of table names, nothing else.
Example: ["table1", "table2"]
"""

    print(f"[STAGE 1] Finding relevant tables...")
    print(f"[STAGE 1] Prompt length: {len(prompt)} chars")

    response = get_llm().invoke(prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)

    print(f"[STAGE 1] Response: {response_text}")

    # Parse JSON response
    try:
        # Extract JSON array from response
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            tables = json.loads(match.group())
            # Validate tables exist
            valid_tables = [t for t in tables if t in all_tables]
            print(f"[STAGE 1] Selected tables: {valid_tables}")
            return valid_tables
    except json.JSONDecodeError:
        pass

    # Fallback: return all tables if parsing fails
    print(f"[STAGE 1] Failed to parse, using all tables")
    return all_tables


def stage2_generate_sql(question: str, relevant_tables: list) -> str:
    """
    STAGE 2: Generate SQL using only the relevant tables.
    This is accurate because we include full schema for selected tables only.
    """
    global engine

    print(f"[STAGE 2] Generating SQL with tables: {relevant_tables}")

    # Create SQLDatabase with only relevant tables
    db = SQLDatabase(
        engine,
        include_tables=relevant_tables,
        sample_rows_in_table_info=1,
    )

    # Create chain and generate SQL
    query_chain = create_sql_query_chain(get_llm(), db)
    sql_response = query_chain.invoke({"question": question})

    print(f"[STAGE 2] Raw response: {repr(sql_response)}")

    return sql_response


def extract_sql(sql_response) -> str:
    """Extract clean SQL from LLM response"""
    if hasattr(sql_response, 'content'):
        text = sql_response.content
    else:
        text = str(sql_response)

    text = text.strip()

    # If inside triple backticks, grab content
    fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    # Find where SQL starts
    upper_text = text.upper()
    for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']:
        if keyword in upper_text:
            idx = upper_text.index(keyword)
            sql = text[idx:]
            for stopper in ['\n\n', '\nSQLResult', '\nAnswer', '\nNote:', '\nExplanation']:
                if stopper in sql:
                    sql = sql.split(stopper)[0]
            return sql.strip()

    return text


def run_sql(query: str) -> pd.DataFrame:
    """Execute SQL and return results as DataFrame"""
    global db_path
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description] if cursor.description else []
    conn.close()
    return pd.DataFrame(rows, columns=col_names)


def ask_database(question: str):
    """Main function: Two-stage approach to answer questions"""
    global engine, all_tables

    if not engine or not all_tables:
        return "❌ Load a database first!", "", pd.DataFrame()

    try:
        import time
        start = time.time()

        # ===== STAGE 1: Find relevant tables =====
        stage1_start = time.time()
        relevant_tables = stage1_find_relevant_tables(question)
        stage1_time = time.time() - stage1_start

        if not relevant_tables:
            return "❌ No relevant tables found", "", pd.DataFrame()

        tables_info = f"📋 Tables selected: {', '.join(relevant_tables)} ({stage1_time:.2f}s)"

        # ===== STAGE 2: Generate SQL =====
        stage2_start = time.time()
        sql_response = stage2_generate_sql(question, relevant_tables)
        clean_sql = extract_sql(sql_response)
        stage2_time = time.time() - stage2_start

        # Failsafe: remove SQLQuery prefix if still present
        if "SQLQuery" in clean_sql:
            if "SELECT" in clean_sql.upper():
                idx = clean_sql.upper().index("SELECT")
                clean_sql = clean_sql[idx:]

        print(f"[FINAL] SQL: {clean_sql}")

        # ===== Execute SQL =====
        df = run_sql(clean_sql)

        total_time = time.time() - start
        timing_info = f"\n⏱️ Stage 1: {stage1_time:.2f}s | Stage 2: {stage2_time:.2f}s | Total: {total_time:.2f}s"

        return tables_info + timing_info, clean_sql, df

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", "", pd.DataFrame()


# ---- Gradio UI ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🧠 Smart NLP-to-SQL (Two-Stage Approach)")
    gr.Markdown("""
    **How it works:**
    1. **Stage 1**: LLM identifies which tables are relevant to your question (fast)
    2. **Stage 2**: SQL is generated using only those tables (accurate)

    This makes queries fast even with 100+ tables!
    """)

    with gr.Row():
        # --- Column 1: Database ---
        with gr.Column(scale=1):
            db_file = gr.File(label="Upload SQLite Database", file_types=[".db", ".sqlite"])
            load_btn = gr.Button("📂 Load Database", variant="primary")
            db_status = gr.Textbox(label="Database Info", lines=6, interactive=False)

            tables = gr.Dropdown(label="Browse Tables", choices=[], interactive=True)
            schema_out = gr.Dataframe(label="Table Schema")
            data_out = gr.Dataframe(label="Sample Data (50 rows)")

        # --- Column 2: AI Assistant ---
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Ask a question in natural language",
                placeholder="e.g., What is the average age of patients with heart disease?",
                lines=2
            )
            ask_btn = gr.Button("🚀 Generate & Run SQL", variant="primary", size="lg")

            stage_info = gr.Textbox(label="Stage Info", interactive=False)
            sql_out = gr.Code(label="Generated SQL", language="sql")
            result_out = gr.Dataframe(label="Query Result", wrap=True)

    # --- Events ---
    load_btn.click(load_database, inputs=db_file, outputs=[tables, db_status])
    tables.change(show_table, inputs=tables, outputs=[schema_out, data_out])
    ask_btn.click(ask_database, inputs=question, outputs=[stage_info, sql_out, result_out])

demo.launch(server_name="0.0.0.0", server_port=7863)
