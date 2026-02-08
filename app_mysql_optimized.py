import gradio as gr
import pandas as pd
import re
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import AzureChatOpenAI
import mysql.connector

db = None
query_chain = None
mysql_config = None
engine = None
all_tables = []

# ---- Connect to MySQL Database (Step 1: Just connect, don't load LLM yet) ----
def connect_database(host, port, user, password, database):
    global mysql_config, engine, all_tables
    try:
        mysql_config = {
            "host": host,
            "port": int(port) if port else 3306,
            "user": user,
            "password": password,
            "database": database
        }

        # Test connection
        test_conn = mysql.connector.connect(**mysql_config)
        cursor = test_conn.cursor()
        cursor.execute("SHOW TABLES")
        all_tables = [row[0] for row in cursor.fetchall()]
        test_conn.close()

        # Create engine (but don't create SQLDatabase yet - wait for table selection)
        connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}:{mysql_config['port']}/{database}"
        engine = create_engine(connection_string)

        return (
            gr.update(choices=all_tables, value=all_tables[:5] if len(all_tables) >= 5 else all_tables),
            f"✅ Connected! {len(all_tables)} tables found. Select tables below, then click 'Initialize AI'."
        )
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ Connection failed: {str(e)}"


# ---- Initialize AI with selected tables only (Step 2) ----
def initialize_ai(selected_tables):
    global db, query_chain, engine

    if not engine:
        return "❌ Connect to database first!", gr.update(choices=[], value=None)

    if not selected_tables or len(selected_tables) == 0:
        return "❌ Select at least one table!", gr.update(choices=[], value=None)

    try:
        print(f"[INFO] Initializing AI with {len(selected_tables)} tables: {selected_tables}")

        # Create SQLDatabase with ONLY selected tables and minimal sample rows
        db = SQLDatabase(
            engine,
            include_tables=selected_tables,
            sample_rows_in_table_info=1,  # Only 1 sample row (faster)
        )

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

        return (
            f"✅ AI Ready! Using {len(selected_tables)} tables.",
            gr.update(choices=selected_tables, value=selected_tables[0] if selected_tables else None)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", gr.update(choices=[], value=None)


# ---- Show Schema + Data ----
def show_table(table_name):
    global mysql_config
    if not mysql_config or not table_name:
        return pd.DataFrame(), pd.DataFrame()
    try:
        conn = mysql.connector.connect(**mysql_config)
        schema_query = f"""
            SELECT
                COLUMN_NAME as 'Column',
                DATA_TYPE as 'Type',
                IS_NULLABLE as 'Nullable',
                COLUMN_KEY as 'Key'
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{mysql_config['database']}'
            AND TABLE_NAME = '{table_name}'
        """
        schema = pd.read_sql(schema_query, conn)
        data = pd.read_sql(f"SELECT * FROM `{table_name}` LIMIT 50", conn)
        conn.close()
        return schema, data
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]}), pd.DataFrame()


def extract_sql(sql_response) -> str:
    """Extracts a clean SQL query from the LLM response."""
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

    # Find where SQL starts
    upper_text = text.upper()
    for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']:
        if keyword in upper_text:
            idx = upper_text.index(keyword)
            sql = text[idx:]
            for stopper in ['\n\n', '\nSQLResult', '\nAnswer', '\nNote:', '\nExplanation']:
                if stopper in sql:
                    sql = sql.split(stopper)[0]
            sql = sql.strip()
            print(f"[DEBUG] Extracted SQL: {sql}")
            return sql

    return text


def run_sql_with_headers(query):
    """Run SQL and return Pandas DataFrame with headers."""
    global mysql_config
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description] if cursor.description else []
    conn.close()
    return pd.DataFrame(rows, columns=col_names)


# ---- AI → SQL → Result ----
def ask_database(question):
    global query_chain, db
    if not query_chain or not db:
        return "❌ Initialize AI first! Select tables and click 'Initialize AI'", pd.DataFrame()

    try:
        import time
        start = time.time()

        sql_response = query_chain.invoke({"question": question})

        llm_time = time.time() - start
        print(f"[TIMING] LLM response took {llm_time:.2f}s")

        clean_sql = extract_sql(sql_response)

        if "SQLQuery" in clean_sql:
            if "SELECT" in clean_sql.upper():
                idx = clean_sql.upper().index("SELECT")
                clean_sql = clean_sql[idx:]

        print(f"[DEBUG] Final SQL: {repr(clean_sql)}")

        df = run_sql_with_headers(clean_sql)

        total_time = time.time() - start
        print(f"[TIMING] Total time: {total_time:.2f}s")

        return clean_sql, df

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", pd.DataFrame()


# ---- Gradio UI ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🐬 AI-Powered MySQL Database Explorer (Optimized)")
    gr.Markdown("⚡ **Tip:** Select only the tables you need to query for faster responses!")

    with gr.Row():
        # --- Column 1: DB Connection + Schema/Data ---
        with gr.Column(scale=1):
            gr.Markdown("### 🔌 Step 1: Connect to MySQL")
            with gr.Group():
                host_input = gr.Textbox(label="Host", value="zkw-test-mysql.mysql.database.azure.com")
                port_input = gr.Textbox(label="Port", value="3306")
                user_input = gr.Textbox(label="Username", value="mysqladmin")
                password_input = gr.Textbox(label="Password", type="password", value="ZkwTest2026Pass1")
                database_input = gr.Textbox(label="Database", value="zkwmbdb_global_elunic")

            connect_btn = gr.Button("🔗 Connect to Database", variant="primary")
            connection_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### 📋 Step 2: Select Tables for AI")
            table_select = gr.Dropdown(
                label="Select Tables (multi-select)",
                choices=[],
                multiselect=True,
                interactive=True,
                info="Only selected tables will be used by AI"
            )
            init_btn = gr.Button("🤖 Initialize AI", variant="primary")
            ai_status = gr.Textbox(label="AI Status", interactive=False)

            gr.Markdown("### 🏗 Browse Tables")
            browse_table = gr.Dropdown(label="View Table", choices=[], interactive=True)
            schema_out = gr.Dataframe(label="Schema")
            data_out = gr.Dataframe(label="Data (50 rows)")

        # --- Column 2: AI SQL Assistant ---
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Step 3: Ask Questions")
            question = gr.Textbox(
                label="Ask a question in natural language",
                placeholder="e.g., Show me all records from the first table",
                lines=2
            )
            ask_btn = gr.Button("🚀 Generate & Run SQL", variant="primary", size="lg")
            sql_out = gr.Code(label="Generated SQL Query", language="sql")
            result_out = gr.Dataframe(label="Query Result", wrap=True)

    # --- Events ---
    connect_btn.click(
        connect_database,
        inputs=[host_input, port_input, user_input, password_input, database_input],
        outputs=[table_select, connection_status]
    )

    init_btn.click(
        initialize_ai,
        inputs=[table_select],
        outputs=[ai_status, browse_table]
    )

    browse_table.change(show_table, inputs=browse_table, outputs=[schema_out, data_out])
    ask_btn.click(ask_database, inputs=question, outputs=[sql_out, result_out])

demo.launch(server_name="0.0.0.0", server_port=7862)
