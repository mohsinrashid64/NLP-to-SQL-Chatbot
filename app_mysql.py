import gradio as gr
import pandas as pd
import re
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import AzureChatOpenAI
import mysql.connector

db = None
query_chain = None
mysql_config = None

# ---- Connect to MySQL Database ----
def connect_database(host, port, user, password, database):
    global db, query_chain, mysql_config
    try:
        # Store config for later use
        mysql_config = {
            "host": host,
            "port": int(port) if port else 3306,
            "user": user,
            "password": password,
            "database": database
        }

        # Test connection first
        test_conn = mysql.connector.connect(**mysql_config)
        test_conn.close()

        # Setup SQLAlchemy + LangChain DB
        connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}:{mysql_config['port']}/{database}"
        engine = create_engine(connection_string)
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
        conn = mysql.connector.connect(**mysql_config)
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        return (
            gr.update(choices=tables, value=tables[0] if tables else None),
            f"✅ Connected to {database} ({len(tables)} tables found)"
        )
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ Connection failed: {str(e)}"


# ---- Show Schema + Data ----
def show_table(table_name):
    global mysql_config
    if not mysql_config or not table_name:
        return pd.DataFrame(), pd.DataFrame()
    try:
        conn = mysql.connector.connect(**mysql_config)

        # Get schema
        schema_query = f"""
            SELECT
                COLUMN_NAME as 'Column',
                DATA_TYPE as 'Type',
                IS_NULLABLE as 'Nullable',
                COLUMN_KEY as 'Key',
                COLUMN_DEFAULT as 'Default'
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{mysql_config['database']}'
            AND TABLE_NAME = '{table_name}'
        """
        schema = pd.read_sql(schema_query, conn)

        # Get data
        data = pd.read_sql(f"SELECT * FROM `{table_name}` LIMIT 50", conn)
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
        return "❌ Connect to a database first!", pd.DataFrame()

    try:
        sql_response = query_chain.invoke({"question": question})
        print(f"[DEBUG] query_chain returned type: {type(sql_response)}")
        print(f"[DEBUG] query_chain returned: {repr(sql_response)}")

        clean_sql = extract_sql(sql_response)

        # FAILSAFE: If "SQLQuery" is still in the SQL, something went wrong
        if "SQLQuery" in clean_sql:
            print(f"[DEBUG] FAILSAFE TRIGGERED - SQLQuery still in SQL!")
            if "SELECT" in clean_sql.upper():
                idx = clean_sql.upper().index("SELECT")
                clean_sql = clean_sql[idx:]
                print(f"[DEBUG] After failsafe extraction: {clean_sql}")

        print(f"[DEBUG] Final SQL to execute: {repr(clean_sql)}")

        # Use helper to get results with headers
        df = run_sql_with_headers(clean_sql)

        return clean_sql, df

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", pd.DataFrame()


# ---- Gradio UI ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🐬 AI-Powered MySQL Database Explorer")

    with gr.Row():
        # --- Column 1: DB Connection + Schema/Data ---
        with gr.Column(scale=1):
            gr.Markdown("### 🔌 MySQL Connection")
            with gr.Group():
                host_input = gr.Textbox(label="Host", value="zkw-test-mysql.mysql.database.azure.com", placeholder="localhost")
                port_input = gr.Textbox(label="Port", value="3306", placeholder="3306")
                user_input = gr.Textbox(label="Username", value="mysqladmin", placeholder="root")
                password_input = gr.Textbox(label="Password", type="password", value="ZkwTest2026Pass1", placeholder="Enter password")
                database_input = gr.Textbox(label="Database Name", value="zkwmbdb_global_elunic", placeholder="e.g., ZKW Test")

            connect_btn = gr.Button("Connect to Database", variant="primary")
            connection_status = gr.Textbox(label="Status", interactive=False)

            tables = gr.Dropdown(label="Tables", choices=[], interactive=True)

            gr.Markdown("### 🏗 Schema")
            schema_out = gr.Dataframe(label="Table Schema")
            gr.Markdown("### 📊 Data (first 50 rows)")
            data_out = gr.Dataframe(label="Table Data")

        # --- Column 2: AI SQL Assistant ---
        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Ask a question in natural language",
                placeholder="e.g., Show me all customers from Germany"
            )
            ask_btn = gr.Button("Generate & Run SQL", variant="primary")
            sql_out = gr.Code(label="Generated SQL Query", language="sql")
            result_out = gr.Dataframe(label="Query Result", wrap=True)

    # --- Events ---
    connect_btn.click(
        connect_database,
        inputs=[host_input, port_input, user_input, password_input, database_input],
        outputs=[tables, connection_status]
    )
    tables.change(show_table, inputs=tables, outputs=[schema_out, data_out])
    ask_btn.click(ask_database, inputs=question, outputs=[sql_out, result_out])

demo.launch(server_name="0.0.0.0", server_port=7861)
