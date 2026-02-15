"""
Smart NLP-to-SQL App with Embeddings-Based Table Selection (MySQL)
==================================================================
This app uses vector embeddings to find relevant tables in MySQL databases:

1. On connect: Create embeddings for each table's description
2. On query: Find similar tables via vector search (fast!)
3. Generate SQL with only those tables

Optimized for large MySQL databases with 100+ tables!
"""

import gradio as gr
import pandas as pd
import re
import json
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import mysql.connector
import chromadb
from chromadb.config import Settings

# Global state
mysql_config = None
engine = None
all_tables = []
table_schemas = {}
llm = None
embeddings = None
chroma_client = None
collection = None


def get_llm():
    """Get or create the LLM instance"""
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


def get_embeddings():
    """Get or create the embeddings instance"""
    global embeddings
    if embeddings is None:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            api_version="2024-08-01-preview",
            azure_endpoint="https://elunic-stulzgpt-openai.openai.azure.com/",
            api_key="CoFdfEEVHlFmjmf0jaEPTL2c2kp5R7FI9p1BhjYpq9rJlfNfc7vMJQQJ99BJACfhMk5XJ3w3AAABACOGYmBN",
        )
    return embeddings


def setup_chromadb():
    """Setup ChromaDB for vector storage"""
    global chroma_client, collection

    chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))

    try:
        chroma_client.delete_collection("table_schemas")
    except:
        pass

    collection = chroma_client.create_collection(
        name="table_schemas",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def create_table_description(table_name: str, columns: list) -> str:
    """Create a searchable description for a table"""
    col_info = ", ".join([f"{c['name']} ({c['type']})" for c in columns])
    return f"Table '{table_name}' with columns: {col_info}"


def connect_database(host, port, user, password, database):
    """Connect to MySQL and create embeddings for all tables"""
    global mysql_config, engine, all_tables, table_schemas, collection

    try:
        mysql_config = {
            "host": host,
            "port": int(port) if port else 3306,
            "user": user,
            "password": password,
            "database": database
        }

        # Test connection
        print(f"[CONNECT] Connecting to {host}:{port}/{database}...")
        test_conn = mysql.connector.connect(**mysql_config)
        cursor = test_conn.cursor()

        # Get all tables
        cursor.execute("SHOW TABLES")
        all_tables = [row[0] for row in cursor.fetchall()]
        print(f"[CONNECT] Found {len(all_tables)} tables")

        # Get schema for each table
        table_schemas = {}
        for table in all_tables:
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{database}' AND TABLE_NAME = '{table}'
            """)
            columns = cursor.fetchall()
            table_schemas[table] = {
                "columns": [{"name": col[0], "type": col[1]} for col in columns],
                "column_names": [col[0] for col in columns]
            }

        test_conn.close()

        # Create SQLAlchemy engine
        connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}:{mysql_config['port']}/{database}"
        engine = create_engine(connection_string)

        # Setup ChromaDB and create embeddings
        collection = setup_chromadb()

        print(f"[EMBEDDINGS] Creating embeddings for {len(all_tables)} tables...")

        documents = []
        ids = []
        metadatas = []

        for table in all_tables:
            description = create_table_description(table, table_schemas[table]["columns"])
            documents.append(description)
            ids.append(table)
            metadatas.append({
                "table_name": table,
                "num_columns": len(table_schemas[table]["columns"])
            })

        # Get embeddings from Azure OpenAI
        try:
            embed_model = get_embeddings()
            vectors = embed_model.embed_documents(documents)

            collection.add(
                documents=documents,
                embeddings=vectors,
                ids=ids,
                metadatas=metadatas
            )
            embed_status = f"✅ Embeddings created for {len(all_tables)} tables"
            print(f"[EMBEDDINGS] Success!")

        except Exception as e:
            embed_status = f"⚠️ Embedding failed: {str(e)[:100]}... (will use LLM fallback)"
            print(f"[EMBEDDINGS] Error: {e}")

        # Build summary
        summary = f"🐬 Connected to {database}\n"
        summary += f"📊 {len(all_tables)} tables found\n"
        summary += f"{embed_status}\n\n"
        summary += "Sample tables:\n"

        for t in all_tables[:10]:
            cols = ", ".join(table_schemas[t]["column_names"][:4])
            if len(table_schemas[t]["column_names"]) > 4:
                cols += "..."
            summary += f"  • {t}: {cols}\n"

        if len(all_tables) > 10:
            summary += f"  ... and {len(all_tables) - 10} more"

        return (
            gr.update(choices=all_tables, value=all_tables[0] if all_tables else None),
            summary
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return gr.update(choices=[], value=None), f"❌ Connection failed: {str(e)}"


def show_table(table_name):
    """Show schema and data for selected table"""
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


def find_relevant_tables_embedding(question: str, top_k: int = 5) -> list:
    """Find relevant tables using embedding similarity (FAST!)"""
    global collection, all_tables

    if collection is None or collection.count() == 0:
        print("[SEARCH] No embeddings available")
        return all_tables[:5]

    try:
        embed_model = get_embeddings()
        question_embedding = embed_model.embed_query(question)

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(top_k, len(all_tables))
        )

        relevant_tables = results['ids'][0] if results['ids'] else []
        distances = results['distances'][0] if results['distances'] else []

        print(f"[SEARCH] Embedding search found: {list(zip(relevant_tables, distances))}")

        return relevant_tables

    except Exception as e:
        print(f"[SEARCH] Embedding search error: {e}")
        return all_tables[:5]


def find_relevant_tables_llm(question: str) -> list:
    """Fallback: Use LLM to find relevant tables"""
    global all_tables, table_schemas

    table_summary = "Available tables:\n"
    for table in all_tables:
        cols = ", ".join(table_schemas[table]["column_names"][:10])
        if len(table_schemas[table]["column_names"]) > 10:
            cols += "..."
        table_summary += f"- {table}: {cols}\n"

    prompt = f"""Given these database tables:

{table_summary}

Question: {question}

Which tables are needed? Return ONLY a JSON array of table names.
Example: ["table1", "table2"]
"""

    print(f"[SEARCH] Using LLM fallback...")
    response = get_llm().invoke(prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)

    try:
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            tables = json.loads(match.group())
            valid_tables = [t for t in tables if t in all_tables]
            print(f"[SEARCH] LLM selected: {valid_tables}")
            return valid_tables
    except:
        pass

    return all_tables[:5]


def generate_sql(question: str, relevant_tables: list) -> str:
    """Generate SQL using only the relevant tables"""
    global engine

    print(f"[SQL] Generating with tables: {relevant_tables}")

    db = SQLDatabase(
        engine,
        include_tables=relevant_tables,
        sample_rows_in_table_info=1,
    )

    query_chain = create_sql_query_chain(get_llm(), db)
    return query_chain.invoke({"question": question})


def extract_sql(sql_response) -> str:
    """Extract clean SQL from LLM response"""
    if hasattr(sql_response, 'content'):
        text = sql_response.content
    else:
        text = str(sql_response)

    text = text.strip()

    fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    upper_text = text.upper()
    for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']:
        if keyword in upper_text:
            idx = upper_text.index(keyword)
            sql = text[idx:]
            for stopper in ['\n\n', '\nSQLResult', '\nAnswer', '\nNote:']:
                if stopper in sql:
                    sql = sql.split(stopper)[0]
            return sql.strip()

    return text


def run_sql(query: str) -> pd.DataFrame:
    """Execute SQL and return results"""
    global mysql_config
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description] if cursor.description else []
    conn.close()
    return pd.DataFrame(rows, columns=col_names)


def ask_database(question: str, use_embeddings: bool = True):
    """Main function: Find tables and generate SQL"""
    global engine, all_tables, collection

    if not engine or not all_tables:
        return "❌ Connect to a database first!", "", pd.DataFrame()

    try:
        import time
        start = time.time()

        # ===== STAGE 1: Find relevant tables =====
        stage1_start = time.time()

        if use_embeddings and collection and collection.count() > 0:
            relevant_tables = find_relevant_tables_embedding(question, top_k=5)
            method = "🔍 Embeddings"
        else:
            relevant_tables = find_relevant_tables_llm(question)
            method = "🤖 LLM"

        stage1_time = time.time() - stage1_start

        if not relevant_tables:
            return "❌ No relevant tables found", "", pd.DataFrame()

        # ===== STAGE 2: Generate SQL =====
        stage2_start = time.time()
        sql_response = generate_sql(question, relevant_tables)
        clean_sql = extract_sql(sql_response)
        stage2_time = time.time() - stage2_start

        # Failsafe
        if "SQLQuery" in clean_sql and "SELECT" in clean_sql.upper():
            idx = clean_sql.upper().index("SELECT")
            clean_sql = clean_sql[idx:]

        # ===== Execute SQL =====
        exec_start = time.time()
        df = run_sql(clean_sql)
        exec_time = time.time() - exec_start

        total_time = time.time() - start

        info = f"""📋 Tables: {', '.join(relevant_tables)}
{method} search: {stage1_time:.3f}s
🔧 SQL generation: {stage2_time:.2f}s
⚡ Query execution: {exec_time:.3f}s
⏱️ Total: {total_time:.2f}s"""

        return info, clean_sql, df

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", "", pd.DataFrame()


# ---- Gradio UI ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🚀 Smart MySQL Explorer with Embeddings")
    gr.Markdown("""
    **Optimized for large databases (100+ tables):**
    1. **On Connect**: Creates vector embeddings for all table descriptions
    2. **On Query**: Fast vector search finds relevant tables (~0.1s)
    3. **Generate**: SQL created using only matched tables

    ⚡ Works efficiently even with 142+ tables!
    """)

    with gr.Row():
        # --- Column 1: Connection & Tables ---
        with gr.Column(scale=1):
            gr.Markdown("### 🔌 MySQL Connection")
            with gr.Group():
                host_input = gr.Textbox(label="Host", value="zkw-test-mysql.mysql.database.azure.com")
                port_input = gr.Textbox(label="Port", value="3306")
                user_input = gr.Textbox(label="Username", value="mysqladmin")
                password_input = gr.Textbox(label="Password", type="password", value="ZkwTest2026Pass1")
                database_input = gr.Textbox(label="Database", value="zkwmbdb_global_elunic")

            connect_btn = gr.Button("🔗 Connect & Create Embeddings", variant="primary")
            db_status = gr.Textbox(label="Connection Status", lines=10, interactive=False)

            gr.Markdown("### 📋 Browse Tables")
            tables = gr.Dropdown(label="Select Table", choices=[], interactive=True)
            schema_out = gr.Dataframe(label="Schema")
            data_out = gr.Dataframe(label="Data (50 rows)")

        # --- Column 2: AI Assistant ---
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            question = gr.Textbox(
                label="Ask a question in natural language",
                placeholder="e.g., Show me all records from the users table",
                lines=2
            )

            with gr.Row():
                ask_btn = gr.Button("🚀 Generate & Run SQL", variant="primary", size="lg")
                use_embed = gr.Checkbox(
                    label="Use Embeddings",
                    value=True,
                    info="Uncheck to use LLM-based table search"
                )

            stage_info = gr.Textbox(label="Performance Info", lines=5, interactive=False)
            sql_out = gr.Code(label="Generated SQL", language="sql")
            result_out = gr.Dataframe(label="Query Result", wrap=True)

    # --- Events ---
    connect_btn.click(
        connect_database,
        inputs=[host_input, port_input, user_input, password_input, database_input],
        outputs=[tables, db_status]
    )
    tables.change(show_table, inputs=tables, outputs=[schema_out, data_out])
    ask_btn.click(
        ask_database,
        inputs=[question, use_embed],
        outputs=[stage_info, sql_out, result_out]
    )

demo.launch(server_name="0.0.0.0", server_port=7865)
