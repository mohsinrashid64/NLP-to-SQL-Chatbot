"""
Smart NLP-to-SQL App with Embeddings-Based Table Selection
==========================================================
This app uses vector embeddings to find relevant tables:

1. On load: Create embeddings for each table's description
2. On query: Find similar tables via vector search (fast!)
3. Generate SQL with only those tables

This is faster than LLM-based table selection for large databases!
"""

import gradio as gr
import sqlite3
import pandas as pd
import re
import json
import os
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import chromadb
from chromadb.config import Settings

# Global state
db_path = None
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
            azure_deployment="text-embedding-ada-002",  # Common embedding model name
            api_version="2024-08-01-preview",
            azure_endpoint="https://elunic-stulzgpt-openai.openai.azure.com/",
            api_key="CoFdfEEVHlFmjmf0jaEPTL2c2kp5R7FI9p1BhjYpq9rJlfNfc7vMJQQJ99BJACfhMk5XJ3w3AAABACOGYmBN",
        )
    return embeddings


def setup_chromadb():
    """Setup ChromaDB for vector storage"""
    global chroma_client, collection

    # Use in-memory ChromaDB
    chroma_client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True
    ))

    # Create or get collection
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


def load_database(file):
    """Load SQLite database and create embeddings for tables"""
    global db_path, engine, all_tables, table_schemas, collection

    try:
        if isinstance(file, dict):
            db_path = file["name"]
        else:
            db_path = file.name

        # Create SQLAlchemy engine
        engine = create_engine(f"sqlite:///{db_path}")

        # Get all table names and schemas
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_tables = [row[0] for row in cursor.fetchall()]

        # Cache schema for each table
        table_schemas = {}
        for table in all_tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            table_schemas[table] = {
                "columns": [{"name": col[1], "type": col[2]} for col in columns],
                "column_names": [col[1] for col in columns]
            }
        conn.close()

        # Setup ChromaDB
        collection = setup_chromadb()

        # Create embeddings for each table
        print(f"[EMBEDDINGS] Creating embeddings for {len(all_tables)} tables...")

        documents = []
        ids = []
        metadatas = []

        for table in all_tables:
            description = create_table_description(table, table_schemas[table]["columns"])
            documents.append(description)
            ids.append(table)
            metadatas.append({"table_name": table, "num_columns": len(table_schemas[table]["columns"])})

        # Get embeddings from Azure OpenAI
        try:
            embed_model = get_embeddings()
            vectors = embed_model.embed_documents(documents)

            # Add to ChromaDB
            collection.add(
                documents=documents,
                embeddings=vectors,
                ids=ids,
                metadatas=metadatas
            )
            embed_status = f"✅ Embeddings created for {len(all_tables)} tables"
            print(f"[EMBEDDINGS] {embed_status}")

        except Exception as e:
            embed_status = f"⚠️ Embedding failed: {str(e)[:50]}... (falling back to LLM)"
            print(f"[EMBEDDINGS] Error: {e}")

        # Build summary
        summary = f"📊 Loaded {len(all_tables)} tables\n{embed_status}\n\nTables:\n"
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


def find_relevant_tables_embedding(question: str, top_k: int = 5) -> list:
    """
    Find relevant tables using embedding similarity.
    This is FAST (~0.1-0.3 seconds)!
    """
    global collection

    if collection is None or collection.count() == 0:
        print("[SEARCH] No embeddings available, returning all tables")
        return all_tables[:5]

    try:
        # Embed the question
        embed_model = get_embeddings()
        question_embedding = embed_model.embed_query(question)

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(top_k, len(all_tables))
        )

        relevant_tables = results['ids'][0] if results['ids'] else []
        distances = results['distances'][0] if results['distances'] else []

        print(f"[SEARCH] Found tables: {list(zip(relevant_tables, distances))}")

        return relevant_tables

    except Exception as e:
        print(f"[SEARCH] Error: {e}")
        return all_tables[:5]


def find_relevant_tables_llm(question: str) -> list:
    """Fallback: Use LLM to find relevant tables"""
    global all_tables, table_schemas

    table_summary = "Available tables:\n"
    for table in all_tables:
        cols = ", ".join(table_schemas[table]["column_names"])
        table_summary += f"- {table}: {cols}\n"

    prompt = f"""Given these database tables:

{table_summary}

Question: {question}

Which tables are needed? Return ONLY a JSON array of table names.
Example: ["table1", "table2"]
"""

    response = get_llm().invoke(prompt)
    response_text = response.content if hasattr(response, 'content') else str(response)

    try:
        match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if match:
            tables = json.loads(match.group())
            return [t for t in tables if t in all_tables]
    except:
        pass

    return all_tables[:5]


def generate_sql(question: str, relevant_tables: list) -> str:
    """Generate SQL using only the relevant tables"""
    global engine

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
    global db_path
    conn = sqlite3.connect(db_path)
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
        return "❌ Load a database first!", "", pd.DataFrame()

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
        df = run_sql(clean_sql)

        total_time = time.time() - start

        info = f"""📋 Tables: {', '.join(relevant_tables)}
{method} search: {stage1_time:.3f}s
🔧 SQL generation: {stage2_time:.2f}s
⏱️ Total: {total_time:.2f}s"""

        return info, clean_sql, df

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", "", pd.DataFrame()


# ---- Gradio UI ----
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🚀 Smart NLP-to-SQL with Embeddings")
    gr.Markdown("""
    **How it works:**
    1. **On Load**: Creates vector embeddings for all table descriptions
    2. **On Query**: Finds similar tables via fast vector search (~0.1s)
    3. **Generate**: Creates SQL using only relevant tables

    ⚡ **Much faster** than LLM-based table selection!
    """)

    with gr.Row():
        # --- Column 1: Database ---
        with gr.Column(scale=1):
            db_file = gr.File(label="Upload SQLite Database", file_types=[".db", ".sqlite"])
            load_btn = gr.Button("📂 Load Database & Create Embeddings", variant="primary")
            db_status = gr.Textbox(label="Database Info", lines=8, interactive=False)

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

            with gr.Row():
                ask_btn = gr.Button("🚀 Generate & Run SQL", variant="primary", size="lg")
                use_embed = gr.Checkbox(label="Use Embeddings", value=True, info="Uncheck to use LLM search")

            stage_info = gr.Textbox(label="Performance Info", lines=4, interactive=False)
            sql_out = gr.Code(label="Generated SQL", language="sql")
            result_out = gr.Dataframe(label="Query Result", wrap=True)

    # --- Events ---
    load_btn.click(load_database, inputs=db_file, outputs=[tables, db_status])
    tables.change(show_table, inputs=tables, outputs=[schema_out, data_out])
    ask_btn.click(ask_database, inputs=[question, use_embed], outputs=[stage_info, sql_out, result_out])

demo.launch(server_name="0.0.0.0", server_port=7864)
