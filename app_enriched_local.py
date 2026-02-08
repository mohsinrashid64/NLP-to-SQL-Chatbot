"""
NLP to SQL Chatbot with Enriched Schema Descriptions
- Uses LOCAL Ollama LLM (qwen3:8b) for SQL generation
- Uses LOCAL sentence-transformers for embeddings
- 100% LOCAL - No API keys needed!
- Stores descriptions in JSON file for reuse
"""

import os
import json
import time
import gradio as gr
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
import chromadb
from sentence_transformers import SentenceTransformer

# Ollama Configuration (local LLM)
OLLAMA_MODEL = "qwen2.5:3b"
# Schema descriptions file
SCHEMA_FILE = "schema_descriptions.json"

# ChromaDB persistent storage path (separate from Azure version)
CHROMA_DB_PATH = "./chroma_db_local"

# Global variables
db = None
llm = None
chain = None
chroma_client = None
collection = None
schema_descriptions = {}

# Initialize local embedding model
print("Loading local embedding model (sentence-transformers)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Local embedding model loaded")

# Initialize Ollama LLM (local)
print(f"Connecting to Ollama ({OLLAMA_MODEL})...")
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url="http://127.0.0.1:11434",
    temperature=0,
    num_ctx=8192,  # context window
)
print("✅ Ollama LLM ready")


def setup_chromadb(recreate=False):
    """Setup ChromaDB for vector storage (persistent)"""
    global chroma_client, collection

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if recreate:
        try:
            chroma_client.delete_collection("enriched_schemas_local")
        except:
            pass

    try:
        collection = chroma_client.get_collection(
            name="enriched_schemas_local",
            metadata={"hnsw:space": "cosine"}
        )
    except:
        collection = chroma_client.create_collection(
            name="enriched_schemas_local",
            metadata={"hnsw:space": "cosine"}
        )

    return collection


def load_existing_embeddings():
    """Load existing embeddings from persistent storage"""
    global chroma_client, collection

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_collection("enriched_schemas_local")
        count = collection.count()
        if count > 0:
            return True, count
        return False, 0
    except Exception as e:
        print(f"No existing embeddings found: {e}")
        return False, 0


def get_connection_string(host, port, user, password, database):
    """Get MySQL connection string with SSL for Azure"""
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?ssl=true&ssl_verify_cert=false"


def get_table_sample_data(engine, table_name, limit=3):
    """Get sample data from a table"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM `{table_name}` LIMIT {limit}"))
            rows = result.fetchall()
            columns = result.keys()
            if rows:
                return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        print(f"Error getting sample data for {table_name}: {e}")
    return []


def get_table_schema(engine, table_name):
    """Get detailed schema info for a table"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"DESCRIBE `{table_name}`"))
            return [dict(zip(result.keys(), row)) for row in result.fetchall()]
    except Exception as e:
        print(f"Error getting schema for {table_name}: {e}")
    return []


def analyze_table_with_llm(table_name, schema_info, sample_data, all_tables):
    """Use LLM to generate rich description for a table"""

    prompt = f"""Analyze this database table and provide a rich description.

Table Name: {table_name}

Schema:
{json.dumps(schema_info, indent=2, default=str)}

Sample Data (first 3 rows):
{json.dumps(sample_data, indent=2, default=str)}

Other tables in database: {', '.join(all_tables)}

Provide a JSON response with:
{{
    "description": "2-3 sentence description of what this table stores and its purpose",
    "columns": {{
        "column_name": "description of what this column means and possible values"
    }},
    "business_terms": ["list of business/domain terms this table relates to"],
    "related_tables": ["list of other tables this might JOIN with based on column names"],
    "example_questions": ["2-3 natural language questions users might ask about this table"]
}}

Respond with ONLY valid JSON, no markdown or explanation."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        return json.loads(content)
    except Exception as e:
        print(f"Error analyzing {table_name}: {e}")
        return {
            "description": f"Table {table_name}",
            "columns": {},
            "business_terms": [],
            "related_tables": [],
            "example_questions": []
        }


def generate_schema_descriptions(host, port, user, password, database, progress=gr.Progress()):
    """Analyze all tables and generate rich descriptions"""
    global schema_descriptions

    try:
        connection_string = get_connection_string(host, port, user, password, database)
        engine = create_engine(connection_string)

        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result.fetchall()]

        if not tables:
            return "No tables found in database", None

        schema_descriptions = {"database": database, "tables": {}}
        total = len(tables)

        progress(0, desc="Starting analysis...")

        for i, table_name in enumerate(tables):
            progress((i + 1) / total, desc=f"Analyzing {table_name} ({i+1}/{total})")

            schema_info = get_table_schema(engine, table_name)
            sample_data = get_table_sample_data(engine, table_name)
            analysis = analyze_table_with_llm(table_name, schema_info, sample_data, tables)

            schema_descriptions["tables"][table_name] = {
                "schema": schema_info,
                "analysis": analysis
            }

            time.sleep(0.5)

        with open(SCHEMA_FILE, 'w') as f:
            json.dump(schema_descriptions, f, indent=2, default=str)

        return f"✅ Analyzed {total} tables and saved to {SCHEMA_FILE}", json.dumps(schema_descriptions, indent=2, default=str)[:5000]

    except Exception as e:
        return f"❌ Error: {str(e)}", None


def load_schema_descriptions():
    """Load schema descriptions from file"""
    global schema_descriptions

    if os.path.exists(SCHEMA_FILE):
        with open(SCHEMA_FILE, 'r') as f:
            schema_descriptions = json.load(f)
        return True
    return False


def create_embeddings_from_descriptions(progress=gr.Progress()):
    """Create embeddings from enriched descriptions using LOCAL model"""
    global collection, schema_descriptions

    if not schema_descriptions or "tables" not in schema_descriptions:
        return "❌ No schema descriptions loaded. Generate or load them first."

    try:
        collection = setup_chromadb(recreate=True)
        tables = schema_descriptions["tables"]
        total = len(tables)

        progress(0, desc="Creating embeddings with local model...")

        for i, (table_name, info) in enumerate(tables.items()):
            progress((i + 1) / total, desc=f"Embedding {table_name} ({i+1}/{total})")

            analysis = info.get("analysis", {})

            rich_text = f"""
Table: {table_name}
Description: {analysis.get('description', '')}
Business Terms: {', '.join(analysis.get('business_terms', []))}
Example Questions: {' | '.join(analysis.get('example_questions', []))}
Columns: {', '.join([f"{k}: {v}" for k, v in analysis.get('columns', {}).items()])}
Related Tables: {', '.join(analysis.get('related_tables', []))}
"""

            # Create embedding using LOCAL model
            embedding = embedding_model.encode(rich_text).tolist()

            collection.add(
                ids=[table_name],
                embeddings=[embedding],
                metadatas=[{
                    "table_name": table_name,
                    "description": analysis.get('description', '')[:500],
                    "business_terms": ', '.join(analysis.get('business_terms', []))
                }],
                documents=[rich_text]
            )

        return f"✅ Created embeddings for {total} tables (using local model)"

    except Exception as e:
        return f"❌ Error creating embeddings: {str(e)}"


def connect_and_setup(host, port, user, password, database, selected_tables=None):
    """Connect to database with selected tables"""
    global db, chain

    try:
        connection_string = get_connection_string(host, port, user, password, database)

        if selected_tables:
            db = SQLDatabase.from_uri(
                connection_string,
                include_tables=selected_tables,
                sample_rows_in_table_info=2
            )
        else:
            db = SQLDatabase.from_uri(connection_string, sample_rows_in_table_info=1)

        chain = create_sql_query_chain(llm, db)
        return True
    except Exception as e:
        print(f"Error connecting: {e}")
        return False


def find_relevant_tables(question: str, top_k: int = 5) -> list:
    """Find relevant tables using LOCAL embeddings"""
    global collection

    if collection is None:
        return []

    try:
        # Create embedding using LOCAL model (no API needed!)
        query_embedding = embedding_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if results and results['ids'] and results['ids'][0]:
            tables_with_scores = []
            for i, table_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance
                tables_with_scores.append((table_id, similarity))
            return tables_with_scores
    except Exception as e:
        print(f"Error finding tables: {e}")

    return []


def extract_sql(response: str) -> str:
    """Extract clean SQL from LLM response"""
    if not response:
        return ""

    sql = response.strip()

    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0]
    elif "```" in sql:
        parts = sql.split("```")
        if len(parts) >= 2:
            sql = parts[1]

    if "SQLQuery:" in sql:
        sql = sql.split("SQLQuery:")[1]

    sql = sql.strip()

    sql_upper = sql.upper()
    found_sql = False
    for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]:
        if keyword in sql_upper:
            idx = sql_upper.index(keyword)
            sql = sql[idx:]
            found_sql = True
            break

    if not found_sql:
        return ""

    invalid_phrases = [
        "unfortunately", "i cannot", "i can't", "not possible",
        "no tables", "don't have", "unable to", "please provide",
        "if you have", "with the given tables"
    ]

    sql_lower = sql.lower()
    for phrase in invalid_phrases:
        if phrase in sql_lower:
            return ""

    sql_stripped = sql.strip().upper()
    if not any(sql_stripped.startswith(kw) for kw in ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]):
        return ""

    return sql.strip()


def ask_question(question: str, host: str, port: str, user: str, password: str, database: str):
    """Main query function with enriched table selection"""
    global chain

    if not question.strip():
        return "Please enter a question", "", "", pd.DataFrame()

    try:
        start_time = time.time()

        tables_with_scores = find_relevant_tables(question, top_k=5)

        if not tables_with_scores:
            return "❌ No relevant tables found. Make sure embeddings are created.", "", "", pd.DataFrame()

        search_time = time.time() - start_time

        selected_tables = [t[0] for t in tables_with_scores]
        table_info = "\n".join([f"  - {t[0]} (similarity: {t[1]:.2f})" for t in tables_with_scores])

        enriched_context = ""
        for table_name, _ in tables_with_scores:
            if table_name in schema_descriptions.get("tables", {}):
                analysis = schema_descriptions["tables"][table_name].get("analysis", {})
                enriched_context += f"\n{table_name}: {analysis.get('description', '')}"
                if analysis.get('columns'):
                    enriched_context += f"\n  Columns: {', '.join([f'{k}={v}' for k,v in list(analysis['columns'].items())[:5]])}"

        if not connect_and_setup(host, port, user, password, database, selected_tables):
            return "❌ Failed to connect to database", "", "", pd.DataFrame()

        enhanced_question = f"""Context about relevant tables:{enriched_context}

User Question: {question}

Important:
- Do NOT add LIMIT unless the user asks for a specific number
- Use appropriate JOINs based on related tables
- For counting or reporting, use COUNT() and GROUP BY as needed"""

        sql_start = time.time()
        response = chain.invoke({"question": enhanced_question})
        sql_time = time.time() - sql_start

        sql_query = extract_sql(response)

        if not sql_query:
            llm_response = response if isinstance(response, str) else str(response)
            return f"❌ Could not generate SQL.\n\n💬 LLM Response:\n{llm_response[:500]}\n\n💡 Try rephrasing your question.", "", "", pd.DataFrame()

        exec_start = time.time()
        result = db.run(sql_query)
        exec_time = time.time() - exec_start

        # Parse results
        try:
            import ast
            import re

            clean_result = re.sub(
                r'datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+))?\)',
                lambda m: f'"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d} {int(m.group(4)):02d}:{int(m.group(5)):02d}:{int(m.group(6)):02d}"',
                result
            )
            clean_result = re.sub(r"Decimal\('([^']+)'\)", r'\1', clean_result)

            if clean_result.startswith("[") and clean_result.endswith("]"):
                data = ast.literal_eval(clean_result)
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict):
                        df = pd.DataFrame(data)
                    elif isinstance(data[0], tuple):
                        columns = None
                        sql_upper = sql_query.upper()
                        if "SELECT" in sql_upper and "FROM" in sql_upper:
                            select_part = sql_query[sql_upper.index("SELECT")+6:sql_upper.index("FROM")]
                            cols = [c.strip() for c in select_part.split(",")]
                            clean_cols = []
                            for col in cols:
                                if " AS " in col.upper():
                                    col = col.upper().split(" AS ")[-1].strip()
                                elif " as " in col:
                                    col = col.split(" as ")[-1].strip()
                                if "." in col:
                                    col = col.split(".")[-1]
                                col = col.replace("`", "").replace("'", "").replace('"', "").strip()
                                clean_cols.append(col)
                            if len(clean_cols) == len(data[0]):
                                columns = clean_cols

                        if columns:
                            df = pd.DataFrame(data, columns=columns)
                        else:
                            df = pd.DataFrame(data, columns=[f"Col_{i+1}" for i in range(len(data[0]))])
                    else:
                        df = pd.DataFrame({"Result": data})
                else:
                    df = pd.DataFrame({"Result": ["No data returned"]})
            else:
                df = pd.DataFrame({"Result": [clean_result]})
        except Exception as parse_error:
            print(f"Parse error: {parse_error}")
            df = pd.DataFrame({"Result": [result]})

        timing = f"Search: {search_time:.2f}s | SQL Gen: {sql_time:.2f}s | Exec: {exec_time:.2f}s | Total: {time.time()-start_time:.2f}s"

        return f"✅ Found tables:\n{table_info}\n\n⏱️ {timing}", sql_query, str(result)[:2000], df

    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", pd.DataFrame()


def load_existing_schema():
    """Load existing schema file if available"""
    if load_schema_descriptions():
        tables = list(schema_descriptions.get("tables", {}).keys())
        return f"✅ Loaded {len(tables)} table descriptions from {SCHEMA_FILE}", json.dumps(schema_descriptions, indent=2)[:5000]
    return "No existing schema file found. Generate new descriptions.", ""


# Gradio UI
with gr.Blocks(title="NLP to SQL - Fully Local", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 NLP to SQL - Fully Local")
    gr.Markdown(f"Uses **Ollama ({OLLAMA_MODEL})** for SQL + **sentence-transformers** for embeddings - 100% LOCAL, no API keys!")

    with gr.Tab("1️⃣ Database Connection"):
        gr.Markdown("### MySQL Connection Details")
        with gr.Row():
            host = gr.Textbox(label="Host", value="zkw-test-mysql.mysql.database.azure.com")
            port = gr.Textbox(label="Port", value="3306")
        with gr.Row():
            user = gr.Textbox(label="Username", value="mysqladmin")
            password = gr.Textbox(label="Password", value="ZkwTest2026Pass1", type="password")
        database = gr.Textbox(label="Database", value="zkwmbdb_global_elunic")

    with gr.Tab("2️⃣ Generate Schema Descriptions"):
        gr.Markdown(f"""### Analyze Database with LLM
        This will analyze each table using **Ollama ({OLLAMA_MODEL})** and generate rich descriptions.
        """)

        with gr.Row():
            generate_btn = gr.Button("🔍 Analyze Database (Uses LLM)", variant="primary")
            load_btn = gr.Button("📂 Load Existing Schema File")

        schema_status = gr.Textbox(label="Status", interactive=False)
        schema_preview = gr.Textbox(label="Schema Preview", lines=15, interactive=False)

        generate_btn.click(
            generate_schema_descriptions,
            inputs=[host, port, user, password, database],
            outputs=[schema_status, schema_preview]
        )

        load_btn.click(
            load_existing_schema,
            outputs=[schema_status, schema_preview]
        )

    with gr.Tab("3️⃣ Create Embeddings"):
        gr.Markdown("### Create Embeddings (Local Model - FREE)")
        gr.Markdown("Uses `all-MiniLM-L6-v2` - runs locally, no API needed!")

        with gr.Row():
            load_embed_btn = gr.Button("📂 Load Existing Embeddings", variant="secondary")
            embed_btn = gr.Button("🧮 Create New Embeddings", variant="primary")

        embed_status = gr.Textbox(label="Status", interactive=False)

        def check_and_load_embeddings():
            exists, count = load_existing_embeddings()
            if exists:
                load_schema_descriptions()
                return f"✅ Loaded {count} existing embeddings from {CHROMA_DB_PATH}"
            return "❌ No existing embeddings found. Create new ones."

        load_embed_btn.click(
            check_and_load_embeddings,
            outputs=[embed_status]
        )

        embed_btn.click(
            create_embeddings_from_descriptions,
            outputs=[embed_status]
        )

    with gr.Tab("4️⃣ Query Database"):
        gr.Markdown("### Ask Questions in Natural Language")

        question = gr.Textbox(
            label="Your Question",
            placeholder="e.g., Give me the NOK parts on 9. January for workstation X",
            lines=2
        )

        ask_btn = gr.Button("🚀 Ask", variant="primary")

        with gr.Row():
            status = gr.Textbox(label="Status & Matched Tables", lines=6)
            sql_output = gr.Textbox(label="Generated SQL", lines=6)

        raw_result = gr.Textbox(label="Raw Result", lines=3)
        result_table = gr.Dataframe(label="Results")

        ask_btn.click(
            ask_question,
            inputs=[question, host, port, user, password, database],
            outputs=[status, sql_output, raw_result, result_table]
        )

        gr.Markdown("### Example Questions")
        gr.Markdown("""
        - Show all NOK parts from January 2025
        - Count defective parts grouped by workstation
        - List all workstations with their production counts
        - Get failed tests for serial number XYZ
        """)

    with gr.Tab("📝 Edit Schema"):
        gr.Markdown("### Manually Edit Schema Descriptions")

        schema_editor = gr.Textbox(label="Schema JSON", lines=20)

        with gr.Row():
            load_edit_btn = gr.Button("📂 Load Schema for Editing")
            save_edit_btn = gr.Button("💾 Save Changes", variant="primary")

        edit_status = gr.Textbox(label="Status", interactive=False)

        def load_for_edit():
            if os.path.exists(SCHEMA_FILE):
                with open(SCHEMA_FILE, 'r') as f:
                    return f.read(), "Schema loaded"
            return "", "No schema file found"

        def save_edited_schema(content):
            global schema_descriptions
            try:
                schema_descriptions = json.loads(content)
                with open(SCHEMA_FILE, 'w') as f:
                    f.write(content)
                return "✅ Schema saved successfully"
            except json.JSONDecodeError as e:
                return f"❌ Invalid JSON: {e}"

        load_edit_btn.click(load_for_edit, outputs=[schema_editor, edit_status])
        save_edit_btn.click(save_edited_schema, inputs=[schema_editor], outputs=[edit_status])

if __name__ == "__main__":
    load_schema_descriptions()
    exists, count = load_existing_embeddings()
    if exists:
        print(f"✅ Loaded {count} existing embeddings from {CHROMA_DB_PATH}")
    else:
        print("ℹ️ No existing embeddings found. Create them in Tab 3.")
    demo.launch(server_port=7867, share=False)
