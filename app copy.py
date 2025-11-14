import gradio as gr
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
from langchain.utilities import SQLDatabase
from langchain.llms import Ollama
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import ollama
from langchain.chains import create_sql_query_chain
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Initialize database and LLM (replace with your actual initialization)
engine = create_engine('sqlite:///heart_disease.db')
db = SQLDatabase(engine)
llm = Ollama(model="qwen2.5:7b")

# Initialize SQL database chain
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True)

def get_table_names():
    """Get all table names from the database"""
    return db.get_usable_table_names()

def get_table_schema(table_name):
    """Get schema for a specific table in SQLite"""
    if engine is None or not table_name:
        return pd.DataFrame()

    try:
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(Patients);"))
            rows = result.fetchall()
            if not rows:
                return pd.DataFrame()
            schema = pd.DataFrame(
                rows,
                columns=['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk']
            )
        return schema
    except Exception as e:
        print("Error fetching schema:", e)
        return pd.DataFrame()

def get_table_data(table_name, limit=100):
    """Get sample data from a table"""
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return pd.read_sql_query(query, engine)
    except:
        return pd.DataFrame()

def execute_sql_query(query):
    """Execute SQL query and return results"""
    try:
        result = pd.read_sql_query(query, engine)
        return result, None
    except Exception as e:
        return pd.DataFrame(), str(e)

def generate_sql_and_execute(natural_language_query):
    """Generate SQL from natural language and execute it"""
    try:
        # Generate SQL query
        response = db_chain(natural_language_query)
        sql_query = response['intermediate_steps'][1]
        
        # Execute the query
        result_df, error = execute_sql_query(sql_query)
        
        return sql_query, result_df, error, response['result']
    except Exception as e:
        return "", pd.DataFrame(), str(e), f"Error: {str(e)}"

def analyze_results(analysis_prompt, sql_results_df):
    """Perform analysis on the SQL results"""
    if sql_results_df.empty:
        return "No data available for analysis. Please run a successful query first."
    
    # Convert DataFrame to JSON for the LLM
    data_sample = sql_results_df.head(10).to_json(orient='records', indent=2)
    
    # Create analysis prompt
    analysis_template = f"""
    Based on the following data sample:
    {data_sample}
    
    The dataset has {len(sql_results_df)} rows and {len(sql_results_df.columns)} columns.
    Column names: {', '.join(sql_results_df.columns)}
    
    Please provide insights about: {analysis_prompt}
    
    Your response should be clear, concise, and informative.
    """
    
    try:
        # Use LLM for analysis
        analysis_response = llm(analysis_template)
        return analysis_response
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def create_visualization(sql_results_df, chart_type="table"):
    """Create visualization based on data and chart type"""
    if sql_results_df.empty:
        return None
    
    try:
        if chart_type == "table":
            # Return as table (handled by Gradio)
            return sql_results_df
        
        elif chart_type == "bar" and len(sql_results_df.columns) >= 2:
            # Bar chart - use first column as x, second as y
            numeric_cols = sql_results_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 1:
                x_col = sql_results_df.columns[0]
                y_col = numeric_cols[0]
                fig = px.bar(sql_results_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                return fig
        
        elif chart_type == "line" and len(sql_results_df.columns) >= 2:
            # Line chart
            numeric_cols = sql_results_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 1:
                x_col = sql_results_df.columns[0]
                y_col = numeric_cols[0]
                fig = px.line(sql_results_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                return fig
        
        elif chart_type == "scatter" and len(sql_results_df.columns) >= 3:
            # Scatter plot
            numeric_cols = sql_results_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                color_col = sql_results_df.columns[0] if sql_results_df.columns[0] not in numeric_cols else None
                fig = px.scatter(sql_results_df, x=x_col, y=y_col, color=color_col, 
                                title=f"{y_col} vs {x_col}")
                return fig
        
        elif chart_type == "histogram":
            # Histogram
            numeric_cols = sql_results_df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 1:
                col = numeric_cols[0]
                fig = px.histogram(sql_results_df, x=col, title=f"Distribution of {col}")
                return fig
        
        # Default to table if no suitable visualization
        return sql_results_df
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        return sql_results_df

# Create the Gradio interface
with gr.Blocks(title="SQL Database Chat Interface", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 SQL Database Chat Interface")
    gr.Markdown("Query your database using natural language and analyze the results with AI.")
    
    # Store the current SQL results for analysis
    current_results = gr.State(pd.DataFrame())
    
    

    with gr.Tab("Query Database"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Database Explorer")
                table_names = get_table_names()
                table_selector = gr.Dropdown(choices=table_names, label="Select Table", value=table_names[0] if table_names else None)
                
                with gr.Row():
                    view_schema_btn = gr.Button("View Schema")
                    view_data_btn = gr.Button("View Sample Data")
                
                schema_output = gr.Dataframe(label="Table Schema", interactive=False)
                print("SCHEMA OUTPUT MAN",schema_output)
                sample_data_output = gr.Dataframe(label="Sample Data", interactive=False)
            
            with gr.Column(scale=2):
                gr.Markdown("### Natural Language Query")
                query_input = gr.Textbox(
                    label="Enter your question in natural language",
                    placeholder="e.g., Show me patients with heart disease aged over 50",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Generate & Execute SQL", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                sql_output = gr.Code(label="Generated SQL Query", language="sql", interactive=False)
                result_output = gr.Dataframe(label="Query Results", interactive=False)
                natural_language_output = gr.Textbox(label="Natural Language Response", interactive=False)
                error_output = gr.Textbox(label="Error Messages", visible=False)
    
    with gr.Tab("Analyze Results"):
        gr.Markdown("### Analyze Query Results")
        
        with gr.Row():
            analysis_input = gr.Textbox(
                label="What would you like to know about this data?",
                placeholder="e.g., Summarize the key findings, identify trends, or compare values",
                lines=2,
                scale=4
            )
            analysis_btn = gr.Button("Analyze", variant="primary", scale=1)
        
        analysis_output = gr.Textbox(label="Analysis Results", interactive=False)
        
        gr.Markdown("### Visualize Data")
        chart_type = gr.Radio(
            choices=["table", "bar", "line", "scatter", "histogram"],
            value="table",
            label="Select Visualization Type"
        )
        visualize_btn = gr.Button("Generate Visualization")
        visualization_output = gr.Plot(label="Data Visualization")
    
    # Query tab functionality
    def update_table_info(table_name):
        schema_df = get_table_schema(table_name)
        sample_df = get_table_data(table_name)
        return schema_df, sample_df
    
    def process_query(query):
        if not query:
            return "", pd.DataFrame(), "", "Please enter a query."
        
        sql, results_df, error, nl_response = generate_sql_and_execute(query)
        
        # Show error if exists
        error_display = error if error else ""
        
        return sql, results_df, nl_response, error_display, results_df
    
    def clear_inputs():
        return "", "", pd.DataFrame(), "", "", pd.DataFrame()
    
    # Analysis tab functionality
    def perform_analysis(analysis_prompt, results_df):
        if results_df.empty:
            return "No data available for analysis. Please run a successful query first."
        return analyze_results(analysis_prompt, results_df)
    
    def generate_vis(chart_type, results_df):
        return create_visualization(results_df, chart_type)
    
    # Event handlers
    table_selector.change(
        fn=update_table_info,
        inputs=table_selector,
        outputs=[schema_output, sample_data_output]
    )
    
    view_schema_btn.click(
        fn=lambda table: get_table_schema(table),
        inputs=table_selector,
        outputs=schema_output
    )
    
    view_data_btn.click(
        fn=lambda table: get_table_data(table),
        inputs=table_selector,
        outputs=sample_data_output
    )
    
    submit_btn.click(
        fn=process_query,
        inputs=query_input,
        outputs=[sql_output, result_output, natural_language_output, error_output, current_results]
    )
    
    clear_btn.click(
        fn=clear_inputs,
        inputs=None,
        outputs=[query_input, sql_output, result_output, natural_language_output, error_output, current_results]
    )
    
    analysis_btn.click(
        fn=perform_analysis,
        inputs=[analysis_input, current_results],
        outputs=analysis_output
    )
    
    visualize_btn.click(
        fn=generate_vis,
        inputs=[chart_type, current_results],
        outputs=visualization_output
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(share=True)