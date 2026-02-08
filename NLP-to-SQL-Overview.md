# NLP-to-SQL System Overview

## How the System Works

### Step 1: Schema Analysis (one-time)
GPT-4 analyzes each of the 142 tables and generates rich descriptions including:
- What the table stores
- What each column means
- Business terms (e.g., "NOK", "inspection", "workstation")
- Related tables for JOINs

This is saved to a JSON file and only needs to run once.

### Step 2: Embedding Creation (one-time)
Each table description is converted into a vector embedding (a numerical representation that captures meaning). These are stored locally and persist across restarts.

### Step 3: Query Processing (each question)
1. User asks: *"Show NOK parts for workstation X"*
2. The question is converted to an embedding
3. We search for the most similar table embeddings (~0.1 sec) → finds `_tblpartinspentry`, `tbltestunit`, etc.
4. Only those 3-5 relevant tables are loaded (not all 142)
5. GPT-4 generates SQL using only the selected tables (~2-3 sec)
6. Query executes and results display

## Why It's Fast

| Before | After |
|--------|-------|
| Loading all 142 table schemas | Loading only 3-5 relevant tables |
| ~85 seconds per query | ~2-4 seconds per query |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE-TIME SETUP                           │
├─────────────────────────────────────────────────────────────┤
│  Database (142 tables)                                      │
│         ↓                                                   │
│  GPT-4 analyzes each table → Rich descriptions              │
│         ↓                                                   │
│  Descriptions → Vector Embeddings → Stored locally          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    EACH QUERY                               │
├─────────────────────────────────────────────────────────────┤
│  User Question: "Show NOK parts for workstation X"          │
│         ↓                                                   │
│  Embedding Search → Finds relevant tables (0.1 sec)         │
│         ↓                                                   │
│  GPT-4 generates SQL using only selected tables (2-3 sec)   │
│         ↓                                                   │
│  Execute SQL → Display Results                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Database Info

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `_tblpartinspentry` | Part inspections | SerialNr, TestUnitID, PartState (0=NOK, 1=OK) |
| `tbltestunit` | Workstations | TestUnitID, TestUnitName |
| `_tblbuscomentry` | Process logs | ProcessState ("iO"=OK), ProcessDuration |

## PartState Values
- `0` = Failed / NOK
- `1` = Passed / OK
