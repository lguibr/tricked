import sqlite3, json

conn = sqlite3.connect('/home/lg/lab/tricked/test.db')
conn.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, name TEXT, type TEXT, status TEXT, config JSON, tags JSON, artifacts_dir TEXT, start_time DATETIME, end_time DATETIME);')
config_json = json.dumps({"training": {"training_steps": 100}, "paths": {"model_checkpoint_path": "", "metrics_file_path": "/tmp/tricked_artifacts/metrics.csv", "workspace_db_path": "/home/lg/lab/tricked/test.db"}})
conn.execute('INSERT OR REPLACE INTO runs (id, name, type, status, config, artifacts_dir) VALUES (?, ?, ?, ?, ?, ?);', 
    ('debug_123', 'debug_123', 'EXPERIMENT', 'RUNNING', config_json, '/tmp/tricked_artifacts'))
conn.commit()
