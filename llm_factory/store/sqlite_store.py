import os
import sqlite3
from datetime import datetime
import json

class ConversationStore:
    def __init__(self, db_path="llm_factory/store/conversation.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """DB 테이블 초기화"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # 메시지 로그 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    sender TEXT,         -- e.g., 'agent_coin', 'agent_news'
                    receiver TEXT,       -- e.g., 'orchestrator'
                    message_type TEXT,   -- e.g., 'market_status', 'risk_alert'
                    content JSON,        -- 구조화된 JSON 데이터
                    raw_content TEXT     -- 원본 텍스트 (디버깅용)
                )
            """)
            # 실행 로그 (Runs)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    trigger TEXT,
                    status TEXT
                )
            """)

    def log_message(self, sender: str, receiver: str, msg_type: str, content: dict):
        """메시지 저장"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO messages (timestamp, sender, receiver, message_type, content, raw_content)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                sender,
                receiver,
                msg_type,
                json.dumps(content, ensure_ascii=False),
                json.dumps(content, ensure_ascii=False) # 일단 raw도 json string으로
            ))
        # print(f"📝 [LOG] {sender} -> {receiver} ({msg_type}) Saved.")
        print(f"[LOG] {sender} -> {receiver} ({msg_type}) Saved.")

    def get_recent_messages(self, sender: str = None, limit: int = 10):
        """최근 메시지 조회"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM messages"
            params = []
            if sender:
                query += " WHERE sender = ?"
                params.append(sender)
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

