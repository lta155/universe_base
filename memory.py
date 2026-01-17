import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple, Union

class SQLDB:
    """
    通用数据库操作封装。
    完全封装 SQL 语句，通过参数构建查询。
    """
    
    def __init__(self, db_path: str = "data.db", auto_persist_path: Optional[str] = None):
        """
        Args:
            db_path: 数据库路径。支持 ":memory:"。
            auto_persist_path: 仅在 db_path=":memory:" 时有效。
                             如果提供，初始化时会自动从此文件加载数据，
                             关闭时会自动保存回此文件。
        """
        self.db_path = db_path
        self.auto_persist_path = auto_persist_path
        
        # 如果是文件模式，确保目录存在
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.row_factory = self._dict_factory
        
        # [Auto-Persist] 启动时加载
        if db_path == ":memory:" and auto_persist_path and os.path.exists(auto_persist_path):
             self._load_from_backup(auto_persist_path)

    def close(self):
        if self.conn:
            # [Auto-Persist] 关闭前保存
            if self.db_path == ":memory:" and self.auto_persist_path:
                self.backup(self.auto_persist_path)
            self.conn.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def _dict_factory(self, cursor, row):
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}
    
    # --- 备份/恢复 Helpers ---

    def backup(self, target_path: str):
        """备份数据库到指定文件"""
        if target_path == ":memory:":
             raise ValueError("Backup target cannot be memory.")
             
        target_dir = os.path.dirname(os.path.abspath(target_path))
        if target_dir: os.makedirs(target_dir, exist_ok=True)

        bck = sqlite3.connect(target_path)
        with self.conn:
            self.conn.backup(bck)
        bck.close()

    def _load_from_backup(self, source_path: str):
        """从文件加载到当前内存库"""
        source = sqlite3.connect(source_path)
        with source:
            source.backup(self.conn)
        source.close()

    def create_table(self, table_name: str, columns: Dict[str, str]):
        """
        创建表。
        Args:
            columns: 字典格式列定义，如 {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}
        """
        cols_def = [f"{k} {v}" for k, v in columns.items()]
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols_def)})"
        with self.conn:
            self.conn.execute(sql)

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """插入单条数据"""
        keys = list(data.keys())
        placeholders = ', '.join(['?'] * len(keys))
        sql = f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({placeholders})"
        with self.conn:
            return self.conn.execute(sql, tuple(data.values())).lastrowid

    def bulk_insert(self, table: str, data_list: List[Dict[str, Any]]) -> int:
        """批量插入数据"""
        if not data_list: return 0
        keys = list(data_list[0].keys())
        placeholders = ', '.join(['?'] * len(keys))
        sql = f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({placeholders})"
        values = [tuple(d[k] for k in keys) for d in data_list]
        with self.conn:
            return self.conn.executemany(sql, values).rowcount

    def select(
        self, 
        table: str, 
        columns: List[str] = None, 
        where: Dict[str, Any] = None, 
        order_by: str = None, 
        limit: int = None,
        offset: int = None
    ) -> List[Dict[str, Any]]:
        """
        查询数据
        Args:
            where: {"name": "Alice", "age": [20, 30]} -> WHERE name='Alice' AND age IN (20, 30)
        """
        # 1. Columns
        cols_str = "*" if not columns else ", ".join(columns)
        
        # 2. Where
        where_clause, args = self._build_where(where)
        
        # 3. Order / Limit
        order_clause = f"ORDER BY {order_by}" if order_by else ""
        limit_clause = f"LIMIT {limit}" if limit is not None else ""
        offset_clause = f"OFFSET {offset}" if offset is not None else ""
        
        sql = f"SELECT {cols_str} FROM {table} {where_clause} {order_clause} {limit_clause} {offset_clause}"
        
        cursor = self.conn.execute(sql, tuple(args))
        return cursor.fetchall()

    def select_one(self, table: str, where: Dict[str, Any] = None, **kwargs) -> Optional[Dict[str, Any]]:
        """查询单条数据"""
        res = self.select(table, where=where, limit=1, **kwargs)
        return res[0] if res else None

    def update(self, table: str, data: Dict[str, Any], where: Dict[str, Any]) -> int:
        """更新数据"""
        if not data: return 0
        
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        where_clause, where_args = self._build_where(where)
        
        if not where_clause:
            raise ValueError("Update requires a WHERE clause to prevent full table update.")
            
        sql = f"UPDATE {table} SET {set_clause} {where_clause}"
        args = tuple(data.values()) + tuple(where_args)
        
        with self.conn:
            return self.conn.execute(sql, args).rowcount

    def delete(self, table: str, where: Dict[str, Any]) -> int:
        """删除数据"""
        where_clause, args = self._build_where(where)
        
        if not where_clause:
            raise ValueError("Delete requires a WHERE clause to prevent full table wipe.")
            
        sql = f"DELETE FROM {table} {where_clause}"
        with self.conn:
            return self.conn.execute(sql, tuple(args)).rowcount

    def execute_raw(self, sql: str, params: Tuple = ()) -> sqlite3.Cursor:
        """执行原生 SQL"""
        with self.conn:
            return self.conn.execute(sql, params)

    def _build_where(self, where: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """构建 WHERE 子句，支持 =, IN, IS NULL"""
        if not where:
            return "", []
        
        clauses = []
        args = []
        for k, v in where.items():
            if v is None:
                clauses.append(f"{k} IS NULL")
            elif isinstance(v, (list, tuple)):
                if not v: continue 
                placeholders = ", ".join(["?"] * len(v))
                clauses.append(f"{k} IN ({placeholders})")
                args.extend(v)
            else:
                clauses.append(f"{k} = ?")
                args.append(v)
                
        return "WHERE " + " AND ".join(clauses), args

if __name__ == "__main__":
    with SQLDB(":memory:", auto_persist_path="./cache.db") as db:
        # 1. 建表
        db.create_table("users", {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT", 
            "name": "TEXT", 
            "role": "TEXT"
        })
        
        # 2. 插入 (这里其实是在 RAM 中操作，极快)
        db.insert("users", {"name": "Alice", "role": "admin"})
        db.insert("users", {"name": "Bob", "role": "user"})
        
        print("Users in memory:", db.select("users"))
        
    # 退出 Context Manager 时，自动保存到了 cache.db
    print("\n--- Session Closed (Auto Saved) ---")
    
    # 验证持久化：重启一个新的 DB 读取文件
    print("Verifying persistence...")
    with SQLDB("./cache.db") as file_db:
        print("Users from file:", file_db.select("users"))
