#!/usr/bin/env python3
"""
전체 코드베이스에서 스키마 불일치 찾기
"""

import re
import os
from pathlib import Path
from collections import defaultdict

def find_create_table_statements(root_dir):
    """모든 CREATE TABLE 문 찾기"""
    create_tables = defaultdict(list)

    for py_file in Path(root_dir).rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')

            # CREATE TABLE 패턴
            pattern = r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\)'
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)

            for match in matches:
                table_name = match.group(1)
                table_def = match.group(2)
                create_tables[table_name].append({
                    'file': str(py_file),
                    'definition': table_def[:200]  # 처음 200자만
                })
        except Exception as e:
            pass

    return create_tables

def find_insert_statements(root_dir):
    """모든 INSERT 문 찾기"""
    inserts = defaultdict(list)

    for py_file in Path(root_dir).rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')

            # INSERT INTO 패턴
            pattern = r'INSERT\s+INTO\s+(\w+)\s*\((.*?)\)'
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)

            for match in matches:
                table_name = match.group(1)
                columns = match.group(2)
                inserts[table_name].append({
                    'file': str(py_file),
                    'columns': columns.strip()[:200]
                })
        except Exception as e:
            pass

    return inserts

def find_select_statements(root_dir):
    """중요한 SELECT 문 찾기"""
    selects = defaultdict(list)

    for py_file in Path(root_dir).rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')

            # SELECT ... FROM table 패턴
            pattern = r'SELECT\s+(.*?)\s+FROM\s+(\w+)'
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)

            for match in matches:
                columns = match.group(1).strip()
                table_name = match.group(2)

                # 너무 긴 SELECT는 제외
                if len(columns) < 300:
                    selects[table_name].append({
                        'file': str(py_file),
                        'columns': columns[:200]
                    })
        except Exception as e:
            pass

    return selects

def main():
    root_dir = "/workspace/rl_pipeline"
    if not Path(root_dir).exists():
        root_dir = "./rl_pipeline"

    print("🔍 스키마 불일치 검사 시작...\n")

    # CREATE TABLE 찾기
    print("=" * 80)
    print("📋 1. CREATE TABLE 문 수집")
    print("=" * 80)
    create_tables = find_create_table_statements(root_dir)

    for table_name, definitions in sorted(create_tables.items()):
        print(f"\n🗂️  테이블: {table_name}")
        if len(definitions) > 1:
            print(f"   ⚠️  경고: {len(definitions)}개의 CREATE TABLE 정의 발견!")
            for i, defn in enumerate(definitions, 1):
                file_path = defn['file'].replace('\\', '/')
                short_path = '/'.join(file_path.split('/')[-3:])
                print(f"   {i}. {short_path}")
                print(f"      {defn['definition'][:100]}...")
        else:
            file_path = definitions[0]['file'].replace('\\', '/')
            short_path = '/'.join(file_path.split('/')[-3:])
            print(f"   ✅ 1개 정의: {short_path}")

    # INSERT INTO 찾기
    print("\n" + "=" * 80)
    print("📥 2. INSERT INTO 문 수집")
    print("=" * 80)
    inserts = find_insert_statements(root_dir)

    for table_name in sorted(create_tables.keys()):
        if table_name in inserts:
            insert_list = inserts[table_name]
            print(f"\n🗂️  테이블: {table_name}")
            print(f"   📊 {len(insert_list)}개의 INSERT 문 발견")

            # 서로 다른 컬럼 조합 찾기
            unique_columns = set()
            for ins in insert_list:
                cols = ins['columns'].replace('\n', ' ').replace('\r', '')
                unique_columns.add(cols)

            if len(unique_columns) > 1:
                print(f"   ⚠️  경고: {len(unique_columns)}개의 서로 다른 컬럼 조합 발견!")
                for i, cols in enumerate(sorted(unique_columns), 1):
                    print(f"   {i}. {cols[:100]}")

    # SELECT 찾기
    print("\n" + "=" * 80)
    print("📤 3. SELECT 문 수집 (주요 테이블만)")
    print("=" * 80)
    selects = find_select_statements(root_dir)

    important_tables = [
        'integrated_analysis_results', 'coin_strategies', 'rl_episodes',
        'regime_routing_results', 'paper_trading_sessions', 'signals',
        'realtime_learning_feedback'
    ]

    for table_name in important_tables:
        if table_name in selects:
            select_list = selects[table_name]
            print(f"\n🗂️  테이블: {table_name}")
            print(f"   📊 {len(select_list)}개의 SELECT 문 발견")

            # final_signal_score 같은 특정 컬럼 사용 확인
            uses_final_signal_score = any('final_signal_score' in s['columns'] for s in select_list)
            uses_signal_score = any('signal_score' in s['columns'] and 'final_signal_score' not in s['columns'] for s in select_list)

            if uses_final_signal_score and uses_signal_score:
                print(f"   ⚠️  경고: 'signal_score'와 'final_signal_score' 모두 사용됨!")

    print("\n" + "=" * 80)
    print("✅ 스키마 검사 완료")
    print("=" * 80)

if __name__ == "__main__":
    main()
