"""
세션 관리 시스템
- 세션 생성/조회/비교/정리
- 세션 인덱스 관리
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any


class SessionManager:
    """디버그 세션 관리자"""

    def __init__(self, base_dir: Path = None):
        """
        Args:
            base_dir: 디버그 로그 베이스 디렉토리
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / "debug_logs"

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_dir / "sessions.json"

    def create_session(self, coins: List[str], intervals: List[str], config: Dict[str, Any] = None) -> str:
        """
        새 세션 생성

        Args:
            coins: 코인 리스트
            intervals: 인터벌 리스트
            config: 설정 정보

        Returns:
            session_id
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 코인과 인터벌 정보를 세션 ID에 포함
        coins_str = "_".join(coins[:3])  # 최대 3개까지만
        intervals_str = f"{len(intervals)}intervals"
        full_session_id = f"{session_id}_{coins_str}_{intervals_str}"

        session_dir = self.base_dir / full_session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # 세션 정보 저장
        session_info = {
            "session_id": full_session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "coins": coins,
            "intervals": intervals,
            "config": config or {},
            "status": "running",
            "summary": {}
        }

        info_file = session_dir / "session_info.json"
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)

        # 인덱스 업데이트
        self._add_to_index(session_info)

        return full_session_id

    def end_session(self, session_id: str, summary: Dict[str, Any] = None):
        """
        세션 종료

        Args:
            session_id: 세션 ID
            summary: 세션 요약 정보
        """
        session_dir = self.base_dir / session_id
        info_file = session_dir / "session_info.json"

        if not info_file.exists():
            print(f"⚠️ 세션 정보 파일이 없습니다: {session_id}")
            return

        # 세션 정보 업데이트
        with open(info_file, "r", encoding="utf-8") as f:
            session_info = json.load(f)

        session_info["end_time"] = datetime.now().isoformat()
        session_info["status"] = "completed"
        session_info["summary"] = summary or {}

        # 이슈 자동 감지
        issues = self._detect_issues(session_dir)
        session_info["issues_found"] = issues

        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(session_info, f, indent=2, ensure_ascii=False)

        # 인덱스 업데이트
        self._update_index(session_info)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 정보 조회

        Args:
            session_id: 세션 ID

        Returns:
            세션 정보 dict
        """
        session_dir = self.base_dir / session_id
        info_file = session_dir / "session_info.json"

        if not info_file.exists():
            return None

        with open(info_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_index(self) -> Dict[str, Any]:
        """인덱스 파일 안전하게 로드"""
        if not self.index_file.exists():
            return {"sessions": []}

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ 세션 인덱스 파일 손상 감지 ({e}). 초기화합니다.")
            # 손상된 파일 백업
            backup_path = self.index_file.with_suffix(f".bak.{datetime.now().strftime('%Y%m%d%H%M%S')}")
            try:
                shutil.copy(self.index_file, backup_path)
                print(f"   └─ 백업 완료: {backup_path.name}")
            except:
                pass
            return {"sessions": []}
        except Exception as e:
            print(f"⚠️ 세션 인덱스 로드 중 오류: {e}")
            return {"sessions": []}

    def list_sessions(self, limit: int = None, status: str = None) -> List[Dict[str, Any]]:
        """
        세션 목록 조회

        Args:
            limit: 최대 반환 개수
            status: 필터링할 상태 (running, completed)

        Returns:
            세션 정보 리스트 (최신순)
        """
        index_data = self._load_index()
        sessions = index_data.get("sessions", [])

        # 상태 필터링
        if status:
            sessions = [s for s in sessions if s.get("status") == status]

        # 최신순 정렬
        sessions.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        # limit 적용
        if limit:
            sessions = sessions[:limit]

        return sessions

    def get_latest_session(self) -> Optional[str]:
        """
        최신 세션 ID 반환

        Returns:
            최신 세션 ID
        """
        index_data = self._load_index()
        return index_data.get("latest")

    def _add_to_index(self, session_info: Dict[str, Any]):
        """세션을 인덱스에 추가"""
        index_data = self._load_index()

        # 중복 제거
        index_data["sessions"] = [
            s for s in index_data.get("sessions", [])
            if s["session_id"] != session_info["session_id"]
        ]

        # 새 세션 추가
        index_data["sessions"].append({
            "session_id": session_info["session_id"],
            "start_time": session_info["start_time"],
            "end_time": session_info.get("end_time"),
            "coins": session_info["coins"],
            "intervals": session_info["intervals"],
            "status": session_info["status"]
        })

        # latest 업데이트
        index_data["latest"] = session_info["session_id"]

        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    def _update_index(self, session_info: Dict[str, Any]):
        """세션 인덱스 업데이트"""
        index_data = self._load_index()

        if "sessions" not in index_data:
            index_data["sessions"] = []

        # 해당 세션 찾아서 업데이트
        updated = False
        for i, s in enumerate(index_data["sessions"]):
            if s["session_id"] == session_info["session_id"]:
                index_data["sessions"][i] = {
                    "session_id": session_info["session_id"],
                    "start_time": session_info["start_time"],
                    "end_time": session_info.get("end_time"),
                    "coins": session_info["coins"],
                    "intervals": session_info["intervals"],
                    "status": session_info["status"],
                    "issues_found": session_info.get("issues_found", [])
                }
                updated = True
                break
        
        if not updated:
             # 없으면 추가
            index_data["sessions"].append({
                "session_id": session_info["session_id"],
                "start_time": session_info["start_time"],
                "end_time": session_info.get("end_time"),
                "coins": session_info["coins"],
                "intervals": session_info["intervals"],
                "status": session_info["status"],
                "issues_found": session_info.get("issues_found", [])
            })

        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    def _rebuild_index(self):
        """세션 인덱스 재구성 (실제 존재하는 세션만)"""
        sessions = []

        for session_dir in self.base_dir.iterdir():
            if not session_dir.is_dir() or session_dir.name == "latest":
                continue

            info_file = session_dir / "session_info.json"
            if not info_file.exists():
                continue

            with open(info_file, "r", encoding="utf-8") as f:
                session_info = json.load(f)

            sessions.append({
                "session_id": session_info["session_id"],
                "start_time": session_info["start_time"],
                "end_time": session_info.get("end_time"),
                "coins": session_info["coins"],
                "intervals": session_info["intervals"],
                "status": session_info["status"],
                "issues_found": session_info.get("issues_found", [])
            })

        # 최신순 정렬
        sessions.sort(key=lambda x: x["start_time"], reverse=True)

        index_data = {
            "sessions": sessions,
            "latest": sessions[0]["session_id"] if sessions else None
        }

        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    def _detect_issues(self, session_dir: Path) -> List[str]:
        """세션 로그를 분석하여 이슈 자동 감지"""
        issues = []

        # training 로그 분석
        training_log = session_dir / "training.jsonl"
        if training_log.exists():
            action_diversity_warnings = 0
            early_stopped = False

            with open(training_log, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)

                        # 액션 다양성 문제
                        if entry.get("type") == "action_distribution":
                            diversity = entry.get("diversity_score", 1.0)
                            if diversity < 0.3:
                                action_diversity_warnings += 1

                        # 조기 종료
                        if "early_stopped" in entry and entry["early_stopped"]:
                            early_stopped = True

                    except:
                        pass

            if action_diversity_warnings > 10:
                issues.append("action_diversity")
            if early_stopped:
                issues.append("early_stopping")

        # validation 로그 분석
        validation_log = session_dir / "validation.jsonl"
        if validation_log.exists():
            with open(validation_log, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)

                        # 과적합
                        if entry.get("overfitting"):
                            issues.append("overfitting")
                            break

                    except:
                        pass

        return list(set(issues))  # 중복 제거

    def _diff_dicts(self, dict1: Dict, dict2: Dict) -> Dict[str, Any]:
        """두 딕셔너리 차이 계산"""
        diff = {
            "added": {},
            "removed": {},
            "changed": {}
        }

        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in all_keys:
            if key not in dict1:
                diff["added"][key] = dict2[key]
            elif key not in dict2:
                diff["removed"][key] = dict1[key]
            elif dict1[key] != dict2[key]:
                diff["changed"][key] = {
                    "old": dict1[key],
                    "new": dict2[key]
                }

        return diff
