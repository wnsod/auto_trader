"""
진화 모니터링 및 로깅 모듈 (Phase 5)
진화 과정 추적 및 리포트 생성

기능:
1. 세그먼트 결과 로깅
2. 변이 로깅
3. 예측 피드백 로깅
4. 진화 리포트 생성
"""

import os
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

from rl_pipeline.db.connection_pool import get_strategy_db_pool

logger = logging.getLogger(__name__)

# 환경변수
ENABLE_EVOLUTION_LOGGING = os.getenv('ENABLE_EVOLUTION_LOGGING', 'true').lower() == 'true'
LOG_FILE_PATH = os.getenv('EVOLUTION_LOG_FILE', '/workspace/data_storage/evolution_logs.jsonl')


class EvolutionLogger:
    """진화 과정 모니터링 및 로깅"""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        초기화
        
        Args:
            log_file: 로그 파일 경로 (None이면 기본값 사용)
        """
        self.log_file = log_file or LOG_FILE_PATH
        self.enabled = ENABLE_EVOLUTION_LOGGING
        self.stats = defaultdict(lambda: {
            'segments': [],
            'mutations': [],
            'predictions': [],
            'performance_history': []
        })
        
        # 개선: 메모리 누수 방지 - 최대 히스토리 수 제한
        self.MAX_HISTORY_PER_STRATEGY = int(os.getenv('MAX_EVOLUTION_HISTORY', '1000'))
        
        # 로그 디렉토리 생성
        if self.enabled:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            logger.info(f"✅ Evolution Logger 초기화 완료 (로그 파일: {self.log_file})")
        else:
            logger.info("✅ Evolution Logger 초기화 완료 (로깅 비활성화)")
    
    def log_segment_result(
        self,
        strategy_id: str,
        segment: Dict[str, Any],
        metrics: Dict[str, Any]
    ):
        """
        세그먼트 결과 로깅
        
        Args:
            strategy_id: 전략 ID
            segment: 세그먼트 정보
            metrics: 성과 지표
        """
        if not self.enabled:
            return
        
        try:
            # 개선: 로그 파일 로테이션 (100MB 초과 시)
            MAX_LOG_FILE_SIZE = 100 * 1024 * 1024  # 100MB
            
            if os.path.exists(self.log_file):
                file_size = os.path.getsize(self.log_file)
                if file_size > MAX_LOG_FILE_SIZE:
                    # 백업 파일명 생성
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_file = f"{self.log_file}.{timestamp}"
                    
                    # 기존 파일 백업
                    os.rename(self.log_file, backup_file)
                    logger.info(f"✅ 로그 로테이션: {backup_file} (크기: {file_size / 1024 / 1024:.1f}MB)")
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'segment_result',
                'strategy_id': strategy_id,
                'segment': segment,
                'metrics': metrics
            }
            
            # 파일에 JSONL 형식으로 저장
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            # 메모리 통계에도 저장
            self.stats[strategy_id]['segments'].append({
                'timestamp': log_entry['timestamp'],
                'metrics': metrics
            })
            
            # 개선: 메모리 누수 방지 - 최근 N개만 유지
            if len(self.stats[strategy_id]['segments']) > self.MAX_HISTORY_PER_STRATEGY:
                self.stats[strategy_id]['segments'].pop(0)
            
            logger.debug(f"📊 세그먼트 결과 로깅: {strategy_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ 세그먼트 결과 로깅 실패: {e}")
    
    def log_mutation(
        self,
        parent_id: str,
        child_id: str,
        changes: Dict[str, Any]
    ):
        """
        변이 로깅
        
        Args:
            parent_id: 부모 전략 ID
            child_id: 자식 전략 ID
            changes: 변이 내용
        """
        if not self.enabled:
            return
        
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'mutation',
                'parent_id': parent_id,
                'child_id': child_id,
                'changes': changes
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            self.stats[child_id]['mutations'].append({
                'timestamp': log_entry['timestamp'],
                'parent_id': parent_id,
                'changes': changes
            })
            
            # 개선: 메모리 누수 방지
            if len(self.stats[child_id]['mutations']) > self.MAX_HISTORY_PER_STRATEGY:
                self.stats[child_id]['mutations'].pop(0)
            
            logger.debug(f"🧬 변이 로깅: {parent_id} → {child_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ 변이 로깅 실패: {e}")
    
    def log_prediction_feedback(
        self,
        strategy_id: str,
        errors: np.ndarray,
        weights: np.ndarray
    ):
        """
        예측 피드백 로깅
        
        Args:
            strategy_id: 전략 ID
            errors: 예측 오차 배열
            weights: 가중치 배열
        """
        if not self.enabled:
            return
        
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'prediction_feedback',
                'strategy_id': strategy_id,
                'error_stats': {
                    'mean': float(np.mean(errors)),
                    'std': float(np.std(errors)),
                    'min': float(np.min(errors)),
                    'max': float(np.max(errors))
                },
                'weight_stats': {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights))
                }
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            self.stats[strategy_id]['predictions'].append({
                'timestamp': log_entry['timestamp'],
                'error_mean': log_entry['error_stats']['mean'],
                'weight_mean': log_entry['weight_stats']['mean']
            })
            
            # 개선: 메모리 누수 방지
            if len(self.stats[strategy_id]['predictions']) > self.MAX_HISTORY_PER_STRATEGY:
                self.stats[strategy_id]['predictions'].pop(0)
            
            logger.debug(f"📈 예측 피드백 로깅: {strategy_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ 예측 피드백 로깅 실패: {e}")
    
    def generate_evolution_report(
        self,
        strategy_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        세대별 진화 리포트 생성
        
        Args:
            strategy_ids: 리포트 생성할 전략 ID 리스트 (None이면 전체)
        
        Returns:
            진화 리포트 딕셔너리
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'strategies': {}
            }
            
            target_strategies = strategy_ids if strategy_ids else list(self.stats.keys())
            
            for strategy_id in target_strategies:
                if strategy_id not in self.stats:
                    continue
                
                stats = self.stats[strategy_id]
                
                # 세그먼트 성과 통계
                segment_performances = [
                    s['metrics'].get('profit', 0.0) for s in stats['segments']
                ]
                
                # 변이 통계
                mutation_count = len(stats['mutations'])
                
                # 예측 피드백 통계
                prediction_errors = [
                    p['error_mean'] for p in stats['predictions']
                ]
                
                report['strategies'][strategy_id] = {
                    'segment_count': len(stats['segments']),
                    'avg_profit': np.mean(segment_performances) if segment_performances else 0.0,
                    'profit_std': np.std(segment_performances) if segment_performances else 0.0,
                    'mutation_count': mutation_count,
                    'avg_prediction_error': np.mean(prediction_errors) if prediction_errors else 0.0
                }
            
            logger.info(f"✅ 진화 리포트 생성 완료: {len(report['strategies'])}개 전략")
            return report
            
        except Exception as e:
            logger.error(f"❌ 진화 리포트 생성 실패: {e}")
            return {}

