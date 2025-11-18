"""
Absolute Zero 시스템 에러 표준화
한 곳에서만 raise하는 래퍼로 AI가 오류 원인을 일관되게 추적
"""

class AZError(Exception):
    """Absolute Zero 시스템 기본 예외"""
    pass

class DataLoadError(AZError):
    """데이터 로딩 실패"""
    pass

class IndicatorError(AZError):
    """지표 계산 실패"""
    pass

class DBWriteError(AZError):
    """데이터베이스 쓰기 실패"""
    pass

class DBReadError(AZError):
    """데이터베이스 읽기 실패"""
    pass

class StrategyError(AZError):
    """전략 생성/검증 실패"""
    pass

class SimulationError(AZError):
    """시뮬레이션 실행 실패"""
    pass

class AnalysisError(AZError):
    """분석 실행 실패"""
    pass

class DNAAnalysisError(AnalysisError):
    """DNA 분석 실패"""
    pass

class FractalAnalysisError(AnalysisError):
    """프랙탈 분석 실패"""
    pass

class SynergyAnalysisError(AnalysisError):
    """시너지 분석 실패"""
    pass

class PerformanceError(AZError):
    """성능 모니터링 실패"""
    pass

class ConfigError(AZError):
    """설정 오류"""
    pass

class ValidationError(AZError):
    """검증 실패"""
    pass

class CacheError(AZError):
    """캐시 오류"""
    pass
