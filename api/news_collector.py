import xml.etree.ElementTree as ET
from urllib.request import urlopen
from urllib.parse import quote # 🆕 quote import 추가
from typing import List, Optional

class NewsCollector:
    """
    실시간 금융 뉴스 수집 및 분석 전담 모듈
    """
    
    def __init__(self):
        # 검색 키워드 확장성 고려
        self.keywords = ["비트코인", "경제", "주식시장", "금리", "FOMC"]
        # 🆕 한글 검색어 URL 인코딩 적용
        query = quote('+OR+'.join(self.keywords))
        self.rss_url = f"https://news.google.com/rss/search?q={query}+when:1d&hl=ko&gl=KR&ceid=KR:ko"
        self._cached_news = []
        self._last_fetch_time = 0

    def get_latest_headlines(self, limit: int = 15) -> List[str]:
        """구글 뉴스 RSS에서 최신 헤드라인 추출"""
        try:
            # 타임아웃을 짧게 가져가서 API 전체 응답 속도 저하 방지
            # 🆕 타임아웃 2.0초 -> 5.0초로 증가 (네트워크 지연 고려)
            with urlopen(self.rss_url, timeout=5.0) as response:
                xml_data = response.read()
                root = ET.fromstring(xml_data)
                
                news = []
                # item 태그 파싱
                for item in root.findall('.//item')[:limit]:
                    title = item.find('title').text
                    
                    # 언론사 이름 제거 및 정제 (ex: "제목 - 언론사" -> "제목")
                    if ' - ' in title:
                        title = title.rsplit(' - ', 1)[0]
                        
                    news.append(title)
                
                self._cached_news = news
                return news
                
        except Exception as e:
            print(f"[NewsCollector] Error: {e}") # 디버깅용 로그
            
            # 에러 발생 시 캐시된 뉴스 반환
            if self._cached_news:
                return self._cached_news
            
            # 캐시도 없으면 Mock Data 반환 (UI 테스트용)
            return [
                "美 연준, 기준금리 결정 앞두고 시장 긴장감 고조",
                "비트코인, 주요 저항선 돌파 시도... 거래량 급증",
                "글로벌 증시, 경기 침체 우려에도 반발 매수세 유입",
                "AI 반도체 관련주, 실적 기대감에 상승세 지속",
                "달러 인덱스 강세 유지, 환율 변동성 확대 주의",
                "국제 유가, 공급 우려 완화되며 소폭 하락 마감",
                "기관 투자자, 가상자산 포트폴리오 비중 확대 움직임",
                "주요 알트코인, 비트코인 상승세에 동반 강세",
                "경제 지표 발표 앞두고 관망세 짙어진 금융 시장",
                "디파이(DeFi) 총 예치금(TVL) 꾸준한 증가세 기록",
                "시스템 트레이딩 알고리즘, 변동성 장세서 수익률 방어",
                "금일 주요 기업 실적 발표 일정 및 관전 포인트",
                "규제 당국, 가상자산 관련 새로운 가이드라인 예고",
                "블록체인 기술, 금융 넘어 물류·유통으로 도입 가속",
                "NFT 시장 거래량 회복 조짐... 바닥 다지기 진입?"
            ]

    def analyze_sentiment(self, headline: str) -> str:
        """(예정) LLM을 활용해 뉴스의 호재/악재 여부 판단"""
        # TODO: OpenAI API or Local LLM 연동
        pass

    def get_market_impact(self, headline: str) -> List[str]:
        """(예정) 뉴스가 영향을 미칠 금융 시장 식별"""
        # TODO: 키워드 매칭 또는 Semantic Search
        pass

# 싱글톤 인스턴스 (필요시)
news_collector = NewsCollector()

