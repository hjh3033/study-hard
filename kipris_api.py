import requests
import xmltodict
from typing import List, Dict, Any
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# KIPRIS API 키 (REST_API.py와 동일)
ACCESS_KEY = "KdGgKnmG6xVLE3SyIVCU5fJhf=/bx7cObNmZxCq7ub8="

def search_patents_by_invention_title(invention_title, access_key, docs_start=1, docs_count=10):
    """
    발명의 명칭으로 특허를 검색하는 함수 (REST_API.py에서 그대로 가져옴)
    
    Args:
        invention_title (str): 검색할 발명의 명칭
        access_key (str): KIPRIS API 접근 키
        docs_start (int): 검색 시작 위치 (기본값: 1)
        docs_count (int): 검색할 문서 수 (기본값: 10, 최대 500)
    
    Returns:
        dict: 검색 결과
    """
    
    # API URL 구성
    base_url = "http://plus.kipris.or.kr/openapi/rest/patUtiModInfoSearchSevice/itemTLSearchInfo"
    
    # 파라미터 설정
    params = {
        'inventionTitle': invention_title,
        'docsStart': docs_start,
        'docsCount': docs_count,
        'accessKey': access_key
    }
    
    try:
        # API 호출
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # HTTP 에러 체크
        
        logger.info(f"API 호출 URL: {response.url}")
        logger.info(f"응답 상태 코드: {response.status_code}")
        
        # XML을 딕셔너리로 변환
        content = response.content
        dict_data = xmltodict.parse(content)
        
        return dict_data
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API 호출 중 오류 발생: {e}")
        return None
    except Exception as e:
        logger.error(f"데이터 처리 중 오류 발생: {e}")
        return None

def extract_patent_info(search_result):
    """
    검색 결과에서 특허 정보를 추출하는 함수 (REST_API.py에서 그대로 가져옴)
    
    Args:
        search_result (dict): API 검색 결과
    
    Returns:
        list: 특허 정보 리스트
    """
    
    try:
        response_data = search_result['response']
        
        # 에러 체크
        if 'error' in response_data:
            logger.error(f"API 에러: {response_data['error']}")
            return []
        
        body = response_data.get('body', {})
        
        # 검색 결과가 없는 경우
        if not body or 'items' not in body:
            logger.warning("검색 결과가 없습니다.")
            return []
        
        items = body['items']
        
        # 실제 특허 데이터는 PatentUtilityInfo 키 안에 있음
        if 'PatentUtilityInfo' not in items:
            logger.warning("특허 정보를 찾을 수 없습니다.")
            return []
        
        patent_data_list = items['PatentUtilityInfo']
        
        # 단일 결과인 경우 리스트로 변환
        if not isinstance(patent_data_list, list):
            patent_data_list = [patent_data_list]
        
        # 총 검색 결과 수 출력
        total_count = items.get('TotalSearchCount', '0')
        logger.info(f"전체 검색 결과: {total_count}건")
        
        patent_list = []
        
        for patent_info in patent_data_list:
            # 필요한 정보 추출 (REST_API.py와 완전히 동일)
            patent_data = {
                '출원인': patent_info.get('Applicant', 'N/A'),
                '발명명칭': patent_info.get('InventionName', 'N/A'),
                '출원번호': patent_info.get('ApplicationNumber', 'N/A'),
                '출원일자': patent_info.get('ApplicationDate', 'N/A'),
                '공개일자': patent_info.get('OpeningDate', 'N/A'),
                '공개번호': patent_info.get('OpeningNumber', 'N/A'),
                '공고번호': patent_info.get('PublicNumber', 'N/A'),
                '공고일자': patent_info.get('PublicDate', 'N/A'),
                '등록번호': patent_info.get('RegistrationNumber', 'N/A'),
                '등록일자': patent_info.get('RegistrationDate', 'N/A'),
                '등록상태': patent_info.get('RegistrationStatus', 'N/A'),
                'IPC분류': patent_info.get('InternationalpatentclassificationNumber', 'N/A'),
                '도면경로': patent_info.get('DrawingPath', 'N/A'),
                '썸네일경로': patent_info.get('ThumbnailPath', 'N/A'),
                '초록': patent_info.get('Abstract', 'N/A')
            }
            
            patent_list.append(patent_data)
        
        return patent_list
        
    except KeyError as e:
        logger.error(f"데이터 구조 오류: {e}")
        return []
    except Exception as e:
        logger.error(f"정보 추출 중 오류 발생: {e}")
        return []

def search_and_extract_patents(query: str, num_patents: int = 5) -> List[Dict[str, Any]]:
    """
    통합 함수: 검색부터 추출까지 한번에 처리
    
    Args:
        query (str): 검색할 발명의 명칭.
        num_patents (int): 검색할 문서 수.

    Returns:
        List[Dict[str, Any]]: 추출된 특허 정보 딕셔너리의 리스트.
    """
    
    try:
        logger.info(f"KIPRIS API 검색 시작: '{query}' ({num_patents}건)")
        
        # 1. 특허 검색 (REST_API.py 함수 그대로 사용)
        search_result = search_patents_by_invention_title(query, ACCESS_KEY, docs_count=num_patents)
        
        if search_result is None:
            logger.error("특허 검색 실패")
            return []
        
        # 2. 특허 정보 추출 (REST_API.py 함수 그대로 사용)
        patent_list = extract_patent_info(search_result)
        
        if patent_list:
            logger.info(f"성공적으로 {len(patent_list)}건의 특허 정보를 추출했습니다.")
            # 첫 번째 특허 정보 로깅으로 확인
            logger.info(f"첫 번째 특허 예시: {patent_list[0]['발명명칭']} - {patent_list[0]['출원인']}")
        else:
            logger.warning("추출된 특허 정보가 없습니다.")
        
        return patent_list
        
    except Exception as e:
        logger.error(f"search_and_extract_patents 오류: {e}")
        return [{"error": f"An error occurred: {e}"}]

def test_kipris_api():
    """KIPRIS API 테스트 함수"""
    test_query = "타이밍벨트"
    logger.info(f"테스트 검색어: {test_query}")
    
    results = search_and_extract_patents(test_query, 3)
    
    print(f"\n=== KIPRIS API 테스트 결과 ===")
    print(f"검색어: {test_query}")
    print(f"결과 수: {len(results)}")
    print("-" * 60)
    
    for i, patent in enumerate(results, 1):
        if 'error' not in patent:
            print(f"{i}. 발명명칭: {patent['발명명칭']}")
            print(f"   출원인: {patent['출원인']}")
            print(f"   출원번호: {patent['출원번호']} (출원일: {patent['출원일자']})")
            print(f"   등록번호: {patent['등록번호']} (등록일: {patent['등록일자']})")
            print(f"   등록상태: {patent['등록상태']}")
            abstract = patent['초록']
            if len(abstract) > 100:
                abstract = abstract[:100] + "..."
            print(f"   초록: {abstract}")
            print("-" * 60)
        else:
            print(f"오류: {patent}")

if __name__ == "__main__":
    # 로깅 레벨 설정
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # API 테스트 실행
    test_kipris_api()