"""
실제 Rhea API에서 데이터 다운로드
"""

import json
import requests
import time
from pathlib import Path
from typing import List, Dict


def fetch_rhea_reactions(ec_number: str, limit: int = 50) -> List[Dict]:
    """
    Rhea 공식 REST API에서 반응 검색
    
    API 문서: https://www.rhea-db.org/help/rest-api
    """
    
    print(f"Fetching reactions for EC {ec_number}...")
    
    try:
        # Rhea 공식 REST endpoint
        url = "https://www.rhea-db.org/rhea/"
        params = {
            "query": f"ec:{ec_number}",
            "columns": "rhea-id,equation,ec",
            "format": "tsv",
            "limit": limit
        }
        
        print(f"  GET {url} with params: {params}")
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            # TSV 응답 파싱
            content = response.text
            lines = content.strip().split('\n')
            
            if len(lines) < 2:
                print(f"  No reactions found")
                return []
            
            # 헤더 파싱
            header = lines[0].split('\t')
            
            # 데이터 파싱
            reactions = []
            for line in lines[1:]:
                fields = line.split('\t')
                if len(fields) >= 3:
                    reaction = {
                        "rhea_id": fields[0],
                        "equation": fields[1],
                        "ec_numbers": [fields[2]] if len(fields) > 2 else [],
                        "source": "rhea_rest_api"
                    }
                    reactions.append(reaction)
            
            print(f"  Found {len(reactions)} reactions")
            return reactions
        else:
            print(f"  Error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return []
            
    except Exception as e:
        print(f"  Exception: {e}")
        return []


def parse_reaction_to_training_format(reaction: Dict) -> Dict:
    """Rhea 반응을 학습 데이터 형식으로 변환"""
    
    # 반응식에서 기질 추출 (간단한 버전)
    equation = reaction.get("equation", "")
    
    # EC 번호로 반응 유형 결정
    ec_numbers = reaction.get("ec_numbers", [])
    if not ec_numbers:
        return None
    
    primary_ec = ec_numbers[0]
    ec_class = primary_ec.split('.')[0] if '.' in primary_ec else "1"
    
    reaction_type = {
        "1": "oxidation",
        "2": "transfer",
        "3": "hydrolysis",
        "4": "lyase",
        "5": "isomerization",
        "6": "ligase"
    }.get(ec_class, "unknown")
    
    return {
        "rhea_id": reaction["rhea_id"],
        "equation": equation,
        "reaction_type": reaction_type,
        "enzyme": {
            "ec_numbers": ec_numbers,
            "primary_ec": primary_ec
        },
        "feasibility_label": True,  # Rhea의 반응은 검증됨
        "confidence": 0.95,
        "source": "rhea"
    }


def download_polyol_oxidation_dataset():
    """폴리올 산화 반응 다운로드"""
    
    print("="*70)
    print("Rhea 데이터 다운로드: 폴리올 산화")
    print("="*70)
    
    ec_classes = [
        "1.1.1.1",   # alcohol dehydrogenase
        "1.1.1.6",   # glycerol dehydrogenase
        "1.1.1.14",  # L-iditol 2-dehydrogenase
    ]
    
    all_reactions = []
    
    for ec in ec_classes:
        reactions = fetch_rhea_reactions(ec, limit=20)
        all_reactions.extend(reactions)
        print(f"  Total so far: {len(all_reactions)}")
        time.sleep(2)
    
    # 학습 형식으로 변환
    training_data = []
    for rxn in all_reactions:
        parsed = parse_reaction_to_training_format(rxn)
        if parsed:
            training_data.append(parsed)
    
    # 저장
    output_file = Path("data/rhea_real_oxidation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n저장 완료: {output_file}")
    print(f"총 {len(training_data)}개 반응 (원본 {len(all_reactions)}개)")
    
    return training_data


def download_isomerization_dataset():
    """이성질화 반응 다운로드"""
    
    print("\n" + "="*70)
    print("Rhea 데이터 다운로드: 이성질화")
    print("="*70)
    
    ec_classes = [
        "5.3.1.5",   # xylose isomerase
        "5.3.1.9",   # glucose-6-phosphate isomerase
    ]
    
    all_reactions = []
    
    for ec in ec_classes:
        reactions = fetch_rhea_reactions(ec, limit=20)
        all_reactions.extend(reactions)
        print(f"  Total so far: {len(all_reactions)}")
        time.sleep(2)
    
    # 학습 형식으로 변환
    training_data = []
    for rxn in all_reactions:
        parsed = parse_reaction_to_training_format(rxn)
        if parsed:
            training_data.append(parsed)
    
    # 저장
    output_file = Path("data/rhea_real_isomerization.json")
    
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n저장 완료: {output_file}")
    print(f"총 {len(training_data)}개 반응 (원본 {len(all_reactions)}개)")
    
    return training_data


def main():
    """실제 Rhea 데이터 다운로드"""
    
    print("실제 Rhea API에서 데이터 다운로드 시작...")
    print("(네트워크 연결 필요, 약 2-3분 소요)\n")
    
    # 1. 산화 반응
    try:
        oxidation_reactions = download_polyol_oxidation_dataset()
    except Exception as e:
        print(f"산화 반응 다운로드 실패: {e}")
        oxidation_reactions = []
    
    # 2. 이성질화 반응
    try:
        isomerization_reactions = download_isomerization_dataset()
    except Exception as e:
        print(f"이성질화 반응 다운로드 실패: {e}")
        isomerization_reactions = []
    
    # 3. 통합
    all_reactions = oxidation_reactions + isomerization_reactions
    
    print("\n" + "="*70)
    print("다운로드 완료")
    print("="*70)
    print(f"총 {len(all_reactions)}개 반응")
    print(f"  산화: {len(oxidation_reactions)}개")
    print(f"  이성질화: {len(isomerization_reactions)}개")
    
    if len(all_reactions) > 0:
        print("\n다음 단계:")
        print("  1. 데이터 정제 및 라벨링")
        print("  2. Stage 1 모델 재학습")
        print("  3. 성능 평가")
    else:
        print("\n⚠️ 다운로드 실패")
        print("  - 네트워크 연결 확인")
        print("  - Rhea API 상태 확인")
        print("  - 시뮬레이션 데이터로 계속 진행 가능")


if __name__ == "__main__":
    main()
