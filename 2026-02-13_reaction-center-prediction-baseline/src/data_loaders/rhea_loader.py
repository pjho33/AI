"""
Rhea 데이터베이스 로더
실제 생화학 반응 데이터 통합
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import time


@dataclass
class RheaReaction:
    """Rhea 반응 데이터"""
    rhea_id: str
    equation: str
    ec_numbers: List[str]
    substrates: List[Dict]
    products: List[Dict]
    is_transport: bool
    is_bidirectional: bool
    status: str  # approved, preliminary, etc.


class RheaDataLoader:
    """
    Rhea 데이터베이스에서 반응 데이터 로드
    
    Rhea: https://www.rhea-db.org/
    - 12,000+ 생화학 반응
    - EC 번호 매핑
    - ChEBI 구조 정보
    """
    
    def __init__(self, cache_dir: str = "data/rhea_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.rhea-db.org/rest/1.0"
    
    def fetch_reactions_by_ec(
        self,
        ec_number: str,
        limit: int = 100
    ) -> List[RheaReaction]:
        """
        EC 번호로 반응 검색
        
        Args:
            ec_number: EC 번호 (예: "1.1.1.1")
            limit: 최대 반응 수
        
        Returns:
            List[RheaReaction]: 반응 리스트
        """
        
        cache_file = self.cache_dir / f"ec_{ec_number.replace('.', '_')}.json"
        
        # 캐시 확인
        if cache_file.exists():
            print(f"캐시에서 로드: {ec_number}")
            with open(cache_file) as f:
                data = json.load(f)
                return [self._parse_reaction(r) for r in data]
        
        print(f"Rhea에서 다운로드: {ec_number}")
        
        try:
            # Rhea API 호출
            url = f"{self.base_url}/search"
            params = {
                "query": f"ec:{ec_number}",
                "limit": limit,
                "format": "json"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            reactions = data.get("results", [])
            
            # 캐시 저장
            with open(cache_file, 'w') as f:
                json.dump(reactions, f, indent=2)
            
            return [self._parse_reaction(r) for r in reactions]
            
        except Exception as e:
            print(f"오류: {e}")
            return []
    
    def fetch_polyol_oxidation_reactions(self) -> List[RheaReaction]:
        """
        폴리올 산화 반응 수집
        EC 1.1.1.x (알코올 산화효소)
        """
        
        ec_classes = [
            "1.1.1.1",   # alcohol dehydrogenase
            "1.1.1.6",   # glycerol dehydrogenase
            "1.1.1.14",  # L-iditol 2-dehydrogenase
            "1.1.1.9",   # D-xylulose reductase
            "1.1.1.21",  # aldehyde reductase
        ]
        
        all_reactions = []
        for ec in ec_classes:
            reactions = self.fetch_reactions_by_ec(ec, limit=50)
            all_reactions.extend(reactions)
            time.sleep(1)  # API 부하 방지
        
        return all_reactions
    
    def fetch_isomerization_reactions(self) -> List[RheaReaction]:
        """
        이성질화 반응 수집
        EC 5.3.1.x (당 이성질화효소)
        """
        
        ec_classes = [
            "5.3.1.5",   # xylose isomerase
            "5.3.1.9",   # glucose-6-phosphate isomerase
            "5.3.1.18",  # ribose-5-phosphate isomerase
        ]
        
        all_reactions = []
        for ec in ec_classes:
            reactions = self.fetch_reactions_by_ec(ec, limit=50)
            all_reactions.extend(reactions)
            time.sleep(1)
        
        return all_reactions
    
    def _parse_reaction(self, raw_data: Dict) -> RheaReaction:
        """원시 데이터를 RheaReaction으로 파싱"""
        
        return RheaReaction(
            rhea_id=raw_data.get("id", ""),
            equation=raw_data.get("equation", ""),
            ec_numbers=raw_data.get("ec", []),
            substrates=raw_data.get("substrates", []),
            products=raw_data.get("products", []),
            is_transport=raw_data.get("isTransport", False),
            is_bidirectional=raw_data.get("isBidirectional", False),
            status=raw_data.get("status", "unknown")
        )
    
    def create_training_dataset(
        self,
        reactions: List[RheaReaction],
        output_file: str = "data/rhea_training_data.json"
    ) -> List[Dict]:
        """
        학습용 데이터셋 생성
        
        각 반응에 대해:
        - 기질 구조
        - 반응 유형
        - 효소 정보
        - 가능성 라벨 (approved = True)
        """
        
        training_data = []
        
        for rxn in reactions:
            # Transport 반응은 제외 (화학 반응 아님)
            if rxn.is_transport:
                continue
            
            # 승인된 반응만 사용
            if rxn.status != "approved":
                continue
            
            # EC 번호가 있는 반응만
            if not rxn.ec_numbers:
                continue
            
            # 반응 유형 결정
            ec_class = rxn.ec_numbers[0].split('.')[0]
            reaction_type = {
                "1": "oxidation",
                "2": "transfer",
                "3": "hydrolysis",
                "4": "lyase",
                "5": "isomerization",
                "6": "ligase"
            }.get(ec_class, "unknown")
            
            for substrate in rxn.substrates:
                entry = {
                    "rhea_id": rxn.rhea_id,
                    "substrate": {
                        "chebi_id": substrate.get("chebiId", ""),
                        "name": substrate.get("name", ""),
                        "smiles": substrate.get("smiles", "")
                    },
                    "reaction_type": reaction_type,
                    "enzyme": {
                        "ec_numbers": rxn.ec_numbers,
                        "primary_ec": rxn.ec_numbers[0]
                    },
                    "feasibility_label": True,  # Rhea의 승인된 반응
                    "confidence": 0.95,  # 실험적으로 검증됨
                    "source": "rhea",
                    "equation": rxn.equation
                }
                
                training_data.append(entry)
        
        # 저장
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"학습 데이터 생성 완료: {len(training_data)}개 반응")
        print(f"저장 위치: {output_path}")
        
        return training_data
    
    def create_negative_examples(
        self,
        positive_reactions: List[Dict],
        n_negative: int = 100
    ) -> List[Dict]:
        """
        부정 예시 생성 (불가능한 반응)
        
        전략:
        1. 잘못된 효소-반응 조합
        2. 극한 조건
        3. 비호환 기질
        """
        
        negative_examples = []
        
        # 1. 효소-반응 불일치
        for rxn in positive_reactions[:n_negative//2]:
            # 산화 반응에 이성질화 효소
            if rxn["reaction_type"] == "oxidation":
                wrong_ec = "5.3.1.5"
            # 이성질화에 산화 효소
            elif rxn["reaction_type"] == "isomerization":
                wrong_ec = "1.1.1.1"
            else:
                continue
            
            negative = rxn.copy()
            negative["enzyme"]["primary_ec"] = wrong_ec
            negative["feasibility_label"] = False
            negative["confidence"] = 0.9
            negative["reason"] = "enzyme_mismatch"
            
            negative_examples.append(negative)
        
        # 2. 극한 조건
        for rxn in positive_reactions[n_negative//2:n_negative]:
            negative = rxn.copy()
            negative["conditions"] = {
                "pH": 2.0,  # 극한 pH
                "temperature": 80  # 효소 변성
            }
            negative["feasibility_label"] = False
            negative["confidence"] = 0.85
            negative["reason"] = "extreme_conditions"
            
            negative_examples.append(negative)
        
        return negative_examples


def main():
    """Rhea 데이터 로더 데모"""
    
    print("="*70)
    print("Rhea 데이터베이스 로더")
    print("="*70)
    
    loader = RheaDataLoader()
    
    # 1. 폴리올 산화 반응 수집
    print("\n1. 폴리올 산화 반응 수집 중...")
    oxidation_reactions = loader.fetch_polyol_oxidation_reactions()
    print(f"   수집 완료: {len(oxidation_reactions)}개 반응")
    
    # 2. 이성질화 반응 수집
    print("\n2. 이성질화 반응 수집 중...")
    isomerization_reactions = loader.fetch_isomerization_reactions()
    print(f"   수집 완료: {len(isomerization_reactions)}개 반응")
    
    # 3. 학습 데이터셋 생성
    print("\n3. 학습 데이터셋 생성 중...")
    all_reactions = oxidation_reactions + isomerization_reactions
    training_data = loader.create_training_dataset(all_reactions)
    
    # 4. 부정 예시 생성
    print("\n4. 부정 예시 생성 중...")
    negative_examples = loader.create_negative_examples(training_data, n_negative=50)
    print(f"   생성 완료: {len(negative_examples)}개 부정 예시")
    
    # 5. 통합 데이터셋
    full_dataset = training_data + negative_examples
    
    with open("data/rhea_full_dataset.json", 'w') as f:
        json.dump(full_dataset, f, indent=2)
    
    print("\n" + "="*70)
    print("데이터셋 요약")
    print("="*70)
    print(f"총 반응 수: {len(full_dataset)}")
    print(f"  긍정 예시: {len(training_data)} ({len(training_data)/len(full_dataset)*100:.1f}%)")
    print(f"  부정 예시: {len(negative_examples)} ({len(negative_examples)/len(full_dataset)*100:.1f}%)")
    
    # 반응 유형별 분포
    reaction_types = {}
    for rxn in training_data:
        rt = rxn["reaction_type"]
        reaction_types[rt] = reaction_types.get(rt, 0) + 1
    
    print("\n반응 유형별 분포:")
    for rt, count in sorted(reaction_types.items(), key=lambda x: -x[1]):
        print(f"  {rt}: {count}개")
    
    print("\n데이터 저장 완료!")
    print("  - data/rhea_training_data.json (긍정 예시)")
    print("  - data/rhea_full_dataset.json (전체)")


if __name__ == "__main__":
    main()
