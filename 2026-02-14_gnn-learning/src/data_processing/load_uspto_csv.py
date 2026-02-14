"""
USPTO 데이터 다운로드 (CSV 형식)
대체 소스 사용
"""

import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
import json


def download_uspto_csv(data_dir: str = "data"):
    """
    USPTO 데이터 CSV 다운로드
    
    여러 대체 소스 시도
    """
    
    print("="*70)
    print("USPTO 데이터 다운로드")
    print("="*70)
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 대체 소스들
    sources = [
        {
            "name": "Schneider50k",
            "url": "https://raw.githubusercontent.com/rxn4chemistry/rxnfp/master/data/schneider50k/test.csv",
            "file": "schneider50k_test.csv"
        },
        {
            "name": "Simple USPTO Sample",
            "url": "https://raw.githubusercontent.com/connorcoley/retrosim/master/data/data_processed.csv",
            "file": "uspto_sample.csv"
        }
    ]
    
    for source in sources:
        print(f"\n시도 중: {source['name']}")
        print(f"URL: {source['url']}")
        
        output_file = data_path / source['file']
        
        if output_file.exists():
            print(f"✓ 파일이 이미 존재: {output_file}")
            return str(output_file)
        
        try:
            response = requests.get(source['url'], timeout=30)
            
            if response.status_code == 200:
                with open(output_file, 'w') as f:
                    f.write(response.text)
                
                print(f"✓ 다운로드 성공!")
                print(f"  저장 위치: {output_file}")
                
                # 미리보기
                df = pd.read_csv(output_file, nrows=5)
                print(f"\n데이터 미리보기:")
                print(df.head())
                
                return str(output_file)
            else:
                print(f"✗ 실패: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"✗ 오류: {e}")
    
    # 모두 실패하면 시뮬레이션 데이터 생성
    print("\n모든 소스 실패. 시뮬레이션 데이터 생성...")
    return create_simulated_uspto_data(data_path)


def create_simulated_uspto_data(data_path: Path, n_samples: int = 1000):
    """
    시뮬레이션 USPTO 데이터 생성
    (실제 다운로드 실패 시 대안)
    """
    
    print(f"\n시뮬레이션 데이터 생성 중... ({n_samples}개)")
    
    # 간단한 반응 템플릿
    reactions = []
    
    # 1. 알코올 산화
    alcohols = ["CCO", "CCCO", "CC(C)O", "CCCCO"]
    for alcohol in alcohols:
        reactions.append({
            "reactants": alcohol,
            "products": alcohol.replace("O", "=O"),
            "reaction_type": "oxidation"
        })
    
    # 2. 에스터화
    acids = ["CC(=O)O", "CCC(=O)O"]
    for acid in acids:
        for alcohol in alcohols:
            reactions.append({
                "reactants": f"{acid}.{alcohol}",
                "products": acid.replace("O", f"O{alcohol[:-1]}"),
                "reaction_type": "esterification"
            })
    
    # 3. 환원
    carbonyls = ["CC=O", "CCC=O", "CC(=O)C"]
    for carbonyl in carbonyls:
        reactions.append({
            "reactants": carbonyl,
            "products": carbonyl.replace("=O", "O"),
            "reaction_type": "reduction"
        })
    
    # 반복해서 n_samples 채우기
    import random
    while len(reactions) < n_samples:
        reactions.append(random.choice(reactions[:20]))
    
    reactions = reactions[:n_samples]
    
    # DataFrame 생성
    df = pd.DataFrame(reactions)
    
    # 저장
    output_file = data_path / "uspto_simulated.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✓ 시뮬레이션 데이터 생성 완료")
    print(f"  저장 위치: {output_file}")
    print(f"  반응 수: {len(df)}")
    
    return str(output_file)


def parse_uspto_data(csv_file: str, max_samples: int = None):
    """
    USPTO CSV 파싱
    
    Args:
        csv_file: CSV 파일 경로
        max_samples: 최대 샘플 수 (None = 전체)
    
    Returns:
        List[Dict]: 파싱된 반응 데이터
    """
    
    print(f"\nUSPTO 데이터 파싱 중...")
    
    df = pd.read_csv(csv_file)
    
    if max_samples:
        df = df.head(max_samples)
    
    print(f"총 {len(df)}개 반응")
    
    # 컬럼 확인
    print(f"컬럼: {df.columns.tolist()}")
    
    reactions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="파싱"):
        # 컬럼 이름은 데이터셋마다 다를 수 있음
        if 'reactants' in df.columns:
            reactants = row['reactants']
            products = row.get('products', '')
        elif 'rxn' in df.columns:
            # rxn 형식: reactants>>products
            parts = row['rxn'].split('>>')
            reactants = parts[0] if len(parts) > 0 else ''
            products = parts[1] if len(parts) > 1 else ''
        else:
            # 첫 번째 컬럼을 반응으로 가정
            reactants = str(row.iloc[0])
            products = str(row.iloc[1]) if len(row) > 1 else ''
        
        reactions.append({
            'reactants': reactants,
            'products': products,
            'index': idx
        })
    
    print(f"✓ 파싱 완료: {len(reactions)}개")
    
    return reactions


def main():
    """USPTO 데이터 다운로드 및 파싱"""
    
    # 다운로드
    csv_file = download_uspto_csv()
    
    if csv_file:
        # 파싱 (일부만)
        reactions = parse_uspto_data(csv_file, max_samples=100)
        
        # 샘플 출력
        print("\n" + "="*70)
        print("샘플 반응")
        print("="*70)
        
        for i, rxn in enumerate(reactions[:5]):
            print(f"\n[반응 {i+1}]")
            print(f"  반응물: {rxn['reactants']}")
            print(f"  생성물: {rxn['products']}")
        
        # JSON 저장
        output_json = Path("data/uspto_parsed.json")
        with open(output_json, 'w') as f:
            json.dump(reactions, f, indent=2)
        
        print(f"\n✓ JSON 저장: {output_json}")
        print(f"  {len(reactions)}개 반응")


if __name__ == "__main__":
    main()
