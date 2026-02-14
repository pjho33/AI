"""
USPTO-50K 데이터셋 다운로드
50,000개 화학 반응 데이터
"""

import urllib.request
import zipfile
import os
from pathlib import Path
import json


def download_uspto_50k(data_dir: str = "data/USPTO_50K"):
    """
    USPTO-50K 데이터셋 다운로드 및 압축 해제
    
    데이터셋 정보:
    - 50,000개 화학 반응
    - 미국 특허청 데이터 (1976-2016)
    - SMILES 형식
    - Train/Valid/Test 분할
    
    Args:
        data_dir: 저장 디렉토리
    """
    
    print("="*70)
    print("USPTO-50K 데이터셋 다운로드")
    print("="*70)
    
    # 디렉토리 생성
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 다운로드 URL (대체 소스)
    # 원본이 404이므로 figshare에서 다운로드
    url = "https://figshare.com/ndownloader/files/12084729"  # USPTO-50K
    zip_path = data_path / "USPTO_50K.zip"
    
    # 이미 다운로드되어 있는지 확인
    if zip_path.exists():
        print(f"✓ 파일이 이미 존재: {zip_path}")
    else:
        print(f"\n다운로드 중: {url}")
        print(f"저장 위치: {zip_path}")
        
        try:
            # 다운로드 (진행률 표시)
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\r진행률: {percent}%", end='')
            
            urllib.request.urlretrieve(url, zip_path, reporthook)
            print("\n✓ 다운로드 완료!")
            
        except Exception as e:
            print(f"\n✗ 다운로드 실패: {e}")
            return False
    
    # 압축 해제
    print(f"\n압축 해제 중...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("✓ 압축 해제 완료!")
        
    except Exception as e:
        print(f"✗ 압축 해제 실패: {e}")
        return False
    
    # 파일 확인
    print("\n" + "="*70)
    print("다운로드된 파일")
    print("="*70)
    
    files = list(data_path.glob("**/*"))
    for f in sorted(files):
        if f.is_file():
            size = f.stat().st_size / 1024 / 1024  # MB
            print(f"  {f.relative_to(data_path)}: {size:.2f} MB")
    
    # 데이터 미리보기
    print("\n" + "="*70)
    print("데이터 미리보기")
    print("="*70)
    
    # src-train.txt 파일 찾기
    train_files = list(data_path.glob("**/src-train.txt"))
    
    if train_files:
        train_file = train_files[0]
        print(f"\n파일: {train_file.name}")
        print("-"*70)
        
        with open(train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"{i+1}. {line.strip()}")
        
        # 총 개수
        with open(train_file, 'r') as f:
            total = sum(1 for _ in f)
        print(f"\n총 {total:,}개 반응 (train)")
    
    print("\n" + "="*70)
    print("다운로드 완료!")
    print("="*70)
    print(f"\n데이터 위치: {data_path.absolute()}")
    print("\n다음 단계:")
    print("  1. 데이터 전처리 (SMILES 정규화)")
    print("  2. 분자 그래프 변환")
    print("  3. GNN 모델 구축")
    
    return True


def parse_reaction_smiles(reaction_smiles: str):
    """
    반응 SMILES 파싱
    
    형식: reactant1.reactant2>reagent>product
    예시: CCO.CC(=O)O>>CCOC(=O)C
    
    Args:
        reaction_smiles: 반응 SMILES 문자열
    
    Returns:
        (reactants, reagents, products)
    """
    
    parts = reaction_smiles.split('>')
    
    if len(parts) != 3:
        return None, None, None
    
    reactants = parts[0].split('.') if parts[0] else []
    reagents = parts[1].split('.') if parts[1] else []
    products = parts[2].split('.') if parts[2] else []
    
    return reactants, reagents, products


def analyze_dataset(data_dir: str = "data/USPTO_50K"):
    """데이터셋 분석"""
    
    print("="*70)
    print("USPTO-50K 데이터셋 분석")
    print("="*70)
    
    data_path = Path(data_dir)
    
    # Train 파일 찾기
    train_files = list(data_path.glob("**/src-train.txt"))
    
    if not train_files:
        print("✗ 데이터 파일을 찾을 수 없습니다.")
        return
    
    train_file = train_files[0]
    
    print(f"\n분석 파일: {train_file.name}")
    print("-"*70)
    
    # 통계
    total_reactions = 0
    valid_reactions = 0
    reactant_counts = []
    product_counts = []
    
    with open(train_file, 'r') as f:
        for line in f:
            total_reactions += 1
            
            reactants, reagents, products = parse_reaction_smiles(line.strip())
            
            if reactants and products:
                valid_reactions += 1
                reactant_counts.append(len(reactants))
                product_counts.append(len(products))
            
            if total_reactions >= 10000:  # 샘플링
                break
    
    print(f"\n총 반응 수: {total_reactions:,}")
    print(f"유효 반응 수: {valid_reactions:,} ({valid_reactions/total_reactions*100:.1f}%)")
    
    if reactant_counts:
        import statistics
        print(f"\n반응물 개수:")
        print(f"  평균: {statistics.mean(reactant_counts):.1f}")
        print(f"  최소: {min(reactant_counts)}")
        print(f"  최대: {max(reactant_counts)}")
        
        print(f"\n생성물 개수:")
        print(f"  평균: {statistics.mean(product_counts):.1f}")
        print(f"  최소: {min(product_counts)}")
        print(f"  최대: {max(product_counts)}")
    
    # 샘플 반응
    print("\n" + "="*70)
    print("샘플 반응")
    print("="*70)
    
    with open(train_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            
            reactants, reagents, products = parse_reaction_smiles(line.strip())
            
            print(f"\n[반응 {i+1}]")
            print(f"반응물: {', '.join(reactants)}")
            if reagents:
                print(f"시약: {', '.join(reagents)}")
            print(f"생성물: {', '.join(products)}")


if __name__ == "__main__":
    # 다운로드
    success = download_uspto_50k()
    
    if success:
        print("\n")
        # 분석
        analyze_dataset()
