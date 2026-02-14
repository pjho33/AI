"""
공식 USPTO 데이터 다운로드
Figshare에서 1M+ 반응 데이터
"""

import urllib.request
import subprocess
from pathlib import Path
import gzip
import json


def download_uspto_official(data_dir: str = "data"):
    """
    공식 USPTO 데이터 다운로드
    
    소스: https://figshare.com/articles/5104873
    크기: ~1GB (압축), ~3GB (압축 해제)
    반응 수: 1,000,000+
    """
    
    print("="*70)
    print("공식 USPTO 데이터 다운로드")
    print("="*70)
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # 파일 경로
    archive_file = data_path / "1976_Sep2016_USPTOgrants_smiles.7z"
    extracted_file = data_path / "1976_Sep2016_USPTOgrants_smiles.rsmi"
    
    # 이미 다운로드되어 있는지 확인
    if extracted_file.exists():
        print(f"✓ 데이터가 이미 존재: {extracted_file}")
        file_size = extracted_file.stat().st_size / 1024 / 1024 / 1024
        print(f"  크기: {file_size:.2f} GB")
        return str(extracted_file)
    
    # 다운로드
    if not archive_file.exists():
        print(f"\n다운로드 중...")
        print(f"  URL: https://ndownloader.figshare.com/files/8664379")
        print(f"  크기: ~1GB (시간이 걸릴 수 있습니다)")
        
        url = "https://ndownloader.figshare.com/files/8664379"
        
        try:
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                mb_downloaded = count * block_size / 1024 / 1024
                mb_total = total_size / 1024 / 1024
                print(f"\r  진행률: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
            
            urllib.request.urlretrieve(url, archive_file, reporthook)
            print("\n✓ 다운로드 완료!")
            
        except Exception as e:
            print(f"\n✗ 다운로드 실패: {e}")
            print("\n대안: 직접 다운로드")
            print(f"  1. 브라우저에서 열기: https://figshare.com/articles/5104873")
            print(f"  2. 파일 다운로드: 1976_Sep2016_USPTOgrants_smiles.7z")
            print(f"  3. 저장 위치: {archive_file}")
            return None
    
    # 압축 해제
    print(f"\n압축 해제 중...")
    
    try:
        # 7z 명령어 사용
        result = subprocess.run(
            ['7z', 'e', str(archive_file), f'-o{data_path}'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ 압축 해제 완료!")
        else:
            print(f"✗ 압축 해제 실패")
            print("\n대안: 수동 압축 해제")
            print(f"  $ 7z e {archive_file} -o{data_path}")
            return None
            
    except FileNotFoundError:
        print("✗ 7z 명령어를 찾을 수 없습니다")
        print("\n설치 방법:")
        print("  $ sudo apt install p7zip-full")
        return None
    
    if extracted_file.exists():
        file_size = extracted_file.stat().st_size / 1024 / 1024 / 1024
        print(f"\n데이터 파일: {extracted_file}")
        print(f"크기: {file_size:.2f} GB")
        return str(extracted_file)
    
    return None


def parse_uspto_rsmi(rsmi_file: str, max_samples: int = 1000):
    """
    USPTO .rsmi 파일 파싱
    
    형식: TSV with header
    Columns: ReactionSmiles, PatentNumber, ParagraphNum, Year, TextMinedYield, CalculatedYield
    
    Args:
        rsmi_file: .rsmi 파일 경로
        max_samples: 최대 샘플 수
    
    Returns:
        List[Dict]: 파싱된 반응 데이터
    """
    
    print(f"\nUSPTO 데이터 파싱 중...")
    print(f"파일: {rsmi_file}")
    
    reactions = []
    
    with open(rsmi_file, 'r') as f:
        # 헤더 스킵
        header = f.readline()
        
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            
            parts = line.strip().split('\t')
            
            if len(parts) < 2:
                continue
            
            reaction_smiles = parts[0]
            patent_number = parts[1] if len(parts) > 1 else ''
            year = parts[3] if len(parts) > 3 else ''
            
            # 반응 SMILES 파싱: reactants>reagents>products
            rxn_parts = reaction_smiles.split('>')
            
            if len(rxn_parts) != 3:
                continue
            
            reactants = rxn_parts[0]
            reagents = rxn_parts[1]
            products = rxn_parts[2]
            
            reactions.append({
                'id': f"{patent_number}_{i}",
                'reactants': reactants,
                'reagents': reagents,
                'products': products,
                'reaction_smiles': reaction_smiles,
                'patent': patent_number,
                'year': year
            })
            
            if (i + 1) % 10000 == 0:
                print(f"  진행: {i+1:,}개 파싱...")
    
    print(f"✓ 파싱 완료: {len(reactions):,}개")
    
    return reactions


def main():
    """USPTO 공식 데이터 다운로드 및 파싱"""
    
    # 다운로드
    rsmi_file = download_uspto_official()
    
    if rsmi_file:
        # 파싱 (일부만)
        print("\n" + "="*70)
        print("데이터 파싱 (샘플 1,000,000개)")
        print("="*70)
        
        reactions = parse_uspto_rsmi(rsmi_file, max_samples=1000000)
        
        # 샘플 출력
        print("\n" + "="*70)
        print("샘플 반응")
        print("="*70)
        
        for i, rxn in enumerate(reactions[:5]):
            print(f"\n[반응 {i+1}] ID: {rxn['id']}")
            print(f"  반응물: {rxn['reactants']}")
            if rxn['reagents']:
                print(f"  시약: {rxn['reagents']}")
            print(f"  생성물: {rxn['products']}")
        
        # JSON 저장
        output_json = Path("data/uspto_official_1m.json")
        with open(output_json, 'w') as f:
            json.dump(reactions, f, indent=2)
        
        print(f"\n✓ JSON 저장: {output_json}")
        print(f"  {len(reactions):,}개 반응")
        
        # 통계
        print("\n" + "="*70)
        print("데이터 통계")
        print("="*70)
        
        reactant_counts = [len(r['reactants'].split('.')) for r in reactions]
        product_counts = [len(r['products'].split('.')) for r in reactions]
        
        import statistics
        print(f"\n반응물 개수:")
        print(f"  평균: {statistics.mean(reactant_counts):.1f}")
        print(f"  최소: {min(reactant_counts)}")
        print(f"  최대: {max(reactant_counts)}")
        
        print(f"\n생성물 개수:")
        print(f"  평균: {statistics.mean(product_counts):.1f}")
        print(f"  최소: {min(product_counts)}")
        print(f"  최대: {max(product_counts)}")


if __name__ == "__main__":
    main()
