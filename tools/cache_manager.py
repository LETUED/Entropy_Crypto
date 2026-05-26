"""
캐시 관리 도구

사용법:
  py tools/cache_manager.py           # 캐시 현황 조회
  py tools/cache_manager.py --clean   # 중복/오래된 캐시 정리 (확인 후 삭제)
  py tools/cache_manager.py --clear   # 전체 MPE 캐시 삭제 (raw 데이터는 유지)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def show_status():
    print("\n" + "=" * 60)
    print("  캐시 현황")
    print("=" * 60)

    # raw 데이터
    raw_files = sorted(RAW_DIR.glob("*.parquet")) if RAW_DIR.exists() else []
    raw_size  = sum(f.stat().st_size for f in raw_files) / 1024 / 1024
    print(f"\n[raw 데이터]  {RAW_DIR}")
    print(f"  파일 수: {len(raw_files)}개  |  크기: {raw_size:.1f} MB")
    for f in raw_files:
        sz = f.stat().st_size / 1024
        print(f"  {f.name:<55} {sz:>7.1f} KB")

    # MPE 캐시
    cache_files = sorted(CACHE_DIR.glob("mpe_*.parquet")) if CACHE_DIR.exists() else []
    cache_size  = sum(f.stat().st_size for f in cache_files) / 1024 / 1024
    print(f"\n[MPE 캐시]  {CACHE_DIR}")
    print(f"  파일 수: {len(cache_files)}개  |  크기: {cache_size:.1f} MB")
    for f in cache_files:
        sz  = f.stat().st_size / 1024
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {f.name:<70} {sz:>7.1f} KB  ({mtime})")

    print(f"\n  합계: {raw_size + cache_size:.1f} MB")


def clean_cache(dry_run: bool = True):
    """동일 코인+기간에 대해 파라미터가 다른 구버전 캐시 정리"""
    if not CACHE_DIR.exists():
        print("캐시 폴더 없음.")
        return

    cache_files = list(CACHE_DIR.glob("mpe_*.parquet"))
    if not cache_files:
        print("MPE 캐시 파일 없음.")
        return

    # 코인+기간 기준으로 그룹핑
    from collections import defaultdict
    groups = defaultdict(list)
    for f in cache_files:
        # mpe_{sym}_{interval}_{start}_{end}_w{...}.parquet
        parts = f.stem.split("_w")[0]  # 파라미터 앞부분 = 코인+기간
        groups[parts].append(f)

    to_delete = []
    for key, files in groups.items():
        if len(files) > 1:
            # 가장 최근 파일 유지, 나머지 삭제 후보
            files_sorted = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
            print(f"\n  {key}: {len(files)}개 캐시 발견")
            print(f"    유지: {files_sorted[0].name}")
            for f in files_sorted[1:]:
                print(f"    삭제: {f.name}")
                to_delete.append(f)

    if not to_delete:
        print("정리할 중복 캐시 없음.")
        return

    if dry_run:
        print(f"\n  (dry-run) 삭제 예정 {len(to_delete)}개. --clean 플래그로 실제 삭제.")
    else:
        for f in to_delete:
            f.unlink()
            print(f"  삭제: {f.name}")
        print(f"\n  {len(to_delete)}개 캐시 삭제 완료.")


def clear_all_mpe():
    if not CACHE_DIR.exists():
        print("캐시 폴더 없음.")
        return
    files = list(CACHE_DIR.glob("mpe_*.parquet"))
    if not files:
        print("MPE 캐시 파일 없음.")
        return

    print(f"MPE 캐시 {len(files)}개 삭제합니다. 계속? [y/N] ", end="")
    ans = input().strip().lower()
    if ans != "y":
        print("취소.")
        return
    for f in files:
        f.unlink()
        print(f"  삭제: {f.name}")
    print(f"\n{len(files)}개 삭제 완료. (raw 데이터는 유지됨)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="캐시 관리 도구")
    parser.add_argument("--clean", action="store_true", help="중복 캐시 정리")
    parser.add_argument("--clear", action="store_true", help="전체 MPE 캐시 삭제")
    args = parser.parse_args()

    if args.clear:
        clear_all_mpe()
    elif args.clean:
        clean_cache(dry_run=False)
    else:
        show_status()
        clean_cache(dry_run=True)
