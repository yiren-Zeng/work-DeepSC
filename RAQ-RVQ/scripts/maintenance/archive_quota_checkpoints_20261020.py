"""Archive the 27 checkpoint files approved by the user on 2026-10-20.

Run with --archive to copy, verify SHA-256, and then remove each original.
Run with --restore to return pending intermediate weights; existing targets
are never overwritten. The known incomplete resume file remains archived,
as requested by the user. Use --relative-path to restore a specific weight.
"""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT = Path('/workspace/yi/work/RAQ-RVQ')
SOURCE_ROOT = PROJECT / 'checkpoints'
ARCHIVE_ROOT = Path('/var/tmp/yi/raq-rvq-quota-archive-20261020')
MANIFEST = ARCHIVE_ROOT / 'manifest.json'
REVIEW_MANIFEST = PROJECT / 'experiments/quota_archive_manifest_20261020.json'

APPROVED = (
    ('shiyan_independent_raq_rvq_single_scale_src64_trg2-64_d2_curriculum_rate094_A_patch_ch256_res6-6_mixture_per_k_unet1_ds8_k64', range(20, 200, 20)),
    ('shiyan_independent_raq_rvq_src128-128_trg2-128_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k128', range(20, 200, 20)),
    ('shiyan_independent_raq_rvq_src16-16_trg2-16_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k16', (160, 180)),
    ('shiyan_independent_raq_rvq_src32-32_trg2-32_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k32', (160, 180)),
    ('shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_rate094_A_patch_ch256-512_res6-6_mixture_per_k_unet2_ds8x2_k64', (120, 140, 160, 180)),
)
BROKEN_RESUME = Path('shiyan_independent_raq_rvq_src64-64_trg2-64_d2_uniform_channel1_rate094_A_patch_ch256-512_res6-6_unet2_ds8x2_k64/last_checkpoint.pth')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    stat = path.stat()
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def save_manifest(manifest):
    data = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    for path in (MANIFEST, REVIEW_MANIFEST):
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def verified_move(source, target, expected_hash):
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Refusing to overwrite: {target}')
    original = fingerprint(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'Expected a regular file: {source}')
    if sha256(source) != expected_hash:
        raise ValueError(f'Source hash changed: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    with source.open('rb') as reader, temporary.open('xb') as writer:
        shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if sha256(temporary) != expected_hash or fingerprint(source) != original:
        raise ValueError(f'Copy verification failed; source retained: {source}')
    shutil.copystat(source, temporary)
    os.replace(temporary, target)
    source.unlink()


def archive():
    if ARCHIVE_ROOT.exists() or REVIEW_MANIFEST.exists():
        raise FileExistsError('An archive or review manifest already exists.')
    relatives = [
        Path(directory) / f'epoch_{epoch:04d}_vq_deepsc.pth'
        for directory, epochs in APPROVED for epoch in epochs
    ] + [BROKEN_RESUME]
    if len(relatives) != 27 or len(set(relatives)) != 27:
        raise ValueError('The approved scope must contain exactly 27 files.')
    entries = []
    for relative in relatives:
        source = SOURCE_ROOT / relative
        stat = source.stat()
        if source.is_symlink() or not source.is_file() or stat.st_uid != os.getuid():
            raise ValueError(f'Unexpected source file: {source}')
        entries.append({
            'relative_path': str(relative),
            'source': str(source),
            'archive': str(ARCHIVE_ROOT / relative),
            'size_bytes': stat.st_size,
            'allocated_bytes': stat.st_blocks * 512,
            'original_mtime_ns': stat.st_mtime_ns,
            'status': 'pending',
        })
    total = sum(entry['size_bytes'] for entry in entries)
    if shutil.disk_usage('/var/tmp').free < total + 1024**3:
        raise OSError('Insufficient archive disk space.')
    for index, entry in enumerate(entries, 1):
        print(f'Hashing {index}/27: {entry["relative_path"]}', flush=True)
        entry['sha256'] = sha256(Path(entry['source']))
    ARCHIVE_ROOT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if ARCHIVE_ROOT.parent.is_symlink() or ARCHIVE_ROOT.parent.stat().st_uid != os.getuid():
        raise ValueError('Archive parent must be owned by the current user.')
    ARCHIVE_ROOT.mkdir(mode=0o700)
    manifest = {
        'created_at': now(),
        'authorization': 'User approved temporary archival of the 27 files listed in experiments/quota_cleanup_proposal_20261020.md.',
        'source_root': str(SOURCE_ROOT),
        'archive_root': str(ARCHIVE_ROOT),
        'file_count': len(entries),
        'total_size_bytes': total,
        'total_allocated_bytes': sum(entry['allocated_bytes'] for entry in entries),
        'files': entries,
    }
    save_manifest(manifest)
    for index, entry in enumerate(entries, 1):
        print(f'Archiving {index}/27: {entry["relative_path"]}', flush=True)
        verified_move(Path(entry['source']), Path(entry['archive']), entry['sha256'])
        entry['status'] = 'archived_verified'
        entry['archived_at'] = now()
        save_manifest(manifest)
    manifest['completed_at'] = now()
    save_manifest(manifest)
    print(f'Archive complete: {MANIFEST}', flush=True)


def restore(relative_path):
    manifest = json.loads(MANIFEST.read_text())
    entries = [
        entry for entry in manifest['files']
        if (relative_path is None or entry['relative_path'] == relative_path)
        and entry['status'] == 'archived_verified'
        and Path(entry['relative_path']) != BROKEN_RESUME
    ]
    if not entries:
        raise ValueError('No pending complete checkpoint matches this request.')
    destinations = []
    for entry in entries:
        target = Path(entry['source'])
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Refusing to overwrite: {target}')
        destinations.append((entry, target))
    for index, (entry, target) in enumerate(destinations, 1):
        print(f'Restoring {index}/{len(destinations)}: {target}', flush=True)
        verified_move(Path(entry['archive']), target, entry['sha256'])
        entry['status'] = 'restored_verified'
        entry['restored_to'] = str(target)
        entry['restored_at'] = now()
        save_manifest(manifest)
        print(f'Restored and verified: {target}', flush=True)
    manifest['last_restore_completed_at'] = now()
    manifest['restore_exclusions'] = [{
        'relative_path': str(BROKEN_RESUME),
        'reason': 'User requested leaving the old incomplete checkpoint in the archive.',
    }]
    save_manifest(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--archive', action='store_true')
    action.add_argument('--restore', action='store_true')
    parser.add_argument('--relative-path', help='Restore only this manifest-relative path.')
    args = parser.parse_args()
    if args.archive:
        if args.relative_path is not None:
            parser.error('--relative-path is only supported with --restore')
        archive()
    else:
        restore(args.relative_path)
