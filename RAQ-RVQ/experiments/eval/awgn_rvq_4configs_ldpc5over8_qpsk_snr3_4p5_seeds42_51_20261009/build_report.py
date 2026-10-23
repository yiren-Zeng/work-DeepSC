"""Verify the new RVQ AWGN Monte Carlo results and export dB-only metric tables."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from xml.sax.saxutils import escape
import zipfile

OUTPUT=Path(__file__).resolve().parent
ROOT=OUTPUT.parents[2]
CASES=[
    ('case01_k2-4_k2-4',[[2,4],[2,4]]),
    ('case02_k8-2_k8-2',[[8,2],[8,2]]),
    ('case03_k32-2_k8-8',[[32,2],[8,8]]),
    ('case04_k8-64_k8-2',[[8,64],[8,2]]),
]
SNRS=[3,4.5]
SEEDS=list(range(42,52))


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            hasher.update(block)
    return hasher.hexdigest()


def close(actual,expected,context):
    assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),(context,actual,expected)


def to_db(value):
    return -10*math.log10(1-value)


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def column_name(index):
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result

def write_xlsx(path, sheets):
    """Write numeric Excel cells with standard-library OOXML, without extra dependencies."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        overrides = ''.join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(sheets) + 1)
        )
        archive.writestr('[Content_Types].xml',
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + overrides + '</Types>')
        archive.writestr('_rels/.rels',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>')
        sheet_tags = ''.join(
            f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, (name, _) in enumerate(sheets, 1)
        )
        archive.writestr('xl/workbook.xml',
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets>' + sheet_tags + '</sheets></workbook>')
        relationships = ''.join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheets) + 1)
        )
        archive.writestr('xl/_rels/workbook.xml.rels',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + relationships + f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '</Relationships>')
        archive.writestr('xl/styles.xml',
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000"/></numFmts><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>')
        for i, (_, rows) in enumerate(sheets, 1):
            xml_rows = []
            for r, values in enumerate(rows, 1):
                cells = []
                for c, value in enumerate(values, 1):
                    coordinate = f"{column_name(c)}{r}"
                    if isinstance(value, (int, float)):
                        style = ' s="1"' if isinstance(value, float) else ''
                        cells.append(f'<c r="{coordinate}"{style}><v>{value!r}</v></c>')
                    else:
                        cells.append(f'<c r="{coordinate}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
                xml_rows.append(f'<row r="{r}">' + ''.join(cells) + '</row>')
            archive.writestr(f'xl/worksheets/sheet{i}.xml',
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
                f'<cols><col min="1" max="{len(rows[0])}" width="24" customWidth="1"/></cols>'
                '<sheetData>' + ''.join(xml_rows) + '</sheetData></worksheet>')


def main():
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    status=json.loads((OUTPUT/'status.json').read_text())
    assert all(record['state']=='completed' and record['exit_code']==0 for record in status.values())
    assert digest(manifest['checkpoint'])==manifest['checkpoint_sha256']
    observed_source_changes=[]
    launcher='scripts/eval/test_independent_raq_rvq_src64_64_d2_res6_6_mixture_per_k_combined.sh'
    for path,expected in manifest['source_sha256'].items():
        current=digest(ROOT/path)
        if current==expected:
            continue
        assert path==launcher,path
        content=(ROOT/path).read_text()
        at_launch=content.replace('${SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS:-2,4;2,4}',
                                 '${SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS:-8,64;8,2}')
        at_launch=at_launch.replace('${SNRS:-4.5}','${SNRS:-8}')
        assert hashlib.sha256(at_launch.encode()).hexdigest()==expected,path
        observed_source_changes.append({
            'path':path,'at_launch_sha256':expected,'current_sha256':current,
            'changes':'Only fallback K lists and SNR changed externally during evaluation',
            'effect':'Explicit environment overrides were used; all output parameters are verified',
        })
    for name,expected in manifest['dataset_sha256'].items():
        assert digest(Path(manifest['dataset'])/name)==expected,name
    summaries,seed_rows=[],[]
    output_hashes={}
    for case,k_lists in CASES:
        path=OUTPUT/f'{case}.json'
        report=json.loads(path.read_text())
        assert report['checkpoint']==manifest['checkpoint']
        assert report['num_embeddings_list']==[64,64]
        assert report['rvq_k_lists']==k_lists and report['rvq_depth']==2
        assert report['independent_raq_rvq_enabled']
        assert report['rvq_training_mode']=='trained_independent_codebook_residual'
        assert report['modulation']=='qpsk' and report['ldpc_rate']==0.625
        assert report['stream_packing']=='combined'
        assert report['channel_types']==['awgn'] and report['channel_seeds']==SEEDS
        assert sorted(float(key) for key in report['results'])==SNRS
        output_hashes[str(path)]=digest(path)
        payload_per_image=1024*sum(int(math.log2(k)) for k in k_lists[0])+256*sum(int(math.log2(k)) for k in k_lists[1])
        symbols_per_image=math.ceil(payload_per_image/160)*128
        for snr in SNRS:
            metrics=next(value for key,value in report['results'].items() if float(key)==snr)
            trials=metrics['trials']
            assert len(trials)==10
            assert [trial['channel_seed'] for trial in trials]==SEEDS
            for trial,seed in zip(trials,SEEDS):
                diag=trial['diagnostics']
                assert diag['channel_seed']==seed and diag['channel_type']=='awgn'
                assert diag['stream_packing']=='combined' and diag['modulation']=='qpsk'
                assert diag['ldpc']=={'k':160,'n':256,'rate':0.625}
                assert diag['total']['num_images']==24
                stream=diag['combined_stream']
                assert stream['source_values']==24*3*256*256
                assert stream['payload_bits']==payload_per_image*24
                assert stream['channel_symbols']==symbols_per_image*24
                assert stream['coded_bits']==stream['channel_symbols']*2
                assert stream['ldpc_input_bits']==math.ceil(payload_per_image/160)*160*24
                assert len(diag['per_stage'])==4
                for segment in diag['per_stage']:
                    scale,stage=segment['scale'],segment['stage']
                    assert segment['num_embeddings']==k_lists[scale][stage]
                    assert segment['num_images']==24
                    assert segment['num_indices']==24*(1024 if scale==0 else 256)
                close(to_db(trial['ms_ssim']),trial['ms_ssim_db'],(case,snr,seed,'ms_ssim_db'))
                assert math.isfinite(trial['psnr']) and math.isfinite(trial['ms_ssim_db'])
                seed_rows.append({
                    'scale0_k':str(k_lists[0]),'scale1_k':str(k_lists[1]),
                    'snr_db':snr,'channel_seed':seed,'psnr_db':trial['psnr'],
                    'ms_ssim_db':trial['ms_ssim_db'],'source_json':str(path),
                })
            psnr_mean=statistics.mean(trial['psnr'] for trial in trials)
            raw_mean=statistics.mean(trial['ms_ssim'] for trial in trials)
            ms_db=to_db(raw_mean)
            psnr_std=statistics.stdev(trial['psnr'] for trial in trials)
            close(psnr_mean,metrics['psnr'],(case,snr,'psnr_mean'))
            close(ms_db,metrics['ms_ssim_db'],(case,snr,'ms_ssim_db'))
            mc=metrics['monte_carlo']
            assert mc['num_trials']==10 and mc['channel_seeds']==SEEDS
            close(psnr_mean,mc['psnr_mean'],(case,snr,'mc_psnr_mean'))
            close(psnr_std,mc['psnr_std'],(case,snr,'mc_psnr_std'))
            close(raw_mean,mc['ms_ssim_mean'],(case,snr,'mc_ms_ssim_mean'))
            close(ms_db,mc['ms_ssim_db_from_mean'],(case,snr,'mc_ms_ssim_db'))
            actual_ratio=symbols_per_image/(3*256*256)
            close(actual_ratio,mc['transmission_ratio'],(case,snr,'transmission_ratio'))
            summaries.append({
                'scale0_k':str(k_lists[0]),'scale1_k':str(k_lists[1]),'snr_db':snr,
                'channel':'awgn','ldpc_rate':'5/8','modulation':'QPSK','num_seeds':10,
                'images_per_seed':24,'psnr_mean_db':psnr_mean,'psnr_std_db':psnr_std,
                'ms_ssim_db_from_mean':ms_db,'source_ber':mc['ber'],
                'actual_transmission_ratio':actual_ratio,'actual_ratio_reciprocal':1/actual_ratio,
                'source_json':str(path),
            })
    assert len(summaries)==8 and len(seed_rows)==80
    write_csv(OUTPUT/'summary_db.csv',summaries)
    write_csv(OUTPUT/'per_seed_db.csv',seed_rows)
    labels=[f'{values[0]} / {values[1]}' for _,values in CASES]
    sheets=[]
    for title,metric in [('PSNR_dB','psnr_mean_db'),('MS_SSIM_dB','ms_ssim_db_from_mean')]:
        rows=[['SNR (dB)',*labels]]
        for snr in SNRS:
            rows.append([snr,*[next(row[metric] for row in summaries if row['snr_db']==snr
                and row['scale0_k']==str(values[0]) and row['scale1_k']==str(values[1])) for _,values in CASES]])
        sheets.append((title,rows))
    sheets+=[
        ('Summary',[list(summaries[0]),*[list(row.values()) for row in summaries]]),
        ('Per_seed_dB',[list(seed_rows[0]),*[list(row.values()) for row in seed_rows]]),
        ('Notes',[
            ['Item','Value'],['Channel','AWGN'],['LDPC','160/256 = 5/8'],
            ['Modulation','QPSK'],['Stream packing','combined'],['Seeds','42 43 44 45 46 47 48 49 50 51'],
            ['Dataset',manifest['dataset']],['Images per trial',24],['Checkpoint',manifest['checkpoint']],
            ['Checkpoint SHA256',manifest['checkpoint_sha256']],
            ['MS-SSIM dB','Mean raw MS-SSIM across images and seeds, then -10*log10(1-mean)'],
            ['K ordering','scale 0 two RVQ stages / scale 1 two RVQ stages'],
        ]),
    ]
    write_xlsx(OUTPUT/'AWGN_RAQ_RVQ_4configs_LDPC5over8_QPSK_dB.xlsx',sheets)
    verified={
        'source':'new_gpu_evaluations','num_conditions':8,'num_trials':80,'images_per_trial':24,
        'channel_seeds':SEEDS,'snrs_db':SNRS,'channel':'awgn','ldpc_rate':'5/8','modulation':'qpsk',
        'stream_packing':'combined','checkpoint_sha256':manifest['checkpoint_sha256'],
        'result_sha256':output_hashes,'ms_ssim_db_definition':'-10*log10(1-mean_raw_ms_ssim)',
        'launcher_default_changes_observed':observed_source_changes,
        'results':summaries,
    }
    (OUTPUT/'summary_db.json').write_text(json.dumps(verified,indent=2)+'\n')
    text=[
        '# RAQ-RVQ 四组配置：AWGN、5/8 LDPC、QPSK\n',
        '使用指定原脚本及其默认 mixture-per-K 检查点。每项 SNR 采用 42–51 共 10 个种子，每种子评估 Kodak 24 张图。\n',
        '码本顺序为 scale 0 的两个 RVQ 阶段、scale 1 的两个 RVQ 阶段；combined 打包。\n',
        'PSNR 对逐图 dB 值及种子求平均。MS-SSIM 先对原始值及种子求平均，再按 -10*log10(1-mean) 转成 dB。\n',
        '| scale 0 K | scale 1 K | SNR/dB | PSNR/dB | MS-SSIM/dB | 实际传输比 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for row in summaries:
        text.append(f"| {row['scale0_k']} | {row['scale1_k']} | {row['snr_db']:g} | "
                    f"{row['psnr_mean_db']:.4f} | {row['ms_ssim_db_from_mean']:.4f} | 1/{row['actual_ratio_reciprocal']:g} |")
    text+=['\n实际传输比为复数信道符号数 / RGB 元素数。\n',
           '已核验评估代码与检查点未改变、数据集哈希一致、8 个条件的参数和 80 次完整评估计数正确，'
           '并从逐种子记录重新计算均值。\n',
           '原启动脚本在评估期间仅有默认码本及默认 SNR 的外部更新；本次采用显式环境参数，'
           '四份结果中的实际配置已逐项核对，未受默认值变化影响。\n',
           '[Excel](AWGN_RAQ_RVQ_4configs_LDPC5over8_QPSK_dB.xlsx) · [汇总 CSV](summary_db.csv)'
           ' · [逐种子 dB CSV](per_seed_db.csv) · [运行清单](manifest.json)']
    (OUTPUT/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print('\n'.join(text))


if __name__=='__main__':
    main()
