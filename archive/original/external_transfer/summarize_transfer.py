"""Format all predefined transfer results; never select or refit a policy."""
from pathlib import Path
import csv,json
OUT=Path(__file__).resolve().parents[1]
with (OUT/'results/pooled_metrics.csv').open() as f: data=list(csv.DictReader(f))
methods=list(dict.fromkeys(r['method'] for r in data));summary=[]
for m in methods:
    values=[r for r in data if r['method']==m];assert len(values)==5
    row={'method':m}
    for metric in ['macro_f1','malicious_recall','false_positive_rate','fp','fn','corrected','introduced','switches']:
        v=[float(r[metric]) for r in values]
        row[metric+'_min']=min(v);row[metric+'_max']=max(v)
    summary.append(row)
with (OUT/'results/bundle_range_summary.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
def interval(r,k,percent=False):
    scale=100 if percent else 1
    if percent:return f"{scale*r[k+'_min']:.3f}–{scale*r[k+'_max']:.3f}"
    return f"{r[k+'_min']:,.0f}–{r[k+'_max']:,.0f}"
lines=['# 冻结策略的 IoT-23 迁移结果（独立协议）','',
       '实际完成：7 个预先合格场景的 1,159,418 个作者原生流；465,990 良性、693,428 恶意。每个流应用同样的 5 套 USTC 冻结模型和 11 种原有策略，没有训练、重新校准或选阈值。385 行场景指标与 55 行人口合并指标全部保留。','',
       '## 所有预定策略','',
       '下表是五套冻结配置的最小—最大值，不是置信区间，也不是选择最佳配置。每行各指标端点不一定来自同一配置；精确配对见 pooled_metrics.csv。Macro-F1 对两类混淆计数计算；各单类场景的 Macro-F1 记为缺失，不取伪平均。','',
       '| 策略 | Macro-F1 (%) | 恶意召回 (%) | 误报率 (%) | 纠错 C | 新增错误 D | 切换数 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append('| '+r['method']+' | '+' | '.join(interval(r,k,p) for k,p in [('macro_f1',True),('malicious_recall',True),('false_positive_rate',True),('corrected',False),('introduced',False),('switches',False)])+' |')
lines+=['','## 结果判断','',
        '全部 55 个配置—策略组合的 Macro-F1 为 22.463%–40.554%，误报率为 92.811%–100%。高恶意召回在这里伴随大量良性误报，不能单独当作有效检测。没有获得冻结策略直接跨来源可用的证据。','',
        '同源配置 1 上，Fixed 相对 Temporal 纠正 431,664 条、引入 28,770 条错误，但仍有 461,231 个误报（误报率 98.979%）。这是同一困难外部人口上的局部纠错，不是可部署性能。其余配置的正、零、负净纠错亦完整保留。','',
        'Global4 在配置 1 冻结为 1.01，即不切换；不能因这个外部结果差而重新选阈值。两个学习策略在不同配置上并未呈现相对简单规则的一致优势。','',
        '## 可解释范围','',
        '- 这是来源和样本单位同时变化的迁移：IoT-23 作者流，对比原 USTC 的60秒空闲分段。它不是同分段协议的外部复现，也不合并40k/625,523段分数。',
        '- 标签与结构此前已为修复核查而接触；不得称全新盲测。7个场景、5套拟合配置和百万流都不是相同数量的独立活动。共享CTU实验室与相关谱系限制仍在。',
        '- 输入保持原Temporal40前min(n,64)包、Stats8完整流、首实际报文源方向、wire长度、原短流处理；只转换资格已冻结的作者流，没有按分数或长度筛掉样本。',
        '- 每个合格流的全部报文摘要、数量、原包头已实际读回核验。独立实现抽取33流/3,375包逐元素复核40+8特征；它不是全量第二次独立特征重解析。',
        '- 当前结果不能把失败归因于捷径、短流、采集差异或仲裁选择中的某一个因素；这些因素没有在此被单独控制。',
        '- 不把迁移失败包装为新机制优势或“百万级验证”；其作用是明确限制主稿受控发现的适用范围。','',
        '## 执行与来源','',
        '原2小时试跑门槛失败记录保留；用户随后明确授权最多3小时。修正WinAPI内存句柄与评分前标签摘要检查，保留脚本旧版及协议修订链。正式运行未改模型、网格、输入或人口。小样本试跑包含显著首调用开销，保守估时不代表实际全量耗时；本次共享特征推理不是端到端策略效率基准。','',
        '来源合同、全部模型摘要、原开发选择记录见 protocol_lock.json / protocol_amendment_001.json。标签来源继承 ../mad_etd_icassp2027_external_repair_001/source_manifest.csv 和 label_contract.json；公共作者来源说明：https://www.stratosphereips.org/datasets-iot23 。本地镜像文件各自钉住摘要，未声称整包与Zenodo发布二进制一致。','',
        '结果：results/scenario_metrics.csv；results/pooled_metrics.csv；results/scenario_contrasts.csv；results/pooled_contrasts.csv。逐流预测：predictions/<scenario>/fold_<0..4>.npz（sample_id、概率、仲裁输出、11列分类）。所有模型输出使用同一已冻结样本集合。','',
        '未新增下载、安装、付费、发布或投稿；旧R1—R4、受控回放、PCAP结果及upgrade023均未改写。']
with (OUT/'RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
with (OUT/'results/summary_receipt.json').open('x') as f:
    json.dump({'rows':len(data),'policies':len(methods),'range_type':'descriptive over all 5 dependent source bundles, not CI or ranking','all_results_included':True},f,indent=2)
print('All 55 pooled results retained in the 11-policy descriptive summary.')
