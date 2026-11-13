四组 rollout case study 数据
==========================

index.json 是总索引；其中所有相对路径均以本目录为根。
cases/<case>/case.json 包含来源 A、两个 B 的加载/不加载对照、普通后续复用和 reward 汇总。
episodes/<attempt>/ 保留完整环境动作/观察、原始 Native 事件、实际输入、结果和 token 文本。
原始 solver 的 trajectory.npz 和原始 curate replay 中保留 token、mask、behavior logprob。
冻结评估器的数值轨迹位于 pair_*/with_skill、without_skill/native-capture.json；其 logprob 缺失时保留原值。
当前展开 22 条核心轨迹；raw_evidence.tar.gz 保留全部 40 条，包括额外的 GRPO 同组轨迹及原始证明文件。
copied_files_manifest.json 记录逐文件来源及 SHA256；original_files_manifest.json 对应完整原始包。

这些是观察结果后选出的机制案例，不是完整测试集统计。普通复用 episode 可以同时加载多个 skill，
其成功不能单独归因给其中一条。utility 来自受控配对；原始任务 reward 与 curate advantage 分开。
统计的是环境动作步数，不等于模型 token 数或端到端 wall time。原始退出码与宿主终止状态未改写。
