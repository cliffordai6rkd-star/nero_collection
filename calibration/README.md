python -m calibration.collect_static \
  --config calibration/config.yaml \
  --output calibration/data/static_baseline.npz

python -m calibration.fit_static \
  --config calibration/config.yaml \
  --data calibration/data/static_baseline.npz \
  --output calibration/results/static_fit.yaml

python -m calibration.validate_internal \
  --config calibration/config.yaml \
  --fit calibration/results/static_fit.yaml \
  --data calibration/data/static_baseline.npz \
  --output calibration/results/internal_validation.yaml

## 数据格式

NPZ 中包含：

- `q`: `(N, 7)` 静态关节角；
- `tau`: `(N, 7)` PyAgxArm 反馈力矩；
- `current`: `(N, 7)` 电流，SDK 不提供时允许为 NaN；
- `timestamp_us`: `(N,)` 采样时间；
- `pose_index`、`pose_names`: 样本与目标姿态的映射；
- `round_index`: 样本所属采集轮次；
- `metadata_json`: 配置、机械臂和数据标签。
