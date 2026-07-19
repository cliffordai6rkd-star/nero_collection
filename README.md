# Nero Dual-Arm Teleop Collection

这个仓库提供 Nero 松灵 PyAgxArm 的一套主从臂遥操数采入口：一台主臂/遥操臂，一台从臂/执行臂。当前实现的是 CAN 协议下的 `master_slave` 模式；`meta_quest3_vr` 和 `keyboard_3d_mouse` 已在配置层预留，但还没有控制器实现。

## 快速运行

首次配置环境：

```bash
bash setup_env.sh
conda activate nero
```

真实数采 CLI 会在连接机械臂之前自动按照 YAML 的通道和 bitrate 调用
`scripts/setup_can.sh`，需要时终端会提示输入 `sudo` 密码。也可以单独执行：

```bash
bash scripts/setup_can.sh
```

如果只配置某一个接口：

```bash
bash scripts/setup_can.sh can0
```

配置并检查 can0/can1 是否有 CAN 帧：

```bash
bash scripts/check_can_links.sh
```

只检查 can1：

```bash
bash scripts/check_can_links.sh can1
```

`setup_env.sh` 会安装当前项目、`python-can` 和 AgileX `pyAgxArm` SDK。真实机械臂运行前，先确认：

```bash
python -c "import pyAgxArm; print('pyAgxArm OK')"
```

真实机械臂数采。Nero 主从夹爪与各自机械臂共用 can0/can1，不需要独立串口或夹爪 server：

```bash
python -m nero_collection.cli --config configs/master_slave_can.yaml
```

正式启动机械臂前，可以独立检查配置中的两台 V4L2 相机：

```bash
python scripts/check_v4l2_cameras.py \
  --config configs/master_slave_can.yaml \
  --duration 3
```

当前启用两台相机：`side` 是 ROCWARE RC15 `/dev/video2`，`wrist` 是 Orbbec
Dabai DC1 `/dev/video4`。检查脚本不会连接或移动机械臂。USB 重插后应先重新确认
两个设备编号与角色仍然一致。

如果 can0/can1 已由 systemd 或其他进程配置，可以跳过自动步骤：

```bash
python -m nero_collection.cli --config configs/master_slave_can.yaml --skip-can-setup
```

`--backend mock` 会自动跳过 CAN 配置。

主臂模式切换验证：

```bash
python scripts/check_master_modes.py --config configs/master_slave_can.yaml
```

如果需要进一步验证主臂能否 `enable` 或进入 `follower_mode`，显式加参数：

```bash
python scripts/check_master_modes.py --enable
python scripts/check_master_modes.py --include-follower-mode
```

单臂实时打印关节角：

```bash
python scripts/print_arm_q.py --channel can1
```

如果要直接读取配置里的从臂：

```bash
python scripts/print_arm_q.py --config configs/master_slave_can.yaml --role follower
```

检查 Nero 真实关节使能/驱动状态：

```bash
python scripts/check_nero_enable_status.py --channel can0
python scripts/check_nero_enable_status.py --channel can1
```

没有硬件时跑通 H5 写入：

```bash
python -m nero_collection.cli \
  --config configs/master_slave_can.yaml \
  --backend mock \
  --episode-limit 1 \
  --dry-run-duration 2.0
```

当前环境里的 `h5py` 如果报 `numpy.dtype size changed`，说明 numpy/h5py 二进制版本不匹配。建议在采集环境里重装依赖：

```bash
python -m pip install --upgrade --force-reinstall "numpy>=1.23,<3" "h5py>=3.11" PyYAML
```

## 单臂主从模式切换

这里的“主/从模式”对应 PyAgxArm 的 `leader_mode` 和 `follower_mode`：

| 模式 | 用途 | 数采程序中的行为 |
| --- | --- | --- |
| `leader_mode` | 主臂/示教输入 | 读取 `get_leader_joint_angles()`；按 `r` 或 `t` 后进入 |
| `follower_mode` | 受控/保持状态 | 两臂复位时都进入该模式并执行 `move_j()` |

模式切换本身不会让两台臂自动建立跟随关系。正式数采时，程序在每个控制周期读取主臂关节角，根据 `teleop_mapping` 计算从臂目标，经 `joint_step_limit_rad` 限幅后下发给从臂。`control_mode: mit` 使用 `move_mit()` 在固件侧执行关节阻抗控制；`control_mode: position` 使用原来的 `move_js()` 位置控制。

启动时程序先连接两台臂并打印当前角色。当前 V112 配置下 `ctrl_mode=6` 识别为 leader，`ctrl_mode=1` 识别为 follower；状态暂不可用时会尝试读取 leader 专用关节反馈。随后主臂临时切到 follower，两臂使能并共同移动到该 pair 的 `follower.rest_q`。双臂分别通过复位误差检查后保持 follower，直到用户按 `r` 或 `t` 才把主臂切回 leader。

### 单臂模式检查脚本

`scripts/check_master_modes.py` 一次只连接配置中的一台机械臂。`--role leader` 选择当前 pair 的主臂，`--role follower` 选择从臂；它只决定测试哪个 CAN endpoint，不会交换 YAML 中两台臂的角色。

默认执行以下检查流程；它会切换模式，但不会使能机械臂，也不会下发运动指令：

```text
connect
  -> 尝试读取 normal joint q
  -> set_normal_mode
  -> set_leader_mode
  -> 读取 leader joint q
  -> disconnect
```

检查配置中的主臂：

```bash
python scripts/check_master_modes.py \
  --config configs/master_slave_can.yaml \
  --pair main \
  --role leader
```

加上 `--enable` 后，会在 `set_normal_mode` 和 `set_leader_mode` 之间调用 `enable()`。加上 `--include-follower-mode` 后，还会执行一次 `leader_mode -> follower_mode -> leader_mode` 往返切换：

```text
connect
  -> set_normal_mode
  -> enable                         # 仅当指定 --enable
  -> set_leader_mode -> 读取主臂 q
  -> set_follower_mode              # 仅当指定 --include-follower-mode
  -> set_leader_mode -> 再次读取 q
  -> disconnect
```

例如，完整检查配置中的从臂能否使能并往返切换模式：

```bash
python scripts/check_master_modes.py \
  --config configs/master_slave_can.yaml \
  --pair main \
  --role follower \
  --enable \
  --include-follower-mode
```

也可以只切换模式并持续观察单臂关节角：

```bash
# 主臂进入 leader_mode，读取 leader joint q
python scripts/print_arm_q.py \
  --config configs/master_slave_can.yaml \
  --role leader \
  --set-leader-mode \
  --source leader

# 从臂进入 follower_mode，读取 normal joint q；该命令不会 enable 从臂
python scripts/print_arm_q.py \
  --config configs/master_slave_can.yaml \
  --role follower \
  --set-follower-mode \
  --source normal
```

运行模式检查前应停止数采程序，避免两个进程同时占用同一 CAN 机械臂。`--enable` 和主臂上的 `--include-follower-mode` 可能改变机械臂的受力/可拖动状态，只能在工作区清空、急停可用时执行。检查脚本退出时只断开连接，不主动调用 `disable()`。

## 终端交互

启动后程序会：

1. `log.info` 打印机械臂启动、连接和输入设备检查。
2. 打印两臂当前角色，将主臂和从臂都置为 `follower_mode` 并使能。
3. 两臂几乎同时执行 `move_j(follower.rest_q)`，然后分别等待完成并检查平均关节误差。
4. 复位完成后两臂保持 follower，终端等待 `r`、`t` 或 `q`。
5. 按 `t` 把主臂切到 `leader_mode`，建立 teleop reference，并开始双臂和夹爪遥操，但不录制数据。
6. 按 `r` 进入遥操录制；如果已经按过 `t`，则沿用当前 teleop reference，不重复切换模式。
7. 未建立 reference 时，主从误差小于 `pre_teleop_align_error_limit_rad` 后才开始遥操。
8. 遥操过程中按空格停止，随后按 `y` 保存或按 `n` 丢弃。
9. 当 `reset_after_episode: true` 时，无论按 `y` 还是 `n`，两臂都会再次进入 follower 并共同复位。

当 `realtime_plot.enabled: true` 时，机械臂启动完成后会额外打开一个实时窗口。按 `r` 开始录制后，窗口按三列显示从臂 `q_follower`、`tau_follower` 和完整辨识模型的关节残差 `tau_ext = tau_model - tau_follower`。在线计算利用默认 `0.5 s` 固定延迟保留目标点前后的实测 `q`，按标定同款平滑样条解析计算 `dq/ddq`，不使用 V112 被 SDK 置零的速度字段；力矩也在同一窗口内重采样并执行零相位滤波。绘图和逆动力学计算运行在独立进程中，不改变机械臂采样率；关闭窗口后遥操和数采继续运行。

程序运行时的终端提示和日志均为英文。

按 `q` 或 `Ctrl-C` 退出。

## 配置

入口配置在 [configs/master_slave_can.yaml](/home/rei/mnt/code/lcx/Nero_collection/configs/master_slave_can.yaml)。关键字段：

- `teleop.mode`: `master_slave`、`meta_quest3_vr`、`keyboard_3d_mouse` 三种模式名已预留。
- `teleop.protocol`: 当前实现 `can`。
- `teleop.backend`: `pyagxarm` 或 `mock`。
- `teleop.master_slave.arm_pairs`: 主从 pair 列表；默认只需要一个 pair，里面有 `leader` 和 `follower` 两台臂，并在 CAN 下配置 `channel`、`interface`、`bitrate`、`firmware`、`rest_q`。当前官方 `pyAgxArm` Nero 配置主要靠 `channel` 区分 CAN 接口，默认不需要 `can_id`。
- `teleop.command.reset_on_start`: 启动后是否让两臂共同回到 `follower.rest_q` 并分别自检。
- `teleop.command.reset_after_episode`: `y` 保存或 `n` 丢弃后是否共同复位两臂；当前配置为 `true`。
- `teleop.command.reset_interpolation_enabled`: 双臂复位是否使用关节空间线性插值 waypoint。
- `teleop.command.reset_interpolation_rate_hz`: 复位 waypoint 下发频率。
- `teleop.command.reset_joint_speed_rad_s`: 根据最大关节距离计算复位时长的目标速度。
- `teleop.command.reset_min_duration_s`、`reset_max_step_rad`: 最短插值时间和单 waypoint 最大关节步长。
- `teleop.command.role_switch_settle_s`: 模式切换指令发出后、使能和验证前的等待时间。
- `teleop.command.role_switch_timeout_s`: 角色硬件确认超时时间。复位前两臂以及遥操前从臂通过 `ctrl_mode=1` 确认；主臂切到 leader 后必须收到更新的 leader 专用关节帧，否则停止进入遥操。
- `teleop.command.teleop_mapping`: `relative_offset` 或 `absolute`。默认 `relative_offset`，即 `q_cmd = follower_q0 + (leader_q - leader_q0)`，避免主从初始姿态误差造成从臂跳变。
- `teleop.command.control_mode`: `mit` 或 `position`。当前配置使用 `mit`；旧配置未填写时默认 `position`。
- `teleop.command.mit.kp`、`kd`、`v_des`、`t_ff`: MIT 控制器的 7 轴参数，顺序为 `joint1 ... joint7`。控制关系为 `tau_ref = kp*(p_des-q) + kd*(v_des-dq) + t_ff`；`t_ff` 单位为 N.m。
- `teleop.command.pre_teleop_align_enabled`: 按 `r` 后是否先要求主臂手动对齐从臂初始 `q`。
- `teleop.command.pre_teleop_align_error_limit_rad`: 主臂 `q` 与从臂初始 `q` 的最大允许误差。
- `teleop.command.reset_test_sample_time`: 每轮双臂 reset 自检的连续采样次数；当前配置为 5，分别使用均值计算有符号关节误差。
- `teleop.command.reset_error_limit_rad`: 两臂平均 `q` 与 `follower.rest_q` 的最大允许关节误差，单位 rad。超限时按每台臂的误差独立微调复位目标。
- `gripper.teleop_enabled`: 是否在遥操期间读取 can0 主夹爪开合宽度并控制 can1 从夹爪。只控制和记录开合宽度，不记录夹爪力或控制模式。
- `gripper.scale`、`offset_m`: 开合宽度映射 `follower_width = scale * leader_width + offset_m`。
- `gripper.min_width_m`、`max_width_m`: 从夹爪命令行程限制，单位 m。
- `gripper.force_n`: 从夹爪命令力，单位 N。首次实机测试应从较小值开始。
- `gripper.command_rate_hz`、`deadband_m`: 从夹爪 CAN 指令频率和最小开合宽度变化。
- `gripper.keepalive_s`: 主夹爪不变时重复发送从夹爪目标的间隔，避免单次命令丢失后不再恢复。
- `realtime_plot.enabled`: 是否打开实时关节数据窗口；默认 `false`。
- `realtime_plot.window_s`: 滑动时间窗口长度；当前配置为 `10.0` 秒。
- `realtime_plot.update_rate_hz`: 图形刷新频率。它只影响显示，不改变 `teleop.command.sample_rate_hz`。
- `realtime_plot.inverse_dynamics.urdf_path`: 用于计算 `tau_id` 的 Pinocchio 模型。
- `realtime_plot.inverse_dynamics.manifest_path`: 辨识输出的 manifest；实时计算从中加载库仑摩擦、黏性摩擦、关节偏置和库仑速度尺度，并校验其 URDF 与关节顺序。
- `realtime_plot.inverse_dynamics.delay_s`: 估计显示延迟；默认计算并显示 `0.5 s` 前的同时间戳 `q/dq/ddq/tau`。
- `realtime_plot.inverse_dynamics.locked_joint_names`: 从完整 URDF 裁剪夹爪关节，使 Pinocchio 模型只保留七个机械臂关节。
- `realtime_plot.inverse_dynamics.gravity_m_s2`: Pinocchio RNEA 使用的基坐标系重力向量。
- `dynamics_processing.*`: 在线固定延迟计算和 H5 保存共同使用的状态重建、力矩中值窗口与零相位低通参数；当前数值与 `calibration/config.yaml` 的标定预处理一致。
- `cameras[*].backend`: 当前两台真实相机使用 `v4l2`，通过 OpenCV 后台线程抓帧，不阻塞机械臂控制循环。
- `cameras[*].device`: 当前 `side=/dev/video2`、`wrist=/dev/video4`。本机相机的 USB 身份字符串发生过变化，因此使用实际验证过的设备节点。
- `cameras[*].pixel_format`、`width`、`height`、`fps`: 两台均请求 `MJPG 640x480@30`。H5 使用各相机逐帧真实时间戳，不假定两台相机帧率相同或硬件同步。
- `cameras[*].crop`: `[y0,y1,x0,x1]`；`output_size` 是写入 H5 前的 `[width,height]`。OpenCV BGR 会转换为连续内存的 RGB `uint8`。
- `cameras[*].buffer_size`: 设为 `1` 以减少旧帧延迟。每台相机只向数采循环交付最新且尚未交付的帧。
- `robot_states`: 支持 `q`、`velocity`、`acceleration`、`ee_pose`、`torque`、`current`。每项都有 `enabled`、`lowpass`、`lowpass_cutoff_hz` 和 `median_window`。`median_window` 必须是正奇数；torque/current 当前使用 3 点因果中值后接一阶 IIR。

未知的 arm 字段会自动放入 `config_kwargs`，传给 `create_agx_arm_config(...)`，所以如果你们本地 PyAgxArm 需要额外 CAN 参数，可以直接加到对应 arm 配置里。

MIT 模式要求安装的 PyAgxArm 提供 Nero `move_mit()`。启动实机前应清空工作区并确保急停可用；先保持较低 `kp`，确认 7 个关节的方向和阻尼均正确后再逐步增加。程序会校验 SDK 公布的参数范围，但参数在范围内不代表对当前负载一定安全。

如果启动时报 `does not expose Nero move_mit()`，升级 PyAgxArm：

```bash
python -m pip install --upgrade "git+https://github.com/agilexrobotics/pyAgxArm.git"
```

## H5 布局

生成文件遵循 `/home/rei/mnt/code/lcx/data/train_episode/wipe_board/wipe_board` 的风格：

- 根属性：`format=factr_multimodal_episode/v3`、`saved_at_us`
- `/config_yaml`: 本次采集配置原文
- `/teleop/timestamp_us`: 主 pair 从臂关节角观测时间；优先使用 CAN 帧 Unix 时间，缺失时使用该次 getter 返回的主机 Unix 时间；重复反馈帧不重复写入
- `/teleop/q_leader`、`q_follower`、`q_cmd`
- `/teleop/dq_leader`、`dq_follower`
- `/teleop/ddq_leader`、`ddq_follower`
- `/teleop/ee_pose`、`ee_pose_leader`、`cmd_ee_pose`
- `/teleop/tau_leader`、`tau_follower`
- `/teleop/current_leader`、`current_follower`
- `/teleop/q_timestamp_{leader,follower}_us`：SDK 聚合关节角 CAN 帧时间戳
- `/teleop/q_acquired_timestamp_{leader,follower}_us`：关节角 getter 返回时的主机 Unix 时间
- `/teleop/motor_timestamp_{leader,follower}_us`：七轴电机状态 CAN 帧时间戳
- `/teleop/motor_acquired_timestamp_{leader,follower}_us`：七轴 motor getter 各自返回时的主机 Unix 时间
- `/teleop/q_{leader,follower}_raw`、`tau_{leader,follower}_raw`：未拟合、未滤波的 SDK 数据
- `/teleop/dq_{leader,follower}_firmware_raw`、`ddq_{leader,follower}_adapter_raw`：仅用于诊断的固件速度和适配器加速度
- `/teleop/tau_id_follower`、`tau_friction_follower`、`tau_bias_follower`、`tau_model_follower`、`tau_ext_follower`：保存后按完整辨识模型计算的力矩分量与残差
- `/teleop/gripper_state`、`gripper_value`：从夹爪实际开合宽度兼容字段，单位 m
- `/teleop/gripper_leader`、`gripper_follower`、`gripper_cmd`：主夹爪开合宽度、从夹爪实际开合宽度和开合命令，单位均为 m
- `/cameras/<name>/frames` 和 `/cameras/<name>/timestamp_us`，仅在相机实际接入时写入

单个主从 pair 时，`q_leader`、`q_follower`、`q_cmd` 的 shape 是 `(N, 7)`，`ee_pose`、`cmd_ee_pose` 的 shape 是 `(N, 4, 4)`，和 `wipe_board` 样例更接近。代码仍保留多 pair 扩展能力；多 pair 时 joint 会按 `/teleop` 的 `arm_names` 属性顺序拼接。

H5 规范 `q_leader/q_follower` 先按各自有效观测时间拟合平滑样条，再重采样到严格递增的
`teleop/timestamp_us` 主从臂公共时间线；`dq`、`ddq` 分别取该样条的一阶、二阶解析导数。
有效观测时间只在 CAN 时间持续更新时采用 CAN，否则采用对应 getter 的真实主机返回时间，
并在 dataset 属性中记录实际使用了哪些回退，不按样本序号伪造时间。
V112 固件速度和适配器瞬时加速度只写入带 `_raw` 后缀的诊断 dataset，不再作为训练真值。

H5 规范 `tau_leader/tau_follower` 先按每轴电机 CAN 时间戳重采样到同一时间线，再执行
3 点中值和四阶 Butterworth 零相位低通。录制期间的因果滤波只用于兼容预览，不会覆盖
保存阶段使用的原始数据。某轴 motor CAN 时间戳缺失、倒退或不再更新时，该轴使用自身
motor getter 的主机返回时间，原始 SDK 时间戳仍保留。每个规范和 raw dataset 都写入处理方法及源时间戳属性。

实时 `tau_ext` 仅用于可视化，目前不写入 H5。固定延迟窗口先获得真实 `dq/ddq` 和对齐后的
`tau`；配置 manifest 后，估计器按离线辨识的
同一模型计算 `tau_model = RNEA(q,dq,ddq) + tau_c*tanh(dq/v_s) + tau_v*dq + bias`，
再计算 `tau_ext = tau_model - tau_follower`；不执行雅可比伪逆或最小二乘。未配置
`manifest_path` 时保持兼容，只使用 RNEA，此时摩擦和力矩偏置仍会进入残差。

## 真实 SDK 接入点

- PyAgxArm 适配器：[nero_collection/arms/pyagx.py](/home/rei/mnt/code/lcx/Nero_collection/nero_collection/arms/pyagx.py)
- 主从控制逻辑：[nero_collection/teleop/master_slave.py](/home/rei/mnt/code/lcx/Nero_collection/nero_collection/teleop/master_slave.py)
- H5 写入：[nero_collection/h5_writer.py](/home/rei/mnt/code/lcx/Nero_collection/nero_collection/h5_writer.py)
- V4L2 相机入口：[nero_collection/cameras.py](/home/rei/mnt/code/lcx/Nero_collection/nero_collection/cameras.py)
