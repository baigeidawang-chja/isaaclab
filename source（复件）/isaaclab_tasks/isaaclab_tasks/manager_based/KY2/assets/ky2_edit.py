# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg

##
# KY2 Actuator Configuration
##

KY2_ACTUATOR_CFG = {
    # 主动齿轮关节：刚度拉满，阻尼匹配，摩擦清零
    "gear_joint": ImplicitActuatorCfg(
        joint_names_expr=["joint_revolute_baselink_gear00"],
        velocity_limit=100.0,    # 解除速度限制
        effort_limit=10000.0,    # 力限制拉满（确保驱动力足够）
        stiffness=100000.0,        # 刚度提高20倍（位置控核心增益）
        damping=200.0,           # 阻尼匹配刚度（防止震荡）
        friction=10.0,            # 清零摩擦，消除阻力
    ),
    # 齿条棱柱关节：刚度更高（棱柱关节需要更大增益）
    "rack_prismatic_joints": ImplicitActuatorCfg(
        joint_names_expr=["joint_prismatic_gear00_rack01", "joint_prismatic_gear00_rack03"],
        velocity_limit=500.0,      # 提高速度限制
        effort_limit=50000.0,    # 棱柱关节力限制拉满
        stiffness=100000.0,        # 刚度提高25倍
        damping=500.0,
        friction=10.0,
    ),
    # 从动齿轮关节：与主动齿轮参数一致
    "slave_gear_revolute_joints": ImplicitActuatorCfg(
        joint_names_expr=[
            "joint_revolute_rack01_gear01", 
            "joint_revolute_rack02_gear02",
            "joint_revolute_rack03_gear03", 
            "joint_revolute_rack04_gear04"
        ],
        velocity_limit=500.0,
        effort_limit=10000.0,
        stiffness=100000.0,
        damping=200.0,
        friction=10.0,
    ),
    # 螺旋桨关节：暂时禁用（减少计算干扰）
    "revolute_joints": DCMotorCfg(
        joint_names_expr=[
            "joint_revolute_servo01_propeller01",
            "joint_revolute_servo02_propeller02",
            "joint_revolute_servo03_propeller03",
            "joint_revolute_servo04_propeller04",
        ],
        saturation_effort=20.0,
        velocity_limit=10.0,
        effort_limit=100.0,
        stiffness=100000.0,  # 禁用驱动
        damping=100.0,
        friction=0.0,
    ),
}

##
# Initial State Configuration (新增关节的初始状态)
##
_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        # 原有关节初始位置
        "joint_revolute_baselink_gear00": 0.0,
        "joint_revolute_servo01_propeller01": 0.0,
        "joint_revolute_servo02_propeller02": 0.0,
        "joint_revolute_servo03_propeller03": 0.0,
        "joint_revolute_servo04_propeller04": 0.0,
        # 新增：齿条棱柱关节初始位置（单位：m，设为0表示初始中位）
        "joint_prismatic_gear00_rack01": 0.0,
        "joint_prismatic_gear00_rack03": 0.0,
        # 新增：从动齿轮旋转关节初始位置（单位：rad，设为0）
        "joint_revolute_rack01_gear01": 0.0,
        "joint_revolute_rack02_gear02": 0.0,
        "joint_revolute_rack03_gear03": 0.0,
        "joint_revolute_rack04_gear04": 0.0,
    },
    joint_vel={
        # 所有关节初始速度设为0
        ".*": 0.0,
    },
)

##
# KY2 Robot Configuration (最终机器人配置，无其他修改)
##
KY2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/chja/myproject/KY201.usd",  # 请根据实际USD文件路径修改
        activate_contact_sensors=False,  # 根据需求设置
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 在环境配置中会禁用重力
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            # max_linear_velocity=100.0,  # m/s
            # max_angular_velocity=100.0,  # rad/s
            max_depenetration_velocity=0.5,  # 防止嵌入
            max_contact_impulse=500.0,  # 限制接触冲量，防止力爆炸
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=24,
            solver_velocity_iteration_count=12,
            sleep_threshold=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=24,
            solver_velocity_iteration_count=12,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
            fix_root_link=False,
        ),
    ),
    init_state=_ZERO_INIT_STATES,
    actuators=KY2_ACTUATOR_CFG,
    soft_joint_pos_limit_factor=0.9,  # 软关节位置限制因子
)