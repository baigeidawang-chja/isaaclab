from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg

## HOUND actuator config
CAR_ACTUATOR_CFG = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr=["joint_front_right_steer", "joint_front_left_steer"],
        velocity_limit=5.0,
        effort_limit=2.0,
        stiffness=20000.0,   # was 1e8
        damping=200.0,    # ok to start; later tune
        friction=0.05,
    ),
    # "throttle_joints": DCMotorCfg(
    #     joint_names_expr=[
    #         "joint_front_right_wheel_link_wheel",
    #         "joint_front_left_wheel_link_wheel",
    #         "joint_back_right_wheel_link_wheel",
    #         "joint_back_left_wheel_link_wheel",
    #     ],
    #     saturation_effort=10.0,   # was 20
    #     effort_limit=10.0,        # was 20
    #     effort_limit_sim=10.0,
    #     velocity_limit=44.0,     # was 100
    #     velocity_limit_sim=44.0,
    #     stiffness=10.0,
    #     damping=2.0,             # was 10
    #     friction=0.0,
    # ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=[
            "joint_front_right_wheel_link_wheel",
            "joint_front_left_wheel_link_wheel",
            "joint_back_right_wheel_link_wheel",
            "joint_back_left_wheel_link_wheel",
        ],
        saturation_effort=1.05,
        effort_limit=1.0,
        velocity_limit=450.,
        stiffness=0.0,      
        damping=0.5,        
        friction=0.02,
    ),
    }

CAR_SUS_ACTUATOR_CFG = { # 4WD
    **CAR_ACTUATOR_CFG,
    "suspension": ImplicitActuatorCfg(
        joint_names_expr=["joint_front_right_damper",
                          "joint_front_left_damper",
                          "joint_back_right_damper",
                          "joint_back_left_damper"
                          ],
        effort_limit=None, # Passive joint
        velocity_limit=None,
        stiffness=1000.0,
        damping=500.0,
        friction=.5,
    ),
}

_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.2),
    joint_pos={
        # "joint_front_right_lower_sus_wheel_link" : 0.0,
        # "joint_front_left_lower_sus_wheel_link" : 0.0,
        "joint_front_right_steer" : 0.0,
        "joint_front_left_steer" : 0.0,
        "joint_front_right_wheel_link_wheel" : 0.0,
        "joint_front_left_wheel_link_wheel" : 0.0,
        "joint_back_right_wheel_link_wheel" : 0.0,
        "joint_back_left_wheel_link_wheel" : 0.0,
        "joint_front_right_damper": 0.0,
        "joint_front_left_damper": 0.0,
        "joint_back_right_damper": 0.0,
        "joint_back_left_damper": 0.0,      
    },
)

CAR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/chja/myproject/car_no_damper_04.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=10.0, # m/s
            max_angular_velocity=10.0, # deg/s
            max_depenetration_velocity=1.0,  # 降低到 1.0，与障碍物匹配
            max_contact_impulse=500.0,  # 添加限制，防止力爆炸（从 0.0 改为 1000.0）
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=False,
        ),
        activate_contact_sensors=True,
    ),
    init_state=_ZERO_INIT_STATES,
    actuators = CAR_SUS_ACTUATOR_CFG
)

