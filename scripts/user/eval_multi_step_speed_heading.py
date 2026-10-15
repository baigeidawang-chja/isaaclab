"""Task-specific entry point for CommandFollow CSV evaluation."""

from command_follow_evaluation import run


if __name__ == "__main__":
    run("Isaac-TractionProtect-CommandFollow-MultiStepSpeedHeading-Car4WD-Play-v0", "multi_step_speed_heading")
