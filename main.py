#!/usr/bin/env python3
"""
NeuroLens+ - Webcam-based Eye Biomarker Platform

Main entry point for running eye tracking tasks.
"""

import argparse
import pygame
from typing import Optional

# IMPORTANT: import tasks directly from modules (do NOT rely on tasks/__init__.py)
from tasks.base import TaskConfig, TaskResult
from tasks.fixation import FixationTask, FixationConfig
from tasks.pursuit import PursuitTask, PursuitConfig
from tasks.saccade import SaccadeTask, SaccadeConfig
from tasks.antisaccade import AntisaccadeTask, AntisaccadeConfig
from tasks.grid9 import Grid9Task, Grid9Config
from tasks.visual_search import VisualSearchTask, VisualSearchConfig

from core.logging import generate_session_id


# Map task keys to their config classes
TASK_CONFIGS = {
    "1": FixationConfig,
    "2": PursuitConfig,
    "3": SaccadeConfig,
    "4": AntisaccadeConfig,
    "5": Grid9Config,
    "6": VisualSearchConfig,
}

# Map task keys to their names and classes
TASKS = {
    "1": ("Fixation Task", FixationTask),
    "2": ("Smooth Pursuit Task", PursuitTask),
    "3": ("Saccade Task", SaccadeTask),
    "4": ("Anti-Saccade Task", AntisaccadeTask),
    "5": ("9-Point Gaze Grid", Grid9Task),
    "6": ("Visual Search Task (Covert Blink)", VisualSearchTask),
}


def run_menu() -> Optional[str]:
    """Run interactive menu for task selection."""
    pygame.init()
    pygame.font.init()

    # Create windowed display for menu
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("NeuroLens+ - Task Selection")

    # Colors
    BG_COLOR = (11, 15, 26)
    TEXT_COLOR = (215, 227, 255)
    HIGHLIGHT_COLOR = (0, 230, 118)

    # Fonts
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 24)

    selected = 0
    running = True
    clock = pygame.time.Clock()

    task_list = list(TASKS.items())  # list of (key, (name, class))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(task_list)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(task_list)

                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    # Return the selected task key
                    chosen_key = task_list[selected][0]
                    pygame.quit()
                    return chosen_key

                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                    pygame.quit()
                    return None

                # Quick select 1-6
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6):
                    key_num = str(event.key - pygame.K_0)
                    if key_num in TASKS:
                        pygame.quit()
                        return key_num

        # Draw
        screen.fill(BG_COLOR)

        # Title
        title = font_large.render("NeuroLens+", True, TEXT_COLOR)
        screen.blit(title, title.get_rect(center=(400, 50)))

        subtitle = font_medium.render("Eye Biomarker Platform", True, (150, 160, 180))
        screen.blit(subtitle, subtitle.get_rect(center=(400, 90)))

        # Task list
        y_start = 150
        for i, (key, (name, _cls)) in enumerate(task_list):
            color = HIGHLIGHT_COLOR if i == selected else TEXT_COLOR
            prefix = "> " if i == selected else "  "
            text = font_medium.render(f"{prefix}{key}. {name}", True, color)
            screen.blit(text, text.get_rect(center=(400, y_start + i * 45)))

        # Instructions
        instructions = [
            "UP/DOWN: Select task",
            "SPACE/ENTER: Start task",
            "1-6: Quick select",
            "Q/ESC: Quit",
        ]
        for i, instr in enumerate(instructions):
            text = font_small.render(instr, True, (100, 110, 130))
            screen.blit(text, text.get_rect(center=(400, 500 + i * 25)))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return None


def run_task(task_key: str, fullscreen: bool = False, session_id: Optional[str] = None) -> Optional[TaskResult]:
    """Run a specific task key."""
    if task_key not in TASKS:
        print(f"Unknown task: {task_key}")
        return None

    task_name, task_class = TASKS[task_key]
    print(f"\nStarting {task_name}...")

    config_class = TASK_CONFIGS.get(task_key, TaskConfig)
    config = config_class(
        session_id=session_id or generate_session_id(),
        fullscreen=fullscreen
    )

    task = task_class(config)

    # IMPORTANT: your task classes should expose .run()
    result = task.run()

    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"Session ID: {config.session_id}")
    print(f"{'='*50}")

    if result and result.success:
        print(f"Valid trials: {result.n_valid_trials}/{result.n_trials} ({result.valid_rate*100:.1f}%)")
        print(f"Quality score: {result.mean_quality_score:.2f}")

        if result.biomarkers:
            print("\nBiomarkers:")
            for name, value in result.biomarkers.items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.3f}")
                else:
                    print(f"  {name}: {value}")

        print("\nOutput files:")
        print(f"  Frame log: {result.frame_log_csv_path}")
        print(f"  Summary:   {result.summary_csv_path}")
        print(f"  Metadata:  {result.session_json_path}")
    else:
        print("Task did not complete successfully.")
        if result and result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")

    if result and result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    return result


def run_all_tasks(fullscreen: bool = False):
    """Run all tasks in sequence under the same session ID."""
    session_id = generate_session_id()
    results = {}

    print(f"\nRunning all tasks with session ID: {session_id}")
    print("=" * 50)

    for key, (name, _cls) in TASKS.items():
        print(f"\n>>> Starting {name}...")
        result = run_task(key, fullscreen=fullscreen, session_id=session_id)
        results[name] = result

        if result and not result.success:
            print(f"Task {name} failed. Continue? (y/n)")
            response = input().strip().lower()
            if response != "y":
                break

    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)

    for name, result in results.items():
        if result:
            status = "PASS" if result.valid_rate >= 0.70 else "FAIL"
            print(f"{name}: {status} ({result.valid_rate*100:.1f}% valid)")
        else:
            print(f"{name}: SKIPPED")

    return results


def main():
    parser = argparse.ArgumentParser(description="NeuroLens+ - Webcam-based Eye Biomarker Platform")

    parser.add_argument(
        "-t", "--task",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        help='Task to run (1-6 or "all")'
    )
    parser.add_argument("-f", "--fullscreen", action="store_true", help="Run in fullscreen mode")
    parser.add_argument("-s", "--session", type=str, help="Session ID (auto-generated if not provided)")
    parser.add_argument("--list", action="store_true", help="List available tasks")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable tasks:")
        for key, (name, _cls) in TASKS.items():
            print(f"  {key}. {name}")
        return

    if args.task:
        if args.task == "all":
            run_all_tasks(fullscreen=args.fullscreen)
        else:
            run_task(args.task, fullscreen=args.fullscreen, session_id=args.session)
    else:
        task_key = run_menu()
        if task_key:
            run_task(task_key, fullscreen=args.fullscreen, session_id=args.session)


if __name__ == "__main__":
    main()
