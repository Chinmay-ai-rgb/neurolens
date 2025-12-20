#!/usr/bin/env python3
"""
NeuroLens+ - Webcam-based Eye Biomarker Platform

Main entry point for running eye tracking tasks.
"""

import sys
import argparse
import pygame
from typing import Optional

from tasks import (
    FixationTask,
    PursuitTask,
    SaccadeTask,
    AntisaccadeTask,
    PLRTask,
    Grid9Task,
    BlinkTask,
)
from tasks.base import TaskConfig, TaskResult
from core.logging import generate_session_id


TASKS = {
    '1': ('Fixation Task', FixationTask),
    '2': ('Smooth Pursuit Task', PursuitTask),
    '3': ('Saccade Task', SaccadeTask),
    '4': ('Anti-Saccade Task', AntisaccadeTask),
    '5': ('Pupillary Light Reflex (PLR)', PLRTask),
    '6': ('9-Point Gaze Grid', Grid9Task),
    '7': ('Blink Rate Monitoring', BlinkTask),
}


def run_menu():
    """Run interactive menu for task selection."""
    pygame.init()
    pygame.font.init()
    
    # Get display info
    info = pygame.display.Info()
    screen_width = info.current_w
    screen_height = info.current_h
    
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
    
    task_list = list(TASKS.items())
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return None
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(task_list)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(task_list)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    pygame.quit()
                    return task_list[selected][0]
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                    pygame.quit()
                    return None
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
                                   pygame.K_5, pygame.K_6, pygame.K_7]:
                    key_num = str(event.key - pygame.K_0)
                    if key_num in TASKS:
                        pygame.quit()
                        return key_num
        
        # Draw
        screen.fill(BG_COLOR)
        
        # Title
        title = font_large.render("NeuroLens+", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(400, 50))
        screen.blit(title, title_rect)
        
        subtitle = font_medium.render("Eye Biomarker Platform", True, (150, 160, 180))
        subtitle_rect = subtitle.get_rect(center=(400, 90))
        screen.blit(subtitle, subtitle_rect)
        
        # Task list
        y_start = 150
        for i, (key, (name, _)) in enumerate(task_list):
            color = HIGHLIGHT_COLOR if i == selected else TEXT_COLOR
            prefix = "> " if i == selected else "  "
            text = font_medium.render(f"{prefix}{key}. {name}", True, color)
            rect = text.get_rect(center=(400, y_start + i * 45))
            screen.blit(text, rect)
        
        # Instructions
        instructions = [
            "UP/DOWN: Select task",
            "SPACE/ENTER: Start task",
            "1-7: Quick select",
            "Q/ESC: Quit"
        ]
        
        for i, instr in enumerate(instructions):
            text = font_small.render(instr, True, (100, 110, 130))
            rect = text.get_rect(center=(400, 500 + i * 25))
            screen.blit(text, rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    return None


def run_task(task_key: str, fullscreen: bool = False, session_id: Optional[str] = None):
    """
    Run a specific task.
    
    Args:
        task_key: Task key ('1'-'7')
        fullscreen: Whether to run in fullscreen mode
        session_id: Optional session ID (auto-generated if not provided)
    
    Returns:
        TaskResult
    """
    if task_key not in TASKS:
        print(f"Unknown task: {task_key}")
        return None
    
    task_name, task_class = TASKS[task_key]
    print(f"\nStarting {task_name}...")
    
    # Create config
    config = TaskConfig(
        session_id=session_id or generate_session_id(),
        fullscreen=fullscreen
    )
    
    # Create and run task
    task = task_class(config)
    result = task.run()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"Session ID: {config.session_id}")
    print(f"{'='*50}")
    
    if result.success:
        print(f"Valid trials: {result.n_valid_trials}/{result.n_trials} ({result.valid_rate*100:.1f}%)")
        print(f"Quality score: {result.mean_quality_score:.2f}")
        
        if result.biomarkers:
            print("\nBiomarkers:")
            for name, value in result.biomarkers.items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.3f}")
                else:
                    print(f"  {name}: {value}")
        
        print(f"\nOutput files:")
        print(f"  Frame log: {result.frame_log_csv_path}")
        print(f"  Summary: {result.summary_csv_path}")
        print(f"  Metadata: {result.session_json_path}")
    else:
        print("Task did not complete successfully")
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    return result


def run_all_tasks(fullscreen: bool = False):
    """Run all tasks in sequence."""
    session_id = generate_session_id()
    results = {}
    
    print(f"\nRunning all tasks with session ID: {session_id}")
    print("="*50)
    
    for key, (name, _) in TASKS.items():
        print(f"\n>>> Starting {name}...")
        result = run_task(key, fullscreen=fullscreen, session_id=session_id)
        results[name] = result
        
        if result and not result.success:
            print(f"Task {name} failed. Continue? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                break
    
    # Summary
    print("\n" + "="*50)
    print("SESSION SUMMARY")
    print("="*50)
    
    for name, result in results.items():
        if result:
            status = "PASS" if result.valid_rate >= 0.70 else "FAIL"
            print(f"{name}: {status} ({result.valid_rate*100:.1f}% valid)")
        else:
            print(f"{name}: SKIPPED")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NeuroLens+ - Webcam-based Eye Biomarker Platform"
    )
    
    parser.add_argument(
        '-t', '--task',
        choices=['1', '2', '3', '4', '5', '6', '7', 'all'],
        help='Task to run (1-7 or "all")'
    )
    
    parser.add_argument(
        '-f', '--fullscreen',
        action='store_true',
        help='Run in fullscreen mode'
    )
    
    parser.add_argument(
        '-s', '--session',
        type=str,
        help='Session ID (auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available tasks'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable tasks:")
        for key, (name, _) in TASKS.items():
            print(f"  {key}. {name}")
        return
    
    if args.task:
        if args.task == 'all':
            run_all_tasks(fullscreen=args.fullscreen)
        else:
            run_task(args.task, fullscreen=args.fullscreen, session_id=args.session)
    else:
        # Interactive menu
        task_key = run_menu()
        if task_key:
            run_task(task_key, fullscreen=args.fullscreen, session_id=args.session)


if __name__ == "__main__":
    main()
