import sys

def print_progress(iteration, total, bar_length=40):
    progress = (iteration / total)
    arrow = '=' * int(round(progress * bar_length) - 1)
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\r[{arrow}{spaces}] {iteration}/{total}')
    sys.stdout.flush()