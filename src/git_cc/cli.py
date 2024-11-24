import os
import sys
import argparse
import subprocess
import signal
import shutil
from pathlib import Path
import joblib
from colorama import Fore, Style, Back
from InquirerPy import inquirer

LOG_HEADING = f"[GIT-CC]"


def format_info(message):
    return f"{Fore.CYAN}{LOG_HEADING}{Style.RESET_ALL}: {message}"


def format_error(message):
    return f"{Fore.RED}{LOG_HEADING}{Style.RESET_ALL}: {message}"


def format_success(message):
    return f"{Fore.GREEN}{LOG_HEADING}{Style.RESET_ALL}: {message}"


def format_warning(message):
    return f"{Fore.YELLOW}{LOG_HEADING}{Style.RESET_ALL}: {message}"


def signal_handler(sig, frame):
    print(f"\n" + format_error(f"{Fore.RED}Commit process terminated.{Style.RESET_ALL}"))
    exit(0)


def get_model_path():
    """Get the path to the model file from the installed package"""
    try:
        if os.path.exists(Path(__file__).parent / "model" / "custom_classifier.joblib"):
            return str(Path(__file__).parent / "model" / "custom_classifier.joblib")
        elif os.path.exists(Path(__file__).parent / "model" / "classifier.joblib"):
            return str(Path(__file__).parent / "model" / "classifier.joblib")
        else:
            raise FileNotFoundError("Model file not found")
    except Exception as e:
        print(f"Error locating model file: {e}", file=sys.stderr)
        sys.exit(1)


def git_commit(message, commit_type):
    """Create a git commit with the given type and message"""
    formatted_message = f"{commit_type}: {message}"
    try:
        subprocess.run(['git', 'commit', '-m', formatted_message], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error creating commit: {e}", file=sys.stderr)
        sys.exit(1)


def copy_model_file(source_path):
    """Copy the model file from the specified source to the default package location."""
    try:
        # Resolve the source path
        source = Path(source_path).resolve()
        if not source.exists():
            print(f"Error: The specified model file does not exist: {source}", file=sys.stderr)
            sys.exit(1)

        # Ensure the target directory exists
        if os.path.exists(Path(__file__).parent):
            shutil.copy(source, Path(__file__).parent / "model" / "custom_classifier.joblib")
            print(format_success(f"Model successfully loaded"))

    except Exception as e:
        print(format_error(f"Error while copying model file: {e}", file=sys.stderr))
        sys.exit(1)


class CommitClassifier:
    def __init__(self):
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Load the pre-trained model"""
        try:
            model_path = get_model_path()
            self.pipeline = joblib.load(model_path)
        except Exception as e:
            print(format_error(f"Error loading model: {e}", file=sys.stderr))
            sys.exit(1)

    def predict(self, message):
        """Predict commit type for a message"""
        try:
            return self.pipeline.predict([message])[0]
        except Exception as e:
            print(f"Error predicting commit type: {e}", file=sys.stderr)
            return 'chore'  # Default fallback


def main():
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(
        description='Git Conventional Commit Classifier',
        usage='git cc -m <message> [--load <model_path>]'
    )

    parser.add_argument('--message', '-m', help='Commit message')
    parser.add_argument("--load", help="Path to load a custom model", default=None)
    args = parser.parse_args()

    if args.load:
        copy_model_file(args.load)
        sys.exit(0)

    classifier = CommitClassifier()
    commit_type = classifier.predict(args.message)

    try:
        user_input = input(
            format_info(
                f"Do you want to proceed with commit type '{Back.GREEN}{Style.BRIGHT}{commit_type}{Style.RESET_ALL}'? [Y/n]: ")
        ).strip().lower()

        if user_input in ('yes', 'y', ''):  # Accepts Yes, Y, or Enter as confirmation
            git_commit(args.message, commit_type)
            print(format_success(f"{Fore.GREEN}Commit process completed.{Style.RESET_ALL}"))
        else:
            print(format_info(f"{Fore.YELLOW}You chose not to proceed with the predicted type.{Style.RESET_ALL}"))

            # Commit types for selection
            commit_types = ['feat', 'fix', 'chore', 'docs', 'style', 'refactor', 'test', 'perf', 'ci']

            # Interactive selection using InquirerPy
            commit_type = inquirer.select(
                message="Select the commit type:",
                choices=commit_types,
                default=commit_type,
            ).execute()

            print(format_info(f"Selected commit type: {Fore.CYAN}{commit_type}{Style.RESET_ALL}"))

            # Ask for final confirmation after manual selection
            user_input = input(
                format_info(f"Do you want to proceed with commit type '{Fore.GREEN}{commit_type}{Style.RESET_ALL}'? [Yes/No]: ")).strip().lower()
            if user_input in ('yes', 'y', ''):
                git_commit(args.message, commit_type)
                print(format_success(f"{Fore.GREEN}Commit process completed.{Style.RESET_ALL}"))
            else:
                print(format_error(f"{Fore.RED}Commit process terminated.{Style.RESET_ALL}"))

    except KeyboardInterrupt:
        sys.exit(0)

    finally:
        sys.exit(0)


if __name__ == '__main__':
    main()
